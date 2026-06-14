"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def _iteration_sqls(history: list[dict]) -> list[str]:
    """Pull the SQL produced at each iteration, in order.

    generate_sql (iter 0) and each revise (iter 1, 2, ...) append an entry with
    a "sql" field to the agent's history. The verify entries don't, so filtering
    on the presence of "sql" recovers exactly the per-iteration attempts.
    """
    return [h["sql"] for h in history if h.get("node") in ("generate_sql", "revise") and "sql" in h]


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness.

    Execution accuracy: run BOTH the agent's SQL and the gold SQL against the
    target DB and compare canonicalized row sets. We score the SQL at *every*
    iteration the agent emitted (from history), so summarize() can report the
    pass rate as if we'd stopped after iter 0, iter 1, etc.
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    # Gold rows are the reference; compute once. A broken gold query just means
    # nothing can match it (matches() returns False on None), which we surface.
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    record: dict = {
        "db_id": db_id,
        "question": question["question"],
        "gold_sql": gold_sql,
        "gold_ok": gold_ok,
        "gold_error": gold_err,
        "pred_sql": "",
        "n_iterations": 0,
        "per_iter": [],
        "final_correct": False,
        "agent_ok": False,
        "error": None,
    }

    # Call the agent over HTTP. A request failure is a non-fatal eval outcome:
    # record it and move on (counts as incorrect, 0 iterations).
    try:
        resp = httpx.post(
            agent_url,
            json={"question": question["question"], "db": db_id,
                  "tags": {"phase": "phase5", "run": "baseline"}},
            timeout=180.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        record["error"] = f"{type(e).__name__}: {e}"
        return record

    record["agent_ok"] = bool(data.get("ok"))
    record["pred_sql"] = data.get("sql", "")
    if data.get("error"):
        record["error"] = data["error"]

    sqls = _iteration_sqls(data.get("history", []))
    record["n_iterations"] = len(sqls)

    # Score each iteration's SQL by executed rows against gold.
    per_iter: list[bool] = []
    for sql in sqls:
        _ok, pred_rows, _err = run_sql(db_id, sql)
        per_iter.append(matches(gold_rows, pred_rows))
    record["per_iter"] = per_iter
    record["final_correct"] = per_iter[-1] if per_iter else False
    return record


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"n_questions": 0, "overall_pass_rate": 0.0, "mean_iterations": 0.0,
                "pass_at_iter": {}, "n_request_errors": 0}

    # Number of iteration slots to report: the deepest any question went
    # (at least 1 so we always emit iter 0).
    max_iters = max((len(r["per_iter"]) for r in results), default=0)
    max_iters = max(max_iters, 1)

    def result_at(per_iter: list[bool], k: int) -> bool:
        # Carry forward the last emitted attempt; failed runs (empty) are False.
        if not per_iter:
            return False
        return per_iter[k] if k < len(per_iter) else per_iter[-1]

    pass_at_iter = {
        str(k): round(sum(result_at(r["per_iter"], k) for r in results) / n, 4)
        for k in range(max_iters)
    }

    overall = sum(r["final_correct"] for r in results) / n
    mean_iters = sum(r["n_iterations"] for r in results) / n
    n_errors = sum(1 for r in results if r.get("error") and not r["per_iter"])
    # How many questions the loop actually revised (>1 attempt).
    n_revised = sum(1 for r in results if r["n_iterations"] > 1)

    return {
        "n_questions": n,
        "overall_pass_rate": round(overall, 4),
        "mean_iterations": round(mean_iters, 4),
        "pass_at_iter": pass_at_iter,
        "n_revised": n_revised,
        "n_request_errors": n_errors,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
