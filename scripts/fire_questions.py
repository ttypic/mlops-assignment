#!/usr/bin/env python3
"""Fire N questions through the agent to populate Langfuse traces (Phase 4).

Each request is tagged with metadata (phase + source + db) so the traces are
filterable in the Langfuse UI - you'll reuse these tags in Phase 6. The summary
flags which runs triggered a revise so you can open one of those for the
verify->revise waterfall screenshot.

Run (agent server must be up on :8001):
    uv run python scripts/fire_questions.py            # 10 questions
    uv run python scripts/fire_questions.py --n 20 --tag run=smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


def load_questions(n: int) -> list[dict]:
    if not EVAL_FILE.exists():
        raise SystemExit(f"{EVAL_FILE} not found - run scripts/load_data.py first")
    rows = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of questions to fire")
    ap.add_argument("--url", default=AGENT_URL_DEFAULT)
    ap.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra metadata tag (repeatable), e.g. --tag run=smoke",
    )
    args = ap.parse_args()

    # Tags attached to every trace. db is added server-side per request.
    tags = {"phase": "phase4", "source": "eval_set"}
    for kv in args.tag:
        k, _, v = kv.partition("=")
        tags[k] = v

    questions = load_questions(args.n)
    revised = 0
    failed = 0
    for i, q in enumerate(questions, 1):
        payload = {"question": q["question"], "db": q["db_id"], "tags": tags}
        try:
            resp = requests.post(args.url, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{i:2}/{len(questions)}] {q['db_id']:>20}  ERROR: {type(e).__name__}: {e}")
            continue

        iters = data.get("iterations", 0)
        did_revise = any(h.get("node") == "revise" for h in data.get("history", []))
        revised += did_revise
        flag = "  <-- REVISED" if did_revise else ""
        ok = "ok" if data.get("ok") else f"fail({data.get('error')})"
        print(f"[{i:2}/{len(questions)}] {q['db_id']:>20}  iters={iters} {ok}{flag}")

    print(
        f"\nDone: {len(questions)} fired, {revised} triggered a revise, {failed} request errors.\n"
        "Open Langfuse (http://localhost:3001) -> Traces. The 'phase:phase4' tag is "
        "visible in the list; open a REVISED trace for the generate_sql/verify/revise waterfall."
    )


if __name__ == "__main__":
    main()
