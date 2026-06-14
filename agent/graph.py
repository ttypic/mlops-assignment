"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = 3

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    # Set when an LLM/infra call fails (e.g. vLLM 400 context-length, 5xx,
    # timeout, connection error). Terminates the loop and is surfaced by the
    # server as a clean ok=false result instead of crashing the request.
    error: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


@lru_cache(maxsize=1)
def llm() -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default).

    Cached so the whole process shares ONE client (and one underlying httpx
    connection pool). Rebuilding it per call - 2-3x per request, thousands of
    requests under load - churns TCP/TLS connections and is a likely source of
    HTTP errors at high RPS (Phase 6, H2). The client is concurrency-safe and
    reused across all in-flight async requests.
    """
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def _parse_verify_json(text: str) -> dict:
    """Pull the {"ok": ..., "issue": ...} object out of a verifier reply.

    The model may wrap the JSON in prose or fences, so grab the first balanced
    {...} block and parse it. On any failure we fail *open* (ok=True) rather
    than spuriously triggering a revise on a parse hiccup.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return {
                "ok": bool(obj.get("ok", True)),
                "issue": str(obj.get("issue", "") or ""),
            }
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"ok": True, "issue": ""}


async def _safe_ainvoke(messages: list) -> tuple[str | None, str | None]:
    """Call the LLM, returning (content, None) on success or (None, error) on
    ANY failure (vLLM 400 context-length, 5xx, timeout, connection error, ...).

    This is the resilience boundary: a single bad call degrades that node to a
    graceful failure (surfaced as ok=false) instead of bubbling up and 500-ing
    the whole request. We deliberately catch broadly - the root cause varies
    and the agent's job is to not crash regardless.
    """
    try:
        response = await llm().ainvoke(messages)
        return response.content, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


async def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    Async + `ainvoke` so the event loop can have many agent runs in flight at
    once (Phase 6, H1): a single process issues concurrent vLLM calls that the
    engine batches, instead of one blocking thread per request.
    """
    content, err = await _safe_ainvoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    if err is not None:
        # No SQL to run. Terminate the loop (route_after_verify checks `error`)
        # and let the server report it as a clean failure.
        return {
            "sql": "",
            "error": err,
            "iteration": state.iteration + 1,
            "history": state.history + [{"node": "generate_sql", "error": err}],
        }
    sql = _extract_sql(content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


async def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Follow the generate_sql_node pattern: build messages from the VERIFY_*
    prompts, call llm(), parse the reply. Ask the model for a small JSON object
    like {"ok": bool, "issue": str} and parse it defensively - the model may
    wrap it in prose or fences. state.execution.render() gives you a compact
    view of the rows or error to feed into the prompt.

    Return: {"verify_ok": <bool>, "verify_issue": <str>}.
    What counts as "not plausible" is yours to define - see the Phase 3 targets
    in the README.
    """
    # An LLM/infra error upstream (generate or revise) means there's nothing
    # worth re-verifying or revising. End the loop cleanly (verify_ok=True so
    # the router terminates); the server surfaces state.error.
    if state.error:
        return {
            "verify_ok": True,
            "verify_issue": state.error,
            "history": state.history + [{"node": "verify", "skipped": True, "error": state.error}],
        }

    execution = state.execution
    result_text = execution.render() if execution is not None else "ERROR: no result"

    # Fast path: a hard SQL error is unambiguously not plausible. Short-circuit
    # to a revise without spending an LLM call - the error text is the issue.
    if execution is None or not execution.ok:
        issue = execution.error if execution is not None else "no execution result"
        verdict = {"ok": False, "issue": issue or "SQL execution failed"}
    else:
        content, err = await _safe_ainvoke([
            ("system", prompts.VERIFY_SYSTEM),
            ("user", prompts.VERIFY_USER.format(
                question=state.question,
                sql=state.sql,
                result=result_text,
            )),
        ])
        if err is not None:
            # We already have an executed, non-empty result; we just can't judge
            # it. Accept it (fail-open) rather than failing the request - serving
            # a plausible answer beats a 500 because the verifier hiccuped.
            verdict = {"ok": True, "issue": f"verify skipped: {err}"}
        else:
            verdict = _parse_verify_json(content)

    return {
        "verify_ok": verdict["ok"],
        "verify_issue": verdict["issue"],
        "history": state.history + [{
            "node": "verify",
            "ok": verdict["ok"],
            "issue": verdict["issue"],
        }],
    }


async def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt should include the failing
    SQL, its execution result, and the verifier's complaint so the model can fix
    it. Bump the iteration counter the same way generate_sql_node does so the
    loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    execution = state.execution
    result_text = execution.render() if execution is not None else "ERROR: no result"
    content, err = await _safe_ainvoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            result=result_text,
            issue=state.verify_issue,
        )),
    ])
    if err is not None:
        # Keep the prior SQL (it may still be the best we have) and end the
        # loop via state.error; don't blank it or spin further failing calls.
        return {
            "error": err,
            "iteration": state.iteration + 1,
            "history": state.history + [{"node": "revise", "error": err}],
        }
    sql = _extract_sql(content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{
            "node": "revise",
            "sql": sql,
            "fixing_issue": state.verify_issue,
        }],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). We also end on a
    captured LLM/infra error - revising won't help if the model call itself
    failed. Otherwise, revise.
    """
    if state.error or state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
