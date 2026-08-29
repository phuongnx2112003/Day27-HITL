"""LangGraph workflow for customer churn-risk decisions with HITL review."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from openai import BadRequestError, OpenAI
from dotenv import load_dotenv

from models import AuditEntry, GraphState


load_dotenv()


CONFIDENCE_THRESHOLD = 0.85
AGENT_ID = "churn-risk-agent"
HIGH_RISK_ACTION = "increase_credit_limit"
LOW_RISK_ACTION = "send_email"
AUDIT_LOG_PATH = Path(__file__).with_name("audit_log.json")

CUSTOMERS: dict[str, dict[str, float]] = {
    "CUST001": {"toi": 100_000_000, "churn_probability": 0.35},
    "CUST002": {"toi": 50_000_000, "churn_probability": 0.85},
    "CUST003": {"toi": 80_000_000, "churn_probability": 0.65},
}


def _customer_profile(customer_id: str) -> dict[str, float]:
    return CUSTOMERS.get(customer_id, CUSTOMERS["CUST001"])


def _llm_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    kwargs: dict[str, str] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs, timeout=30.0, max_retries=2)


def _call_llm(customer_id: str, profile: dict[str, float]) -> dict[str, Any]:
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set")

    client = _llm_client()
    messages = [
            {
                "role": "system",
                "content": (
                    "You are a customer churn-risk analyst. Return only valid JSON with "
                    "proposed_action, confidence_score, and reasoning. "
                    "Allowed actions are send_email and increase_credit_limit. "
                    "confidence_score must be a number from 0 to 1."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "customer_id": customer_id,
                        "total_operating_income": profile["toi"],
                        "churn_probability": profile["churn_probability"],
                    }
                ),
            },
        ]
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except BadRequestError as exc:
        # Some OpenAI-compatible providers reject response_format.
        if "response_format" not in str(exc).lower() and "json" not in str(exc).lower():
            raise
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
        )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned an empty response")
    result = _parse_llm_json(content)

    action = result.get("proposed_action")
    confidence = result.get("confidence_score")
    reasoning = result.get("reasoning")
    if action not in {LOW_RISK_ACTION, HIGH_RISK_ACTION}:
        raise ValueError(f"Unsupported LLM action: {action!r}")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("LLM confidence_score must be between 0 and 1")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("LLM reasoning must be a non-empty string")

    return {
        "proposed_action": action,
        "confidence_score": float(confidence),
        "reasoning": reasoning.strip(),
    }


def _parse_llm_json(content: str) -> dict[str, Any]:
    """Parse strict JSON and common model variants such as Markdown fences."""

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some models prepend a sentence or append a short explanation.
        # Decode the first complete JSON object from the response.
        decoder = json.JSONDecoder()
        result = None
        for index, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                result = candidate
                break
        if result is None:
            raise RuntimeError("LLM did not return valid JSON")

    if not isinstance(result, dict):
        raise RuntimeError("LLM JSON response must be an object")
    return result


def evaluate_customer(state: GraphState) -> dict[str, Any]:
    """Ask the configured OpenAI-compatible model to evaluate a customer."""

    customer_id = state["customer_id"]
    profile = _customer_profile(customer_id)
    churn_probability = profile["churn_probability"]

    result = _call_llm(customer_id, profile)

    return {
        **result,
        "human_decision": None,
        "execution_status": "pending_routing",
    }


def route_action(state: GraphState) -> str:
    """Return the next node, applying hard policy before confidence routing."""

    action = state["proposed_action"]
    if action == HIGH_RISK_ACTION:
        return "execute_high_risk_action"

    if action == LOW_RISK_ACTION and state["confidence_score"] >= CONFIDENCE_THRESHOLD:
        return "execute_low_risk_action"

    return "execute_high_risk_action"


def _serialize_entry(entry: AuditEntry) -> dict[str, Any]:
    if hasattr(entry, "model_dump"):
        return entry.model_dump()
    return entry.dict()


def append_audit_entry(entry: AuditEntry) -> None:
    """Append one audit record while preserving all existing records."""

    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(AUDIT_LOG_PATH.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = []
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    existing.append(_serialize_entry(entry))
    AUDIT_LOG_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _new_audit_entry(state: GraphState, action: str, decision: str) -> AuditEntry:
    return AuditEntry(
        customer_id=state["customer_id"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent_id=AGENT_ID,
        action=action,
        confidence=state["confidence_score"],
        reviewer_id=state.get("reviewer_id") or "system",
        decision=decision,
    )


def execute_low_risk_action(state: GraphState) -> dict[str, str]:
    """Execute an approved low-risk action automatically."""

    action = state["proposed_action"]
    append_audit_entry(_new_audit_entry(state, action, "auto_execute"))
    return {
        "execution_status": "executed",
        "result_message": "Retention email sent automatically.",
    }


def execute_high_risk_action(state: GraphState) -> dict[str, str]:
    """Execute, reject, or edit a high-risk action after human review."""

    decision = (state.get("human_decision") or "").lower()
    original_action = state["proposed_action"]

    if decision == "approve":
        action = original_action
        status = "executed"
        message = f"High-risk action '{action}' executed after approval."
    elif decision == "edit":
        action = state.get("edited_action") or original_action
        status = "executed"
        message = f"Edited action '{action}' executed after human review."
    elif decision == "reject":
        action = original_action
        status = "rejected"
        message = f"High-risk action '{action}' rejected; no action was executed."
    else:
        raise ValueError("human_decision must be approve, reject, or edit")

    append_audit_entry(_new_audit_entry(state, action, decision))
    return {"execution_status": status, "result_message": message}


def build_graph():
    """Build and compile the HITL graph with in-memory checkpointing."""

    builder = StateGraph(GraphState)
    builder.add_node("evaluate_customer", evaluate_customer)
    builder.add_node("execute_low_risk_action", execute_low_risk_action)
    builder.add_node("execute_high_risk_action", execute_high_risk_action)
    builder.add_edge(START, "evaluate_customer")
    builder.add_conditional_edges(
        "evaluate_customer",
        route_action,
        {
            "execute_low_risk_action": "execute_low_risk_action",
            "execute_high_risk_action": "execute_high_risk_action",
        },
    )
    builder.add_edge("execute_low_risk_action", END)
    builder.add_edge("execute_high_risk_action", END)

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["execute_high_risk_action"],
    )


def make_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}
