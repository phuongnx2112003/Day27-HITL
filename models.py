"""Shared state and validation models for the churn-risk HITL workflow."""

from typing import TypedDict

from pydantic import BaseModel, Field


class GraphState(TypedDict, total=False):
    """State persisted by LangGraph while a human decision is pending."""

    customer_id: str
    proposed_action: str
    confidence_score: float
    reasoning: str
    human_decision: str | None
    reviewer_id: str | None
    edited_action: str | None
    execution_status: str | None
    result_message: str | None
    reviewed_at: str | None
    audit_entry: dict | None


class AuditEntry(BaseModel):
    """An append-only record of an agent proposal and its final decision."""

    customer_id: str
    timestamp: str
    agent_id: str
    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    reviewer_id: str
    decision: str
