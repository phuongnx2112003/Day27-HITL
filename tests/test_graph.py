import json

import pytest

import graph as workflow
from graph import (
    CONFIDENCE_THRESHOLD,
    append_audit_entry,
    build_graph,
    evaluate_customer,
    execute_high_risk_action,
    make_config,
    route_action,
    _parse_llm_json,
)
from models import AuditEntry


@pytest.fixture
def audit_file(tmp_path, monkeypatch):
    path = tmp_path / "audit_log.json"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(workflow, "AUDIT_LOG_PATH", path)
    return path


def test_evaluate_customer_returns_required_fields_and_valid_confidence(monkeypatch):
    monkeypatch.setattr(workflow, "_call_llm", lambda customer_id, profile: {
        "proposed_action": "send_email",
        "confidence_score": 0.92,
        "reasoning": "Mocked test reasoning.",
    })
    result = evaluate_customer({"customer_id": "CUST001"})
    assert {"proposed_action", "confidence_score", "reasoning"} <= result.keys()
    assert 0.0 <= result["confidence_score"] <= 1.0


def test_parse_llm_json_accepts_markdown_and_surrounding_text():
    content = "Here is the result:\n```json\n{\"proposed_action\": \"send_email\", \"confidence_score\": 0.9, \"reasoning\": \"ok\"}\n```"
    assert _parse_llm_json(content)["proposed_action"] == "send_email"


def test_hard_rule_overrides_high_confidence():
    assert route_action({"proposed_action": "increase_credit_limit", "confidence_score": 0.99}) == "execute_high_risk_action"


def test_high_confidence_low_risk_auto_executes():
    assert route_action({"proposed_action": "send_email", "confidence_score": CONFIDENCE_THRESHOLD}) == "execute_low_risk_action"


def test_low_confidence_escalates():
    assert route_action({"proposed_action": "send_email", "confidence_score": 0.82}) == "execute_high_risk_action"


def test_audit_entries_are_appended(audit_file):
    append_audit_entry(AuditEntry(
        customer_id="CUST001",
        timestamp="2026-08-29T00:00:00Z", agent_id="agent", action="send_email",
        confidence=0.9, reviewer_id="system", decision="auto_execute",
    ))
    append_audit_entry(AuditEntry(
        customer_id="CUST002",
        timestamp="2026-08-29T00:01:00Z", agent_id="agent", action="increase_credit_limit",
        confidence=0.96, reviewer_id="operator", decision="reject",
    ))
    assert len(json.loads(audit_file.read_text(encoding="utf-8"))) == 2


def test_high_risk_approve_reject_and_edit(audit_file):
    base = {
        "customer_id": "CUST002",
        "proposed_action": "increase_credit_limit",
        "confidence_score": 0.96,
        "reviewer_id": "operator_01",
    }
    assert execute_high_risk_action({**base, "human_decision": "approve"})["execution_status"] == "executed"
    assert execute_high_risk_action({**base, "human_decision": "reject"})["execution_status"] == "rejected"
    assert execute_high_risk_action({**base, "human_decision": "edit", "edited_action": "send_email"})["execution_status"] == "executed"
    entries = json.loads(audit_file.read_text(encoding="utf-8"))
    assert [entry["decision"] for entry in entries] == ["approve", "reject", "edit"]


def test_high_risk_graph_interrupts_before_execution(audit_file, monkeypatch):
    monkeypatch.setattr(workflow, "_call_llm", lambda customer_id, profile: {
        "proposed_action": "increase_credit_limit",
        "confidence_score": 0.96,
        "reasoning": "Mocked high-risk test reasoning.",
    })
    graph = build_graph()
    config = make_config("test-interrupt")
    graph.invoke({
        "customer_id": "CUST002",
        "proposed_action": "",
        "confidence_score": 0.0,
        "reasoning": "",
        "human_decision": None,
    }, config)
    snapshot = graph.get_state(config)
    assert snapshot.values["proposed_action"] == "increase_credit_limit"
    assert snapshot.next == ("human_review_action",)
    assert json.loads(audit_file.read_text(encoding="utf-8")) == []
