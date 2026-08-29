"""Streamlit approval interface for the churn-risk HITL graph."""

import uuid

import streamlit as st

from graph import build_graph, make_config


st.set_page_config(page_title="Churn Risk HITL", page_icon="🧭")
st.title("Customer Churn Risk — Human Review")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

graph = st.session_state.graph
config = make_config(st.session_state.thread_id)

customer_id = st.text_input("Customer ID", value="CUST002")
reviewer_id = st.text_input("Reviewer ID", value="operator_01")

if st.button("Start Evaluation", type="primary"):
    st.session_state.thread_id = str(uuid.uuid4())
    config = make_config(st.session_state.thread_id)
    try:
        graph.invoke(
            {
                "customer_id": customer_id,
                "proposed_action": "",
                "confidence_score": 0.0,
                "reasoning": "",
                "human_decision": None,
            },
            config,
        )
        st.rerun()
    except Exception as exc:
        st.error(f"Không thể gọi LLM hoặc chạy workflow: {exc}")


snapshot = graph.get_state(config)
values = snapshot.values or {}
if values.get("proposed_action"):
    st.subheader("Action Card")
    st.write(f"**Customer:** {values.get('customer_id')}")
    st.write(f"**Proposed action:** `{values.get('proposed_action')}`")
    st.write(f"**Confidence:** {values.get('confidence_score', 0):.2f}")
    st.write(f"**Reasoning:** {values.get('reasoning')}")

    if values.get("execution_status") and values.get("execution_status") != "pending_routing":
        st.write(f"**Execution status:** `{values.get('execution_status')}`")
    if values.get("human_decision"):
        st.write(f"**Human decision:** `{values.get('human_decision')}`")
    if values.get("reviewer_id"):
        st.write(f"**Reviewer:** `{values.get('reviewer_id')}`")
    if values.get("edited_action"):
        st.write(f"**Edited action:** `{values.get('edited_action')}`")
    if values.get("reviewed_at"):
        st.write(f"**Reviewed at:** `{values.get('reviewed_at')}`")

    is_pending_review = bool(snapshot.next)
    if is_pending_review:
        edited_action = st.text_input(
            "Edited action (used only when selecting Edit)",
            value=values.get("proposed_action", ""),
        )
        approve, reject, edit = st.columns(3)

        def resume(decision: str, action: str | None = None) -> None:
            try:
                update = {"human_decision": decision, "reviewer_id": reviewer_id}
                if action is not None:
                    update["edited_action"] = action
                graph.update_state(config, update)
                graph.invoke(None, config)
                st.rerun()
            except Exception as exc:
                st.error(f"Không thể resume workflow: {exc}")

        if approve.button("Approve"):
            resume("approve")
        if reject.button("Reject"):
            resume("reject")
        if edit.button("Edit"):
            resume("edit", edited_action)
    else:
        if values.get("audit_entry"):
            st.write("**Audit entry:**")
            st.json(values["audit_entry"])
        st.success(values.get("result_message", "Workflow completed."))
