# Day 27 — LangGraph Human-in-the-Loop

Ứng dụng mô phỏng đánh giá rủi ro khách hàng rời bỏ và yêu cầu human review trước các hành động rủi ro cao.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cấu hình OpenAI

`evaluate_customer()` gọi OpenAI-compatible Chat Completions API. Có thể tạo file `.env` (file này đã được gitignore) hoặc thiết lập biến môi trường:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-base-url/v1"
export OPENAI_MODEL="your-model"
```

`OPENAI_BASE_URL` có thể bỏ qua nếu dùng endpoint OpenAI mặc định. Không ghi các giá trị thật vào Git hoặc commit file `.env`.

## Chạy ứng dụng

```bash
streamlit run app.py
```

Chạy test:

```bash
pytest -q
```

## Luật routing

- `increase_credit_limit` luôn đi qua human review, kể cả confidence rất cao.
- `send_email` được tự động thực thi khi confidence `>= 0.85`.
- Confidence `< 0.85` sẽ được escalation để human review.

Graph sử dụng `MemorySaver()` và `interrupt_before=["execute_high_risk_action"]`. Streamlit lưu cùng `thread_id`, cập nhật state bằng `graph.update_state()` rồi resume bằng `graph.invoke(None, config)`.

Human có thể Approve, Reject hoặc Edit action. Mọi quyết định được append vào `audit_log.json` cùng customer, timestamp, agent, action, confidence, reviewer và decision. LLM request có timeout 30 giây, retry tối đa 2 lần và fallback cho provider không hỗ trợ `response_format`.

Repository không chứa API key, password, access token, private key hoặc credential thật.
