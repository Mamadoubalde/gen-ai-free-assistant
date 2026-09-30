"""
01_chat_app.py
================
Episode 1 hands-on lab, part 2: the actual "AI assistant"
a Streamlit chat UI with a model selector, backed directly by Bedrock's Converse API
(you can point BASE_URL at the deployed API Gateway endpoint instead).

Usage:
    pip install streamlit boto3
    export AWS_PROFILE=genai-dev
    streamlit run 01_chat_app.py
"""
from __future__ import annotations

import os
import time

import boto3
import streamlit as st
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Swap the model mid-conversation to feel the multi-model design decision firsthand:
# Nova Micro is near-instant and nearly free, Claude Sonnet is the "smart but pricier" option.
# MODEL_OPTIONS = {
#     "Nova Micro (fastest, cheapest)": "amazon.nova-micro-v1:0",
#     "Nova Lite (balanced)": "amazon.nova-lite-v1:0",
#     "Nova Pro (stronger reasoning)": "amazon.nova-pro-v1:0",
#     "Titan Text Express": "amazon.titan-text-express-v1",
#     "Claude 3.5 Haiku (fast Claude)": "anthropic.claude-3-5-haiku-20241022-v1:0",
#     "Claude Sonnet (strongest, priciest)": "anthropic.claude-3-5-sonnet-20241022-v2:0",
# }

MODEL_OPTIONS = {
    "Nova Micro (fastest, cheapest)": "amazon.nova-micro-v1:0",
    "Nova Lite (balanced)": "amazon.nova-lite-v1:0",
    "Nova Pro (stronger reasoning)": "amazon.nova-pro-v1:0",
    "Llama 3 8B (fast, open-weight)": "meta.llama3-8b-instruct-v1:0",
    "Claude Haiku 4.5 (fast Claude)": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}

SYSTEM_PROMPT = (
    "You are a private, self-hosted AI assistant running on the user's own AWS account. "
    "Be helpful, concise, and mention when you're not sure about something."
)


@st.cache_resource
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=REGION)


def call_model(client, model_id: str, messages: list[dict]) -> dict:
    response = client.converse(
        modelId=model_id,
        messages=messages,
        system=[{"text": SYSTEM_PROMPT}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.6},
    )
    reply_message = response["output"]["message"]
    reply_text = "".join(b.get("text", "") for b in reply_message["content"])
    return {
        "text": reply_text,
        "message": reply_message,
        "usage": response["usage"],
        "latency_ms": response["metrics"]["latencyMs"],
    }


def main():
    st.set_page_config(page_title="My AI Assistant ($0/mo)", page_icon="🤖")
    st.title("🤖 My AI Assistant")
    st.caption("Built on Amazon Bedrock — pay-per-request, no subscription, no GPU rental.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_cost_usd" not in st.session_state:
        st.session_state.session_cost_usd = 0.0

    with st.sidebar:
        st.subheader("Model")
        model_label = st.selectbox("Choose a model", list(MODEL_OPTIONS.keys()), index=1)
        model_id = MODEL_OPTIONS[model_label]
        st.caption(f"`{model_id}`")

        st.subheader("This session")
        st.metric("Messages", len(st.session_state.messages) // 2)
        st.metric("Est. cost so far", f"${st.session_state.session_cost_usd:.5f}")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.session_cost_usd = 0.0
            st.rerun()

        st.divider()
        st.caption(
            "Compare: ChatGPT Plus is $20/month flat. This app only costs money "
            "while you're actually sending messages -- see 01_cost_tracker.py."
        )

    for msg in st.session_state.messages:
        role = msg["role"]
        text = "".join(b.get("text", "") for b in msg["content"])
        with st.chat_message(role):
            st.markdown(text)

    prompt = st.chat_input("Ask me anything...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": [{"text": prompt}]})
        with st.chat_message("user"):
            st.markdown(prompt)

        client = get_bedrock_client()
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {model_label}..."):
                try:
                    start = time.time()
                    result = call_model(client, model_id, st.session_state.messages)
                except ClientError as e:
                    st.error(f"Bedrock error: {e}")
                    return
            st.markdown(result["text"])
            st.caption(
                f"{result['usage']['inputTokens']} in / {result['usage']['outputTokens']} out tokens "
                f"· {result['latency_ms']}ms"
            )

        st.session_state.messages.append(result["message"])
        # A tiny, illustrative running cost estimate -- see 01_cost_tracker.py for real pricing math.
        st.session_state.session_cost_usd += (
            result["usage"]["inputTokens"] + result["usage"]["outputTokens"]
        ) / 1000 * 0.001
        st.rerun()


if __name__ == "__main__":
    main()