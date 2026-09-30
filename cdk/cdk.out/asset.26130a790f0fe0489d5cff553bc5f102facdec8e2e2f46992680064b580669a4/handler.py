"""
chat_handler — Lambda function behind API Gateway (POST /chat).

Request body (JSON):
    {
      "session_id": "abc123",
      "message": "What is a foundation model?",
      "model_id": "amazon.nova-lite-v1:0"   # optional, defaults to env DEFAULT_MODEL_ID
    }

Response body (JSON):
    {
      "session_id": "abc123",
      "reply": "...",
      "model_id": "amazon.nova-lite-v1:0",
      "input_tokens": 12,
      "output_tokens": 84
    }

This mirrors what 01_bedrock_basics.py does locally, but wrapped for API Gateway so the
Streamlit app (or curl, or Postman) can call it as a normal REST endpoint. It uses the
Converse API (not InvokeModel) specifically because Converse gives one request/response
shape across Claude, Titan and Nova -- see the module docstring in genai_dev_stack.py.
"""
from __future__ import annotations

import json
import os
import time
import uuid
import boto3

bedrock_runtime = boto3.client("bedrock-runtime")
s3 = boto3.client("s3")

DEFAULT_MODEL_ID = os.environ.get("DEFAULT_MODEL_ID", "amazon.nova-lite-v1:0")
CHAT_HISTORY_BUCKET = os.environ.get("CHAT_HISTORY_BUCKET")


def _load_history(session_id: str) -> list[dict]:
    if not CHAT_HISTORY_BUCKET:
        return []
    try:
        obj = s3.get_object(Bucket=CHAT_HISTORY_BUCKET, Key=f"sessions/{session_id}.json")
        return json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        return []
    except Exception:
        # Don't let a corrupted/missing history object break the chat turn.
        return []


def _save_history(session_id: str, history: list[dict]) -> None:
    if not CHAT_HISTORY_BUCKET:
        return
    s3.put_object(
        Bucket=CHAT_HISTORY_BUCKET,
        Key=f"sessions/{session_id}.json",
        Body=json.dumps(history).encode("utf-8"),
        ContentType="application/json",
    )


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "Request body must be valid JSON"})

    message = body.get("message")
    if not message:
        return _response(400, {"error": "Field 'message' is required"})

    session_id = body.get("session_id") or str(uuid.uuid4())
    model_id = body.get("model_id") or DEFAULT_MODEL_ID

    history = _load_history(session_id)
    history.append({"role": "user", "content": [{"text": message}]})

    try:
        result = bedrock_runtime.converse(
            modelId=model_id,
            messages=history,
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.5,
                "topP": 0.9,
            },
        )
    except bedrock_runtime.exceptions.ValidationException as e:
        return _response(400, {"error": f"Bedrock rejected the request: {e}"})
    except bedrock_runtime.exceptions.ThrottlingException:
        return _response(429, {"error": "Rate limited by Bedrock, retry shortly."})
    except Exception as e:  # noqa: BLE001 - surface unexpected errors to the caller
        return _response(500, {"error": str(e)})

    reply_message = result["output"]["message"]
    reply_text = "".join(block.get("text", "") for block in reply_message["content"])
    history.append(reply_message)
    _save_history(session_id, history)

    usage = result.get("usage", {})

    return _response(
        200,
        {
            "session_id": session_id,
            "reply": reply_text,
            "model_id": model_id,
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
            "latency_ms": result.get("metrics", {}).get("latencyMs"),
            "ts": int(time.time()),
        },
    )
