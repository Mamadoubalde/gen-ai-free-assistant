"""
01_bedrock_basics.py
=====================
Episode 1 hands-on lab, part 1: call three different foundation models on Amazon
Bedrock through ONE unified API (Converse), so you can see model-agnostic design
in action before you ever build the chat app.

Exam tie-in (AIP-C01, Domain 1 - Foundation Model Integration):
  - Bedrock's control-plane API (`bedrock`) lists/describes models.
  - The data-plane API (`bedrock-runtime`) is what actually runs inference.
  - `Converse` / `ConverseStream` give one request/response shape across model
    families; `InvokeModel` requires a model-specific JSON body instead.

Usage:
    pip install boto3
    export AWS_PROFILE=genai-dev   # or configure credentials however you prefer
    python 01_bedrock_basics.py
"""
from __future__ import annotations

import argparse
import json
import sys

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"  # Bedrock model availability varies by region; us-east-1 has the widest selection.

# Model IDs rotate as AWS ships new versions and retires old ones (the exam calls this
# "model lifecycle management" - see EXAM_NOTES.md). Rather than hard-code one Claude
# version that might be deprecated by the time you watch this, we discover what's
# actually enabled in your account at runtime.
# PREFERRED_FAMILIES = ["anthropic.claude", "amazon.titan-text", "amazon.nova"]


# def discover_models(bedrock_client) -> dict[str, str]:
#     """Return {family_label: model_id} for the first available model in each family."""
#     resp = bedrock_client.list_foundation_models(byOutputModality="TEXT")
#     available = {m["modelId"]: m for m in resp["modelSummaries"]}

#     chosen = {}
#     for family in PREFERRED_FAMILIES:
#         # Prefer models flagged ON_DEMAND-inferenceable so we don't need a provisioned
#         # throughput purchase or an inference profile just to run this script.
#         candidates = sorted(
#             mid for mid in available
#             if mid.startswith(family)
#             and "ON_DEMAND" in available[mid].get("inferenceTypesSupported", [])
#         )
#         if candidates:
#             chosen[family] = candidates[-1]  # newest-looking id string, good enough for a demo
#     return chosen

PREFERRED_FAMILIES = ["anthropic.claude", "amazon.nova", "meta.llama3"]

# As of late 2026, Anthropic models on Bedrock auto-enable on first invoke, and this
# account doesn't tag them "ON_DEMAND" in inferenceTypesSupported the way other providers
# still do -- so the ON_DEMAND filter below silently drops every Claude model. Confirmed
# working ids go here and skip that filter. (Titan Text was dropped entirely -- retired
# from on-demand inference in this account/region in favor of Nova.)
MANUAL_MODEL_OVERRIDES = {
    "anthropic.claude": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


def discover_models(bedrock_client) -> dict[str, str]:
    """Return {family_label: model_id} for the first available model in each family."""
    resp = bedrock_client.list_foundation_models(byOutputModality="TEXT")
    available = {m["modelId"]: m for m in resp["modelSummaries"]}

    chosen = {}
    for family in PREFERRED_FAMILIES:
        candidates = sorted(
            mid for mid in available
            if mid.startswith(family)
            and "ON_DEMAND" in available[mid].get("inferenceTypesSupported", [])
        )
        if candidates:
            chosen[family] = candidates[-1]

    for family, override_id in MANUAL_MODEL_OVERRIDES.items():
        if family not in chosen:
            chosen[family] = override_id  # inference-profile ids don't appear in `available`

    return chosen


def ask(bedrock_runtime, model_id: str, prompt: str) -> dict:
    """Send one turn to a model via the Converse API and return reply + usage."""
    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={
            "maxTokens": 300,
            "temperature": 0.5,
        },
        system=[{"text": "You are a concise assistant for a YouTube coding tutorial. Answer in 2-3 sentences."}],
    )
    reply_blocks = response["output"]["message"]["content"]
    reply_text = "".join(b.get("text", "") for b in reply_blocks)
    return {
        "model_id": model_id,
        "reply": reply_text,
        "input_tokens": response["usage"]["inputTokens"],
        "output_tokens": response["usage"]["outputTokens"],
        "latency_ms": response["metrics"]["latencyMs"],
    }


def main():
    parser = argparse.ArgumentParser(description="Bedrock basics: call Claude, Titan, and Nova with one API.")
    parser.add_argument("--prompt", default="Explain what a foundation model is, for someone who has never used AI.")
    parser.add_argument("--region", default=REGION)
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    bedrock = session.client("bedrock")
    bedrock_runtime = session.client("bedrock-runtime")

    print(f"Discovering available models in {args.region} ...")
    models = discover_models(bedrock)
    if not models:
        print(
            "No models found. In the Bedrock console, go to 'Model access' and request "
            "access to Anthropic Claude, Amazon Titan, and Amazon Nova models, then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Using models: {json.dumps(models, indent=2)}\n")
    print(f"Prompt: {args.prompt}\n")

    results = []
    for family, model_id in models.items():
        try:
            result = ask(bedrock_runtime, model_id, args.prompt)
        except ClientError as e:
            print(f"[{family}] ERROR calling {model_id}: {e}")
            continue
        results.append(result)
        print(f"--- {family} ({model_id}) ---")
        print(result["reply"])
        print(
            f"  tokens: {result['input_tokens']} in / {result['output_tokens']} out "
            f"| latency: {result['latency_ms']}ms\n"
        )

    total_in = sum(r["input_tokens"] for r in results)
    total_out = sum(r["output_tokens"] for r in results)
    print(f"Totals across {len(results)} model(s): {total_in} input tokens, {total_out} output tokens.")
    print("Run 01_cost_tracker.py next to turn that into a dollar estimate.")


if __name__ == "__main__":
    main()
