"""
01_cost_tracker.py
====================
Episode 1 hands-on lab, part 3: know what your AI assistant costs BEFORE you run it,
using Bedrock's CountTokens API, then push actual usage to CloudWatch as a custom metric
so you have a running total for the "Results" segment.

Exam tie-in (AIP-C01, Domain 4 - Operational Efficiency and Optimization):
  - CountTokens (bedrock-runtime) estimates token count for a prompt BEFORE inference,
    for cost projection and context-window checks. At launch it covers Anthropic Claude
    models on Bedrock; for other model families, fall back to a heuristic estimate.
  - Bedrock is priced per 1,000 input/output tokens, and the price differs by model
    family and size (Nova Micro << Nova Pro < Claude Sonnet, roughly).
  - Publishing your own CloudWatch custom metric (namespace "GenAIDevTV/Bedrock") lets
    you build a cost dashboard without waiting a day for Cost Explorer data to land.

Usage:
    python 01_cost_tracker.py --prompt "Summarize the plot of Dune in one paragraph." \\
        --model-id anthropic.claude-3-5-haiku-20241022-v1:0
"""
from __future__ import annotations

import argparse
import datetime as dt

import boto3
from botocore.exceptions import ClientError

# Rough, illustrative USD price per 1,000 tokens (on-demand, us-east-1-ish).
# ALWAYS check https://aws.amazon.com/bedrock/pricing/ for current numbers before
# trusting a cost estimate -- Bedrock pricing changes as new model versions ship.
PRICE_PER_1K_TOKENS_USD = {
    "anthropic.claude": {"input": 0.003, "output": 0.015},   # Claude Sonnet-class, illustrative
    "amazon.titan-text": {"input": 0.0002, "output": 0.0006},
    "amazon.nova-micro": {"input": 0.000035, "output": 0.00014},
    "amazon.nova-lite": {"input": 0.00006, "output": 0.00024},
    "amazon.nova-pro": {"input": 0.0008, "output": 0.0032},
}


def price_for(model_id: str) -> dict:
    # Cross-region inference profile ids are prefixed with a region group
    # ("us.", "global.", "eu.", "apac.") -- strip that before matching the base model id.
    prefix_group, _, rest = model_id.partition(".")
    bare_id = rest if prefix_group in ("us", "eu", "apac", "global") else model_id
    for prefix, price in PRICE_PER_1K_TOKENS_USD.items():
        if bare_id.startswith(prefix):
            return price
    return {"input": 0.0, "output": 0.0}

def count_tokens(bedrock_runtime, model_id: str, prompt: str) -> int | None:
    """
    Use the CountTokens API to get an exact pre-inference token count.
    Falls back to None (caller should estimate) if the model/region doesn't support it yet.
    """
    try:
        resp = bedrock_runtime.count_tokens(
            modelId=model_id,
            input={"converse": {"messages": [{"role": "user", "content": [{"text": prompt}]}]}},
        )
        return resp["inputTokens"]
    except (ClientError, KeyError):
        return None


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    price = price_for(model_id)
    return (input_tokens / 1000) * price["input"] + (output_tokens / 1000) * price["output"]


def publish_metric(cloudwatch, model_id: str, cost_usd: float, total_tokens: int) -> None:
    cloudwatch.put_metric_data(
        Namespace="GenAIDevTV/Bedrock",
        MetricData=[
            {
                "MetricName": "EstimatedCostUSD",
                "Dimensions": [{"Name": "ModelId", "Value": model_id}],
                "Timestamp": dt.datetime.now(dt.timezone.utc),
                "Value": cost_usd,
                "Unit": "None",
            },
            {
                "MetricName": "TotalTokens",
                "Dimensions": [{"Name": "ModelId", "Value": model_id}],
                "Timestamp": dt.datetime.now(dt.timezone.utc),
                "Value": total_tokens,
                "Unit": "Count",
            },
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Estimate and track Bedrock inference cost.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-id", default="amazon.nova-lite-v1:0")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--skip-cloudwatch", action="store_true", help="Just print the estimate, don't publish a metric.")
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    bedrock_runtime = session.client("bedrock-runtime")

    pre_count = count_tokens(bedrock_runtime, args.model_id, args.prompt)
    if pre_count is not None:
        print(f"CountTokens (pre-inference): {pre_count} input tokens")
    else:
        print("CountTokens API not available for this model/region; will use actual usage from the response instead.")

    response = bedrock_runtime.converse(
        modelId=args.model_id,
        messages=[{"role": "user", "content": [{"text": args.prompt}]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0.5},
    )
    usage = response["usage"]
    input_tokens, output_tokens = usage["inputTokens"], usage["outputTokens"]
    cost = estimate_cost(args.model_id, input_tokens, output_tokens)

    print(f"\nModel: {args.model_id}")
    print(f"Actual usage: {input_tokens} input / {output_tokens} output tokens")
    print(f"Estimated cost for this call: ${cost:.6f}")
    print("(Prices in this script are illustrative -- check the Bedrock pricing page for current rates.)")

    if not args.skip_cloudwatch:
        try:
            publish_metric(session.client("cloudwatch"), args.model_id, cost, input_tokens + output_tokens)
            print("Published EstimatedCostUSD + TotalTokens to CloudWatch namespace 'GenAIDevTV/Bedrock'.")
        except ClientError as e:
            print(f"Could not publish CloudWatch metric (check IAM permissions): {e}")


if __name__ == "__main__":
    main()
