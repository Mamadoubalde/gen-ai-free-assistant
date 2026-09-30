#!/usr/bin/env python3
"""
CDK entry point for Episode 1: "I Built My Own AI Assistant for Free"

Deploys the genai-dev sandbox used in the video:
  - IAM user "genai-dev" scoped to Bedrock (for local CLI/Streamlit use)
  - AWS Budget alarm at $10/month (so the "free" claim stays true)
  - Lambda "chat_handler" that calls Bedrock's Converse API
  - API Gateway REST endpoint in front of the Lambda
  - S3 bucket that stores chat history (JSON per session)

Usage:
    cd cdk
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    export CDK_NOTIFY_EMAIL="you@example.com"   # required for budget alerts
    cdk bootstrap   # first time only, per account/region
    cdk deploy
"""
import os
import aws_cdk as cdk
from genai_dev_cdk.genai_dev_stack import GenAIDevStack

app = cdk.App()

notify_email = os.environ.get("CDK_NOTIFY_EMAIL", "you@example.com")

GenAIDevStack(
    app,
    "GenAIDevStack",
    notify_email=notify_email,
    monthly_budget_usd=10,
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
    description="Episode 1 sandbox: Bedrock chat assistant (genai-dev-tv series)",
)

app.synth()
