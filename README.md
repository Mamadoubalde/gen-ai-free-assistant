# Episode 1 — I Built My Own AI Assistant for Free

Hands-on lab code for the `genai-dev-tv` YouTube series, Episode 1. Universal-audience video,
AWS-implementation lab, mapped to AIP-C01 Domain 1 (Foundation Model Integration).

## What's here

```
episode-01-ai-assistant/
├── cdk/                          # Infrastructure (Python CDK v2)
│   ├── app.py                    # CDK entry point
│   ├── cdk.json
│   ├── requirements.txt
│   └── genai_dev_cdk/
│       ├── genai_dev_stack.py    # IAM user, Budget, Lambda, API Gateway, S3
│       └── lambda/chat_handler/handler.py
├── scripts/
│   ├── 01_bedrock_basics.py      # Call Claude, Titan, Nova via Converse API
│   ├── 01_chat_app.py            # Streamlit chat UI with a model selector
│   ├── 01_cost_tracker.py        # CountTokens API + CloudWatch cost metrics
│   └── requirements.txt
├── notebooks/
│   └── 01_bedrock_basics.ipynb   # Interactive walkthrough of the basics script
└── architecture.mmd              # Mermaid source for the lab's architecture diagram
```

## Quickstart (local, no infra deploy needed)

You do **not** need to deploy the CDK stack to film the "Hands-on Lab" segment — the scripts
and Streamlit app talk to Bedrock directly. Deploy the CDK stack only for the "serverless API"
part of the demo (the API Gateway + Lambda + S3 path).

```bash
# 1. Request model access (one-time, per AWS account/region)
#    Bedrock console -> Model access -> enable Anthropic Claude, Amazon Titan, Amazon Nova

# 2. Configure credentials for a user/role with Bedrock permissions
export AWS_PROFILE=genai-dev
export AWS_REGION=us-east-1

# 3. Install deps
pip install -r scripts/requirements.txt

# 4. Run the CLI demo (calls Claude + Titan + Nova with one Converse API)
python scripts/01_bedrock_basics.py

# 5. Launch the chat UI
streamlit run scripts/01_chat_app.py

# 6. Check what a single message actually costs
python scripts/01_cost_tracker.py --prompt "What is a foundation model?" --model-id amazon.nova-lite-v1:0
```

## Deploying the serverless path (optional, for the API Gateway demo)

```bash
cd cdk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CDK_NOTIFY_EMAIL="you@example.com"   # budget alerts go here
export CDK_DEFAULT_ACCOUNT=<your-account-id>
export CDK_DEFAULT_REGION=us-east-1
cdk bootstrap   # first time only, per account/region
cdk deploy
```

After deploy, `cdk deploy` prints `ChatApiUrl`. Test it:

```bash
curl -X POST "<ChatApiUrl>chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a foundation model?"}'
```

Create a scoped-down access key for local dev **after** deploy, outside the stack (see the
comment in `genai_dev_stack.py` for why it's not auto-generated):

```bash
aws iam create-access-key --user-name genai-dev
```

When you're done filming: `cdk destroy` to avoid any ongoing charges, and rotate/delete the
`genai-dev` access key.

## Notes on model IDs

Bedrock model IDs and availability change over time (see `EXAM_NOTES` in the production doc for
the model-lifecycle exam point). `01_bedrock_basics.py` and the notebook **discover** enabled
models via `list_foundation_models()` rather than hard-coding one that might be deprecated.
`01_chat_app.py` and `01_cost_tracker.py` reference specific model IDs as of Sept 2026 for
readability on camera — if `converse()` rejects one with `ValidationException`, check
Bedrock console -> Model access, or swap in whatever `01_bedrock_basics.py` discovers.

If your Lambda's built-in boto3 version predates the Converse API, bundle a newer `boto3` /
`botocore` as a Lambda layer (`pip install boto3 -t layer/python`) rather than relying on the
runtime default.

## Cost estimate for this episode

See the production doc's cost table — the short version: model access is free to enable,
Bedrock is pay-per-token (a full afternoon of testing is well under $1 with Nova/Titan, and
still cents with Claude), the CDK stack's idle cost is effectively $0 (Lambda/API Gateway/S3
free tier), and the $10/month Budget alarm is the safety net so "free" stays true even if you
forget to `cdk destroy`.
