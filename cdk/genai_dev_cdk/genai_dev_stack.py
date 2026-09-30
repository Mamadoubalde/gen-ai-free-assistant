"""
GenAIDevStack — Episode 1 infrastructure.

Architecture (see docs/architecture.mmd for the full diagram):

    User -> Streamlit UI (local) -> API Gateway -> Lambda (chat_handler) -> Bedrock Converse API
                                                          |                        |
                                                          v                        v
                                                    S3 (chat history)      Claude / Titan / Nova
                                                          ^
                                                  CloudWatch (cost + usage metrics)

Design decisions (explained in the video's "Why AWS" section):
  1. Converse API over InvokeModel — one unified request/response shape works across
     every Bedrock text model (Claude, Titan, Nova, Llama, Mistral...). InvokeModel needs
     a different JSON body per model family; Converse abstracts that away, which is also
     why the AIP-C01 exam leans on it for "model-agnostic" application design questions.
  2. Serverless (Lambda + API Gateway) over a rented GPU box — Bedrock hosts the model,
     so the compute layer only needs to call an API. No GPU instance, no idle cost,
     true pay-per-request pricing, which is what makes the "$0/month when idle" claim true.
  3. Multi-model access — the IAM user/role is granted bedrock:InvokeModel /
     bedrock:Converse across model ARNs (not just one), so the chat app's model selector
     can route cheap/simple prompts to Nova Micro or Titan and harder ones to Claude,
     which is the M7 "cost optimization" pattern used later in the series.
"""
from __future__ import annotations

import os

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_s3 as s3,
    aws_budgets as budgets,
    aws_logs as logs,
)
from constructs import Construct


class GenAIDevStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        notify_email: str,
        monthly_budget_usd: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------------
        # 1. S3 bucket for chat history (JSON blobs, one object per session)
        # ------------------------------------------------------------------
        chat_history_bucket = s3.Bucket(
            self,
            "ChatHistoryBucket",
            bucket_name=None,  # let CDK auto-name to avoid global collisions
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,  # sandbox account: OK to tear down
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-old-chat-history",
                    expiration=Duration.days(90),
                )
            ],
        )

        # ------------------------------------------------------------------
        # 2. Lambda function: chat_handler
        # ------------------------------------------------------------------
        chat_handler_role = iam.Role(
            self,
            "ChatHandlerRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # Scope Bedrock access to Converse / CountTokens on-demand invocation.
        # In production, narrow the resource ARN to the specific model IDs you use.
        chat_handler_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockInvoke",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:Converse",
                    "bedrock:ConverseStream",
                    "bedrock:CountTokens",
                ],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/*",
                    "arn:aws:bedrock:*:*:inference-profile/*",
                ],
            )
        )
        chat_history_bucket.grant_read_write(chat_handler_role)

        chat_handler_log_group = logs.LogGroup(
            self,
            "ChatHandlerLogGroup",
            log_group_name="/aws/lambda/chat_handler",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        chat_handler_fn = _lambda.Function(
            self,
            "ChatHandlerFunction",
            function_name="chat_handler",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset("genai_dev_cdk/lambda/chat_handler"),
            role=chat_handler_role,
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "CHAT_HISTORY_BUCKET": chat_history_bucket.bucket_name,
                "DEFAULT_MODEL_ID": "amazon.nova-lite-v1:0",
                "POWERTOOLS_LOG_LEVEL": "INFO",
            },
            log_group=chat_handler_log_group,
        )

        # ------------------------------------------------------------------
        # 3. API Gateway REST endpoint in front of the Lambda
        # ------------------------------------------------------------------
        api = apigw.LambdaRestApi(
            self,
            "ChatApi",
            handler=chat_handler_fn,
            proxy=False,
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=5,
                throttling_burst_limit=10,
            ),
        )
        chat_resource = api.root.add_resource("chat")
        chat_resource.add_method("POST")

        # ------------------------------------------------------------------
        # 4. IAM user "genai-dev" — for the Streamlit app / CLI scripts running
        #    on your laptop during the video, separate from the Lambda's role.
        # ------------------------------------------------------------------
        genai_dev_user = iam.User(self, "GenAIDevUser", user_name="genai-dev")
        genai_dev_user.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockDevAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:Converse",
                    "bedrock:ConverseStream",
                    "bedrock:CountTokens",
                    "bedrock:ListFoundationModels",
                    "bedrock:GetFoundationModel",
                ],
                resources=["*"],
            )
        )
        # Read-only visibility into CloudWatch cost/usage metrics for 01_cost_tracker.py
        genai_dev_user.add_to_policy(
            iam.PolicyStatement(
                sid="CostTrackerMetrics",
                effect=iam.Effect.ALLOW,
                actions=[
                    "cloudwatch:PutMetricData",
                    "cloudwatch:GetMetricData",
                    "cloudwatch:GetMetricStatistics",
                    "ce:GetCostAndUsage",
                ],
                resources=["*"],
            )
        )
        chat_history_bucket.grant_read_write(genai_dev_user)

        # NOTE: we deliberately do NOT generate an access key here. Minting a long-lived
        # secret key inside a CloudFormation stack means it sits in plaintext in the
        # stack's outputs/console history forever -- exactly the kind of IAM mistake
        # Episode 10 ("I Hacked My Own AI") is about. After `cdk deploy`, create a
        # short-lived key yourself and store it outside the stack:
        #   aws iam create-access-key --user-name genai-dev > .genai-dev-key.json
        # (add .genai-dev-key.json to .gitignore, and rotate/delete it after filming).
        # Prefer IAM Identity Center / an assumed role over static keys where you can.

        # ------------------------------------------------------------------
        # 5. Budget alarm — the guardrail that keeps "$0/month" honest
        # ------------------------------------------------------------------
        budgets.CfnBudget(
            self,
            "MonthlyGenAIBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name="genai-dev-monthly-budget",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(
                    amount=monthly_budget_usd,
                    unit="USD",
                ),
            ),
            notifications_with_subscribers=[
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type="FORECASTED",
                        comparison_operator="GREATER_THAN",
                        threshold=80,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(
                            subscription_type="EMAIL",
                            address=notify_email,
                        )
                    ],
                ),
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type="ACTUAL",
                        comparison_operator="GREATER_THAN",
                        threshold=100,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(
                            subscription_type="EMAIL",
                            address=notify_email,
                        )
                    ],
                ),
            ],
        )

        # ------------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------------
        CfnOutput(self, "ChatApiUrl", value=api.url, description="POST {chatApiUrl}chat")
        CfnOutput(self, "ChatHistoryBucketName", value=chat_history_bucket.bucket_name)
        CfnOutput(self, "GenAIDevUserName", value=genai_dev_user.user_name)
