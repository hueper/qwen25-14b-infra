# Infrastructure — Qwen2.5 14B on SageMaker

Terraform configuration for deploying [Qwen2.5-14B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4) co-hosted with [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) on a single AWS SageMaker endpoint. A single endpoint serves both text generation and embedding requests.

**Monthly cost: ~$248 (176 working hours × $1.41/hr, eu-central-1)**

## Architecture

```
SageMaker Endpoint (ml.g5.xlarge, 1× A10G 24 GB)
  └── Container
        ├── vLLM (port 8000) — Qwen2.5 14B INT4, ~7 GB VRAM
        ├── embed_server (port 8001) — multilingual-e5-large, ~1.5 GB VRAM
        └── serve.py (port 8080) — SageMaker adapter, routes by task field

Lambda lifecycle (always deployed)
  start: aws lambda invoke --function-name qwen25-14b-start-endpoint
  stop:  aws lambda invoke --function-name qwen25-14b-stop-endpoint

EventBridge scheduling (optional, enable_endpoint_scheduler=true)
  start: 7 AM UTC Mon–Fri    stop: 6 PM UTC Mon–Fri

GitHub Actions (OIDC) → builds and pushes container image to ECR on merge to main
```

The SageMaker endpoint is **ephemeral** — created and deleted by the Lambda functions to avoid idle costs ($1.41/hr). It is not tracked in Terraform state. Everything else (model, endpoint config, ECR, IAM, Lambdas) is long-lived and Terraform-managed.

Both models are open-weight (Apache 2.0) — no HuggingFace token required.

The embedding model weights (~500 MB) are **pre-baked into the container image** at build time to avoid downloading them on every cold start.

## Endpoint API

Both LLM generation and embedding share a single `/invocations` endpoint, routed by the `task` field.

**Text generation:**
```json
POST /invocations
{"task": "generate", "inputs": "your prompt here", "parameters": {"max_new_tokens": 512}}

→ {"generated_text": "..."}
```

**Embedding:**
```json
POST /invocations
{"task": "embed", "texts": ["passage: first text", "passage: second text"], "batch_size": 32}

→ {"embeddings": [[...], ...], "dim": 1024}
```

> Prefix documents with `"passage: "` and queries with `"query: "` — required by multilingual-e5-large's training objective.

## CI Build Environment

The container image is large (~30–40 GB unpacked).
Building it on GitHub-hosted runners may fail due to disk limits.

In production, this repository is built using a **self-hosted GitHub Actions runner**
(EC2 with large disk, long-lived CI infrastructure).

If you encounter `no space left on device` errors in CI, switch the workflow to a
self-hosted runner.

## Prerequisites

- AWS CLI configured with appropriate credentials
- Terraform >= 1.0
- Verify `ml.g5.xlarge` quota in `eu-central-1` before deploying:
  ```bash
  aws service-quotas list-service-quotas --service-code sagemaker --region eu-central-1 \
    --query "Quotas[?contains(QuotaName, 'g5.xlarge')]"
  ```
  If the instance type is unavailable, set `aws_region = "us-east-1"` in `variables.tf`.

## Deploy

```bash
cd terraform

# 1. Provision base infrastructure (creates ECR, OIDC provider, CI role)
terraform init
terraform apply \
  -target=aws_ecr_repository.qwen25_inference \
  -target=aws_iam_openid_connect_provider.github \
  -target=aws_iam_role.github_actions \
  -target=aws_iam_role_policy.github_actions_ecr

# 2. Set AWS_ROLE_ARN secret in GitHub repo settings
#    (value from: terraform output github_actions_role_arn)

# 3. Push to main (or trigger workflow manually) to build the image

# 4. Provision remaining infrastructure with the image tag from CI
terraform apply -var="image_tag=<git-sha>" -var="github_repo=<org/repo>"
# Optionally enable EventBridge scheduling (auto start/stop Mon–Fri):
# terraform apply -var="image_tag=<git-sha>" -var="github_repo=<org/repo>" -var="enable_endpoint_scheduler=true"

# 5. Start the endpoint
aws lambda invoke --function-name qwen25-14b-start-endpoint --payload '{}' response.json
```

> **Note:** The `-target` apply in step 1 is a one-time bootstrap to break the circular dependency (CI needs ECR + OIDC role to push, Terraform needs an image tag to create the model). After the first image is pushed, all subsequent deploys use a normal `terraform apply`.

The endpoint takes ~5–8 minutes to reach `InService`.

## Structure

```
infra/
├── .github/workflows/
│   └── build-image.yml  CI: build and push container image to ECR
└── terraform/
    ├── main.tf          Model, endpoint config, ECR, IAM
    ├── lambda.tf        Endpoint lifecycle (Lambda always-on + optional EventBridge)
    ├── ci.tf            GitHub OIDC provider + CI IAM role
    ├── variables.tf
    ├── container/
    │   ├── Dockerfile       vllm/vllm-openai base + sentence-transformers + pre-baked e5 weights
    │   ├── entrypoint.sh    Starts vLLM (port 8000) + embed_server (port 8001) + serve.py (port 8080)
    │   ├── serve.py         SageMaker adapter — routes task=generate/embed
    │   └── embed_server.py  Sidecar embedding server (multilingual-e5-large)
    └── lambda/
        ├── start_endpoint.py
        └── stop_endpoint.py
```

## Full Shutdown / Destroy

```bash
cd terraform

# 1. Delete the endpoint (if running)
aws sagemaker delete-endpoint --endpoint-name qwen25-14b-endpoint

# 2. Destroy all remaining Terraform-managed resources
terraform destroy -var="image_tag=placeholder" -var="github_repo=placeholder"
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `image_tag` | *(required)* | Container image tag (Git SHA from CI) |
| `github_repo` | *(required)* | GitHub repository (org/repo) for OIDC trust |
| `aws_region` | `eu-central-1` | AWS region |
| `instance_type` | `ml.g5.xlarge` | SageMaker GPU instance (1× A10G 24 GB) |
| `enable_endpoint_scheduler` | `false` | EventBridge cron scheduling (Lambdas always deployed) |
| `endpoint_start_schedule` | `cron(0 7 ? * MON-FRI *)` | Start time (UTC) |
| `endpoint_stop_schedule` | `cron(0 18 ? * MON-FRI *)` | Stop time (UTC) |
