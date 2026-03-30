ARG BASE_IMAGE
# BASE_IMAGE = vllm/vllm-openai:v0.11.0
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install git && \
    pip install fastapi uvicorn httpx transformers accelerate \
        sentence-transformers

# Pre-bake embedding model weights into the image at build time.
# This avoids a ~500 MB download on every container cold start.
# Model: intfloat/multilingual-e5-large (560M params, 1024-dim, Apache 2.0)
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-large'); \
print('Embedding model weights downloaded successfully')"

# Pre-bake LLM weights into the image at build time.
# Avoids a ~7 GB download on every cold start (~20 min → ~2-3 min).
# Model: Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4'); \
print('LLM weights downloaded successfully')"

COPY serve.py /opt/serve.py
COPY embed_server.py /opt/embed_server.py
COPY entrypoint.sh /opt/entrypoint.sh
ENTRYPOINT ["/bin/bash", "/opt/entrypoint.sh"]
