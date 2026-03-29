ARG BASE_IMAGE
FROM ${BASE_IMAGE}  # vllm/vllm-openai:v0.11.0

RUN apt-get update && apt-get install git && \
    pip install fastapi uvicorn httpx transformers accelerate \
        sentence-transformers "numpy<2"

# Pre-bake embedding model weights into the image at build time.
# This avoids a ~500 MB download on every container cold start.
# Model: intfloat/multilingual-e5-large (560M params, 1024-dim, Apache 2.0)
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-large'); \
print('Embedding model weights downloaded successfully')"

COPY serve.py /opt/serve.py
COPY embed_server.py /opt/embed_server.py
COPY entrypoint.sh /opt/entrypoint.sh
ENTRYPOINT ["/bin/bash", "/opt/entrypoint.sh"]
