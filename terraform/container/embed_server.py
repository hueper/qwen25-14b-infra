"""
Embedding server for intfloat/multilingual-e5-large.
Runs as a sidecar on port 8001 alongside the vLLM LLM server.

Contract:
  GET  /health       → {"status": "healthy"}
  POST /embed        → embed a batch of texts

Request body:
  {
    "texts": ["passage: first text", "passage: second text"],
    "batch_size": 32
  }

Response body:
  {
    "embeddings": [[0.123, -0.456, ...], ...],  # list of float lists
    "dim": 1024
  }

Caller conventions (unchanged from the standalone embedding-infra endpoint):
  - Prefix documents with "passage: " at ingest time (chunker_service.py)
  - Prefix queries   with "query: "   at retrieval time (rag_backend.py)
  This asymmetric prefix is required by multilingual-e5-large's training objective
  and significantly improves retrieval precision.

Embeddings are L2-normalized (unit vectors), so cosine similarity == dot product.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("embed-server")
logging.basicConfig(level=logging.INFO)

MODEL_NAME = "intfloat/multilingual-e5-large"

model: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")
    yield
    model = None


app = FastAPI(lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str]
    batch_size: int = 32


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return JSONResponse({"status": "healthy"})


@app.post("/embed")
def embed(request: EmbedRequest):
    """
    Embed a batch of texts. Returns L2-normalized vectors.
    Expected input size: up to 512 tokens per text (multilingual-e5-large limit).
    Typical batch: 32 chunks of ~400 tokens each.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list is empty")

    try:
        embeddings = model.encode(
            request.texts,
            batch_size=request.batch_size,
            normalize_embeddings=True,  # L2-normalize so cosine sim == dot product
            show_progress_bar=False,
        )
        return JSONResponse({
            "embeddings": embeddings.tolist(),
            "dim": embeddings.shape[1],
        })
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
