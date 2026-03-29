"""
SageMaker ↔ vLLM/embedding adapter.

SageMaker sends inference requests to /invocations and health checks to /ping.
This adapter routes to one of two backends based on the "task" field:

  task="generate" → vLLM (port 8000), OpenAI-compatible chat completions
  task="embed"    → embed_server (port 8001), sentence-transformers

Input  (POST /invocations):
  Generate: {"task": "generate", "inputs": "...", "parameters": {"max_new_tokens": ..., "stream": true/false, ...}}
  Embed:    {"task": "embed", "texts": ["passage: text1", ...], "batch_size": 32}

Output:
  Generate: {"generated_text": "..."} or streaming text/plain tokens when stream=true
  Embed:    {"embeddings": [[...], ...], "dim": 1024}
"""

import json
import os
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

VLLM_BASE = "http://localhost:8000"
EMBED_BASE = "http://localhost:8001"
MODEL_NAME = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4")


def _wait_for_server(url: str, name: str, timeout: int = 600, interval: int = 5) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=5)
            if r.status_code == 200:
                print(f"{name} is ready.")
                return
        except Exception:
            pass
        print(f"Waiting for {name}...")
        time.sleep(interval)
    raise RuntimeError(f"{name} did not become ready in time.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _wait_for_server(f"{VLLM_BASE}/health", "vLLM")
    _wait_for_server(f"{EMBED_BASE}/health", "embed_server")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping():
    return JSONResponse({"status": "healthy"})


@app.post("/invocations")
async def invocations(body: dict):
    task = body.get("task")

    if task == "embed":
        return await _handle_embed(body)
    elif task == "generate":
        return await _handle_generate(body)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Missing or unknown 'task' field: {task!r}. Expected 'generate' or 'embed'.",
        )


async def _handle_embed(body: dict) -> JSONResponse:
    texts = body.get("texts")
    batch_size = body.get("batch_size", 32)

    if not texts:
        raise HTTPException(status_code=400, detail="'texts' field is required for task='embed'")

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{EMBED_BASE}/embed",
                json={"texts": texts, "batch_size": batch_size},
                timeout=120,
            )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return JSONResponse(r.json())


async def _handle_generate(body: dict):
    prompt = body.get("inputs", "")
    params = body.get("parameters", {})

    messages = [{"role": "user", "content": prompt}]

    if params.get("stream", False):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": params.get("max_new_tokens", 256),
            "temperature": params.get("temperature", 0.8),
            "top_p": params.get("top_p", 0.9),
            "stream": True,
        }
        stop = params.get("stop_sequences") or params.get("stop")
        if stop:
            payload["stop"] = stop

        async def stream_response():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{VLLM_BASE}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                ) as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            content = obj["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content.encode()
                        except Exception:
                            pass

        return StreamingResponse(stream_response(), media_type="text/plain")

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": params.get("max_new_tokens", 256),
        "temperature": params.get("temperature", 0.8),
        "top_p": params.get("top_p", 0.9),
        "repetition_penalty": params.get("repetition_penalty", 1.0),
    }

    stop = params.get("stop_sequences") or params.get("stop")
    if stop:
        payload["stop"] = stop

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{VLLM_BASE}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=str(e))

    result = r.json()
    generated_text = result["choices"][0]["message"]["content"]
    return {"generated_text": generated_text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
