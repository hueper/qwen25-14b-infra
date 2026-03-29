#!/bin/bash
set -e

echo "=== entrypoint.sh starting, vllm version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>&1) ==="
echo "=== CUDA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1) ==="

# Start embedding server (port 8001) in background
python3 /opt/embed_server.py &

# Start vLLM (port 8000) — Qwen2.5 14B GPTQ INT4, single GPU
python3 -m vllm.entrypoints.openai.api_server \
  --model "${HF_MODEL_ID}" \
  --tensor-parallel-size "${SM_NUM_GPUS:-1}" \
  --port 8000 \
  --max-model-len 32768 \
  --quantization gptq \
  --dtype auto \
  --gpu-memory-utilization 0.82 \
  --enforce-eager \
  --trust-remote-code &

exec python3 /opt/serve.py
