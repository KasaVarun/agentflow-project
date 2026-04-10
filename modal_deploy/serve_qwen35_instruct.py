"""
Modal deployment for Qwen3.5-Instruct models with thinking mode disabled.
Parameterized by MODEL_SIZE env var.

Thinking mode disabled via:
  --override-generation-config '{"enable_thinking": false}'

Sizes supported: 0.8B, 2B, 4B (A10G) | 9B, 27B (A100 + --enforce-eager)

Usage:
    MODEL_SIZE=0.8B PYTHONUTF8=1 modal deploy modal_deploy/serve_qwen35_instruct.py
"""
import os
import modal

MODEL_SIZE = os.environ.get("MODEL_SIZE", "0.8B")
# Qwen3.5 has no -Instruct suffix; the plain name IS the instruct model
MODEL_NAME = f"Qwen/Qwen3.5-{MODEL_SIZE}"
APP_NAME = f"agentflow-qwen35-{MODEL_SIZE.lower().replace('.', '')}"
SERVED_MODEL_NAME = MODEL_NAME

# A100 for 9B+ (--enforce-eager to cut startup), A10G for smaller
if MODEL_SIZE in ("9B", "27B"):
    GPU_CONFIG = "A100"
    EXTRA_ARGS = ["--enforce-eager"]
else:
    GPU_CONFIG = "A10G"
    EXTRA_ARGS = []

MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.88
IDLE_TIMEOUT = 300

app = modal.App(APP_NAME)

results_volume = modal.Volume.from_name("agentflow-results", create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8.0",
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "huggingface_hub>=0.25.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "httpx>=0.27.0",
    )
    .run_commands(
        f"python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"{MODEL_NAME}\")'",
    )
)


@app.function(
    gpu=GPU_CONFIG,
    image=vllm_image,
    timeout=3600,
    scaledown_window=IDLE_TIMEOUT,
    volumes={"/results": results_volume},
    secrets=[hf_secret],
)
@modal.asgi_app()
def serve():
    import subprocess
    import time
    import requests as req_lib
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import httpx

    vllm_cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", "8001",
        "--host", "127.0.0.1",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--trust-remote-code",
        "--served-model-name", SERVED_MODEL_NAME,
        "--override-generation-config", '{"enable_thinking": false}',
    ] + EXTRA_ARGS
    proc = subprocess.Popen(vllm_cmd)

    ready = False
    for _ in range(600):
        time.sleep(1)
        try:
            r = req_lib.get("http://127.0.0.1:8001/health", timeout=2)
            if r.status_code == 200:
                ready = True
                print(f"vLLM ready: {MODEL_NAME}")
                break
        except Exception:
            pass
    if not ready:
        proc.terminate()
        raise RuntimeError(f"vLLM failed to start within 10 minutes for {MODEL_NAME}")

    proxy = FastAPI(title=f"AgentFlow {MODEL_NAME}")
    http_client = httpx.AsyncClient(base_url="http://127.0.0.1:8001", timeout=120.0)

    @proxy.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT"])
    async def forward(path: str, request: Request):
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        resp = await http_client.request(
            method=request.method,
            url=f"/{path}",
            content=body,
            headers=headers,
            params=dict(request.query_params),
        )
        if "application/json" in resp.headers.get("content-type", ""):
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        return JSONResponse(content=resp.text, status_code=resp.status_code)

    return proxy
