"""
Modal deployment for Qwen3.5-0.8B-Instruct on A10G.
Step 3 experiment: smallest Qwen3.5 model.

Usage:
    modal deploy modal_deploy/serve_qwen35_0.8b.py

After deployment, set in .env:
    PLANNER_BASE_URL=https://varunkasa77--agentflow-planner-qwen35-0.8b-serve.modal.run/v1

Cost: ~$1.10/hr on A10G.
"""
import modal

MODEL_NAME = "Qwen/Qwen3.5-0.8B-Instruct"
GPU_CONFIG = "A10G"
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.85
IDLE_TIMEOUT = 300

app = modal.App("agentflow-planner-qwen35-0.8b")

results_volume = modal.Volume.from_name("agentflow-results", create_if_missing=True)

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
)


@app.function(
    gpu=GPU_CONFIG,
    image=vllm_image,
    timeout=3600,
    scaledown_window=IDLE_TIMEOUT,
    volumes={"/results": results_volume},
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
        "--served-model-name", MODEL_NAME,
    ]
    proc = subprocess.Popen(vllm_cmd)

    ready = False
    for _ in range(180):
        time.sleep(1)
        try:
            r = req_lib.get("http://127.0.0.1:8001/health", timeout=2)
            if r.status_code == 200:
                ready = True
                print(f"vLLM server ready: {MODEL_NAME}")
                break
        except Exception:
            pass
    if not ready:
        proc.terminate()
        raise RuntimeError("vLLM failed to start within 3 minutes")

    proxy = FastAPI(title=f"AgentFlow Planner {MODEL_NAME}")
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
