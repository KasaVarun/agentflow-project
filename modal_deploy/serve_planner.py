"""
Modal deployment for AgentFlow planner model.
Exposes vLLM as a full OpenAI-compatible API server.

Usage:
    modal serve modal_deploy/serve_planner.py    # temporary (testing)
    modal deploy modal_deploy/serve_planner.py   # persistent

    After deployment, copy the printed URL to .env:
        PLANNER_BASE_URL=https://your-org--agentflow-planner-serve.modal.run/v1

    Test it:
        python -c "
        from openai import OpenAI
        c = OpenAI(api_key='dummy', base_url='<URL>/v1')
        r = c.chat.completions.create(model='Qwen/Qwen2.5-7B-Instruct',
            messages=[{'role':'user','content':'Hello'}], max_tokens=20)
        print(r.choices[0].message.content)
        "

Costs: A10G = ~$1.10/hr. Container shuts down after IDLE_TIMEOUT seconds idle.

To change model: edit MODEL_NAME below and redeploy.
  Step 1-2: MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
  Step 3:   MODEL_NAME = "Qwen/Qwen3.5-0.8B-Instruct" (or 2B / 4B / 9B)
  Step 5-6: Use train_flow_grpo.py instead
"""
import modal

# ============================================================
# CONFIGURATION: Edit for each experiment, then redeploy
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
GPU_CONFIG = "A10G"          # Modal 1.x: string, not modal.gpu.A10G()
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.90
IDLE_TIMEOUT = 300  # seconds idle before auto-shutdown
# ============================================================

app = modal.App("agentflow-planner")

# Persistent volume for benchmark results collected during inference runs.
# Results written to /results/ survive container shutdown and preemption.
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
    scaledown_window=IDLE_TIMEOUT,   # Modal 1.x: replaces container_idle_timeout
    volumes={"/results": results_volume},
)
@modal.asgi_app()
def serve():
    """
    Expose vLLM's full OpenAI-compatible API as a Modal ASGI endpoint.
    The OpenAI Python client can point directly to the returned URL + /v1.
    """
    import subprocess
    import time
    import requests as req_lib
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import httpx

    # ---- start vLLM in background ----
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

    # Wait up to 3 minutes for model to load
    ready = False
    for _ in range(360):
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
        raise RuntimeError("vLLM failed to start within 6 minutes")

    # ---- proxy FastAPI app ----
    proxy = FastAPI(title="AgentFlow Planner")
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
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        return JSONResponse(content=resp.text, status_code=resp.status_code)

    return proxy


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("AgentFlow Planner - Modal Deployment")
    print("=" * 60)
    print(f"Model:        {MODEL_NAME}")
    print(f"GPU:          {GPU_CONFIG}")
    print(f"Idle timeout: {IDLE_TIMEOUT}s")
    print()
    print("Deploy commands:")
    print("  modal serve modal_deploy/serve_planner.py   # temporary")
    print("  modal deploy modal_deploy/serve_planner.py  # persistent")
    print()
    print("After deploy, set in .env:")
    print("  PLANNER_BASE_URL=https://<org>--agentflow-planner-serve.modal.run/v1")
