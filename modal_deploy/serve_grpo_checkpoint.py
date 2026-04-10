"""
Modal deployment for Flow-GRPO trained LoRA checkpoint.
Merges LoRA weights into base model and serves via vLLM.

Usage:
    modal deploy modal_deploy/serve_grpo_checkpoint.py

After deployment, set PLANNER_BASE_URL to the printed URL + /v1.

Cost: ~$1.10/hr on A10G. Stop app when done.
"""
import modal

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT_NAME = "flow_grpo_0.5b"
SERVED_MODEL_NAME = "flow-grpo-0.5b"
GPU_CONFIG = "A10G"
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.85
IDLE_TIMEOUT = 300

app = modal.App("agentflow-grpo-checkpoint")

checkpoints_volume = modal.Volume.from_name("agentflow-checkpoints", create_if_missing=False)
results_volume = modal.Volume.from_name("agentflow-results", create_if_missing=True)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8.0",
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "peft>=0.14.0",
        "huggingface_hub>=0.25.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "httpx>=0.27.0",
    )
    .run_commands(
        "python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen2.5-0.5B-Instruct\")'",
    )
)


@app.function(
    gpu=GPU_CONFIG,
    image=vllm_image,
    timeout=3600,
    scaledown_window=IDLE_TIMEOUT,
    volumes={
        "/checkpoints": checkpoints_volume,
        "/results": results_volume,
    },
)
@modal.asgi_app()
def serve():
    import subprocess
    import time
    import os
    import requests as req_lib
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import httpx

    # Merge LoRA into base model and save to /tmp/merged
    merged_path = "/tmp/merged_model"
    checkpoint_path = f"/checkpoints/{CHECKPOINT_NAME}/latest"

    if not os.path.exists(merged_path):
        print(f"Merging LoRA from {checkpoint_path} into {BASE_MODEL}...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, checkpoint_path)
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"Merged model saved to {merged_path}")

    vllm_cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", merged_path,
        "--port", "8001",
        "--host", "127.0.0.1",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--trust-remote-code",
        "--served-model-name", SERVED_MODEL_NAME,
        "--enforce-eager",
    ]
    proc = subprocess.Popen(vllm_cmd)

    ready = False
    for _ in range(600):
        time.sleep(1)
        try:
            r = req_lib.get("http://127.0.0.1:8001/health", timeout=2)
            if r.status_code == 200:
                ready = True
                print(f"vLLM ready: {SERVED_MODEL_NAME}")
                break
        except Exception:
            pass
    if not ready:
        proc.terminate()
        raise RuntimeError("vLLM failed to start within 10 minutes")

    proxy = FastAPI(title=f"AgentFlow GRPO {SERVED_MODEL_NAME}")
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
