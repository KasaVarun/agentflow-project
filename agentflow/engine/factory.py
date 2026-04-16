"""
Engine factory for AgentFlow.
Supports two backends:
  - together-<model>: Together AI API (used for fixed engine: executor, verifier, generator)
  - vllm-<model>: vLLM OpenAI-compatible server on Modal (used for planner)
"""
from typing import Any
import os


def create_llm_engine(model_string: str, use_cache: bool = False, is_multimodal: bool = False, **kwargs) -> Any:
    """
    Factory function to create appropriate LLM engine instance.

    Prefixes:
      - "together-" -> Together AI backend
      - "vllm-"     -> vLLM (OpenAI-compatible) backend
    """
    original_model_string = model_string

    # === Together AI ===
    if "together" in model_string:
        from .together import ChatTogether

        if "TOGETHER_API_KEY" not in os.environ:
            raise ValueError("Please set the TOGETHER_API_KEY environment variable.")

        model_string = model_string.replace("together-", "")
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
        }
        return ChatTogether(**config)

    # === vLLM (OpenAI-compatible, served on Modal) ===
    elif "vllm" in model_string:
        from .vllm import ChatVLLM

        model_string = model_string.replace("vllm-", "")
        config = {
            "model_string": model_string,
            "base_url": kwargs.get("base_url", os.environ.get("PLANNER_BASE_URL", "http://localhost:8000/v1")),
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 0.9),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
        }
        return ChatVLLM(**config)

    # === Local LoRA (HuggingFace + PEFT) ===
    elif model_string == "local-LoRA-Qwen2.5-0.5B-GRPO":
        from .local_lora import LocalLoRAEngine

        return LocalLoRAEngine(
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            lora_path=os.environ.get(
                "AGENTFLOW_LORA_CHECKPOINT",
                "checkpoints/flow_grpo_0.5b_step300",
            ),
        )

    else:
        raise ValueError(
            f"Engine '{original_model_string}' not supported. "
            "Use 'together-<model>' for Together AI or 'vllm-<model>' for vLLM on Modal."
        )
