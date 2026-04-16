"""
Local LoRA-backed engine for AgentFlow.
Uses a HuggingFace AutoModelForCausalLM + PEFT LoRA adapter for inference.
Supports optional Pydantic response_format similar to Together/vLLM engines.
"""
from typing import Any, List, Union

import os

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .base import EngineLM


class LocalLoRAEngine(EngineLM):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        lora_path: str = "checkpoints/flow_grpo_0.5b_step300",
        system_prompt: str = None,
        max_new_tokens: int = 512,
    ):
        self.model_string = base_model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.lora_path = lora_path
        self.max_new_tokens = max_new_tokens

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_string)
        base = AutoModelForCausalLM.from_pretrained(self.model_string, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
        self.model = PeftModel.from_pretrained(base, self.lora_path)
        self.model.to(device)
        self.model.eval()

    def _build_prompt(self, prompt: str, system_prompt: str | None) -> str:
        sys_prompt = system_prompt or self.system_prompt
        return f"{sys_prompt}\n\n{prompt}"

    @staticmethod
    def _parse_pydantic(text: str, cls):
        """Parse a JSON string into a Pydantic model, with fallback."""
        import json

        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            data = json.loads(cleaned)
            return cls(**data)
        except Exception:
            return text

    def generate(self, content: Union[str, List[Any]], system_prompt: str | None = None, **kwargs):
        if isinstance(content, list):
            if all(isinstance(item, str) for item in content):
                prompt = "\n".join(content)
            else:
                raise ValueError("LocalLoRAEngine only supports text inputs.")
        else:
            prompt = str(content)

        response_format = kwargs.get("response_format", None)

        # If response_format is a Pydantic model class, add JSON instructions
        pydantic_cls = None
        if response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel):
            pydantic_cls = response_format
            schema = pydantic_cls.model_json_schema()
            fields = schema.get("properties", {})
            field_descriptions = []
            for name, info in fields.items():
                field_type = info.get("type", "string")
                field_descriptions.append(f'  "{name}": <{field_type}>')
            json_template = "{\n" + ",\n".join(field_descriptions) + "\n}"
            prompt += f"\n\nIMPORTANT: Respond with ONLY a valid JSON object matching this schema (no markdown, no extra text):\n{json_template}"

        full_prompt = self._build_prompt(prompt, system_prompt)

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        max_new_tokens = kwargs.get("max_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 0.95)

        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "temperature": max(temperature, 1e-5),
                "top_p": top_p,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            outputs = self.model.generate(**inputs, **generate_kwargs)

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if pydantic_cls:
            return self._parse_pydantic(response_text, pydantic_cls)

        return response_text

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

