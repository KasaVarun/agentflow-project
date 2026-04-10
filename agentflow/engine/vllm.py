"""
vLLM engine for AgentFlow.
Connects to a vLLM OpenAI-compatible server (hosted on Modal).
Supports structured output via response_format (Pydantic models).
"""
from openai import OpenAI

import os
import json
import base64
import platformdirs
from typing import List, Union
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from .base import EngineLM, CachedEngine


class ChatVLLM(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="Qwen/Qwen2.5-7B-Instruct",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        use_cache: bool = True,
        base_url=None,
        api_key=None,
        check_model: bool = True,
        **kwargs,
    ):
        self.model_string = model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("agentflow")
            cache_path = os.path.join(root, f"cache_vllm_{self.model_string}.db")
            os.makedirs(root, exist_ok=True)
            super().__init__(cache_path=cache_path)

        self.base_url = base_url or os.environ.get("PLANNER_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "dummy-token")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    @retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(3))
    def generate(self, content, system_prompt=None, **kwargs):
        try:
            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, **kwargs)
            elif isinstance(content, list):
                if all(isinstance(item, str) for item in content):
                    full_text = "\n".join(content)
                    return self._generate_text(full_text, system_prompt=system_prompt, **kwargs)
                else:
                    raise ValueError("Unsupported content in list.")
        except Exception as e:
            print(f"Error in vLLM generate: {type(e).__name__}: {e}")
            return {"error": type(e).__name__, "message": str(e)}

    def _generate_text(
        self, prompt, system_prompt=None, max_tokens=2048, top_p=0.99, **kwargs
    ):
        sys_prompt = system_prompt or self.system_prompt
        response_format = kwargs.get("response_format", None)

        # Handle Pydantic response_format
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

        if self.use_cache:
            cache_key = sys_prompt + prompt
            cached = self._check_cache(cache_key)
            if cached is not None:
                if pydantic_cls:
                    return self._parse_pydantic(cached, pydantic_cls)
                return cached

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=0,
            stop=None,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=max_tokens,
            top_p=top_p,
        )
        response_text = response.choices[0].message.content

        if self.use_cache:
            self._save_cache(sys_prompt + prompt, response_text)

        if pydantic_cls:
            return self._parse_pydantic(response_text, pydantic_cls)

        return response_text

    @staticmethod
    def _parse_pydantic(text: str, cls):
        """Parse JSON text into a Pydantic model, with fallback to raw text."""
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

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
