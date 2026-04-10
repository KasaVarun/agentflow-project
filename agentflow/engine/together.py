"""
Together AI engine for AgentFlow.
Uses the Together AI API (OpenAI-compatible) for inference.
Supports structured output via response_format (Pydantic models).
"""
import os
import json
import base64
import platformdirs
from typing import List, Union
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pydantic import BaseModel

from .base import EngineLM, CachedEngine

try:
    from together import Together
except ImportError:
    raise ImportError(
        "Install the together package: pip install together"
    )


class ChatTogether(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="Qwen/Qwen2.5-7B-Instruct-Turbo",
        use_cache: bool = False,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
    ):
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.model_string = model_string
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("agentflow")
            cache_path = os.path.join(root, f"cache_together_{model_string}.db")
            super().__init__(cache_path=cache_path)

        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("Please set the TOGETHER_API_KEY environment variable.")

        self.client = Together(api_key=api_key)

    def _format_content(self, content):
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            elif isinstance(item, str):
                formatted_content.append({"type": "text", "text": item})
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content, system_prompt=None, max_tokens=4000, top_p=0.99, **kwargs):
        try:
            if isinstance(content, str):
                return self._generate_text(
                    content, system_prompt=system_prompt,
                    max_tokens=max_tokens, top_p=top_p, **kwargs
                )
            elif isinstance(content, list):
                if all(isinstance(item, str) for item in content):
                    full_text = "\n".join(content)
                    return self._generate_text(
                        full_text, system_prompt=system_prompt,
                        max_tokens=max_tokens, top_p=top_p, **kwargs
                    )
                else:
                    raise ValueError("Unsupported content type in list.")
        except Exception as e:
            print(f"Error in generate: {type(e).__name__}: {e}")
            return {"error": type(e).__name__, "message": str(e)}

    def _generate_text(
        self, prompt, system_prompt=None, temperature=0.0, max_tokens=4000, top_p=0.99, **kwargs
    ):
        sys_prompt = system_prompt or self.system_prompt
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

        if self.use_cache:
            cache_key = sys_prompt + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                if pydantic_cls:
                    return self._parse_pydantic(cache_or_none, pydantic_cls)
                return cache_or_none

        api_kwargs = {
            "model": self.model_string,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": None,
        }

        # Use JSON mode if we have a Pydantic response format
        if pydantic_cls is not None:
            api_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**api_kwargs)
        response_text = response.choices[0].message.content

        if self.use_cache:
            self._save_cache(sys_prompt + prompt, response_text)

        if pydantic_cls:
            return self._parse_pydantic(response_text, pydantic_cls)

        return response_text

    @staticmethod
    def _parse_pydantic(text: str, cls):
        """Parse a JSON string into a Pydantic model, with fallback."""
        try:
            # Strip markdown code fences if present
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            data = json.loads(cleaned)
            return cls(**data)
        except Exception:
            # Return raw text so the caller's regex fallback can handle it
            return text

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
