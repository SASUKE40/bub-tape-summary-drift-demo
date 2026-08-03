from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from any_llm import AnyLLM
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLM:
    """Small provider-neutral JSON client using Bub's any-llm dependency."""

    def __init__(self, model: str, *, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.provider, self.model_id = AnyLLM.split_model_provider(model)

        provider_key = f"BUB_{self.provider.value.upper()}_API_KEY"
        provider_base = f"BUB_{self.provider.value.upper()}_API_BASE"
        api_key = os.getenv(provider_key) or os.getenv("BUB_API_KEY")
        api_base = os.getenv(provider_base) or os.getenv("BUB_API_BASE")
        self.client = AnyLLM.create(self.provider, api_key=api_key, api_base=api_base)

    async def json(self, *, system: str, user: str, schema: type[T]) -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        response = await self.client.acompletion(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"{user}\n\nReturn JSON only. It must validate against "
                        f"this JSON Schema:\n{schema_json}"
                    ),
                },
            ],
            temperature=self.temperature,
            max_tokens=4000,
            stream=False,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise TypeError("model returned non-text content")
        return schema.model_validate(self._parse_json(content))

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:-1] if len(lines) >= 3 else lines
            text = "\n".join(lines)
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise TypeError("expected a JSON object")
        return LLM._normalize_grade_statuses(value)

    @staticmethod
    def _normalize_grade_statuses(value: dict[str, Any]) -> dict[str, Any]:
        """Conservatively map common judge variants onto the strict rubric."""
        assessments = value.get("assessments")
        if not isinstance(assessments, list):
            return value
        allowed = {"preserved", "contradicted", "missing"}
        for item in assessments:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).casefold().replace("-", "_").replace(" ", "_")
            if status in allowed:
                item["status"] = status
            elif "contradict" in status:
                item["status"] = "contradicted"
            else:
                # Partial/ambiguous retention does not earn preservation credit.
                item["status"] = "missing"
        return value
