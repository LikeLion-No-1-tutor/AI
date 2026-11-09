"""Pydantic 모델 → 공급자가 받아들이는 '엄격한' JSON Schema.

Claude structured outputs 제약:
- 모든 object는 additionalProperties: false
- minLength/maxLength/minimum/maximum/multipleOf, minItems(0·1 외), maxItems 미지원
- 외부 $ref 미지원 ($defs 내부 참조는 가능하지만, Gemini 호환을 위해 전부 펼친다)
"""
from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

_UNSUPPORTED = {
    "title", "default", "minLength", "maxLength", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "maxItems", "format",
    "examples", "pattern",
}


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            name = ref.split("/")[-1]
            target = copy.deepcopy(defs[name])
            extra = {k: v for k, v in node.items() if k != "$ref"}
            target.update(extra)
            return _inline_refs(target, defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _strict(node: Any) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _UNSUPPORTED:
                continue
            if key == "minItems" and value not in (0, 1):
                continue
            out[key] = _strict(value)
        if out.get("type") == "object" or "properties" in out:
            out["type"] = "object"
            props = out.get("properties", {})
            out["properties"] = props
            out["required"] = list(props.keys())
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [_strict(v) for v in node]
    return node


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    defs = raw.get("$defs", {})
    return _strict(_inline_refs(raw, defs))
