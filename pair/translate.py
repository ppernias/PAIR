"""Traducción entre los formatos de Anthropic, Gemini y el formato canónico (OpenAI).

El router y los backends solo hablan chat completions de OpenAI; aquí se
convierten las peticiones de entrada y las respuestas de salida.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

# ======================================================================= utilidades


def _json_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"value": val}
    except json.JSONDecodeError:
        return {"_raw": raw}


def _sse(event: str | None, data: dict[str, Any]) -> bytes:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


class StreamStats:
    """Acumula lo necesario para el registro mientras pasa un stream."""

    def __init__(self) -> None:
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.chars = 0
        self.reasoning_tokens = 0

    def see(self, chunk: dict[str, Any]) -> None:
        usage = chunk.get("usage")
        if usage:
            self.prompt_tokens = usage.get("prompt_tokens", self.prompt_tokens)
            self.completion_tokens = usage.get("completion_tokens", self.completion_tokens)
            details = usage.get("completion_tokens_details") or {}
            self.reasoning_tokens = details.get("reasoning_tokens") or self.reasoning_tokens
        for choice in chunk.get("choices", []) or []:
            delta = choice.get("delta") or {}
            self.chars += len(delta.get("content") or "")
            for tc in delta.get("tool_calls") or []:
                self.chars += len((tc.get("function") or {}).get("arguments") or "")


# ======================================================================= Anthropic

_ANTHROPIC_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                   "function_call": "tool_use", "content_filter": "end_turn"}


def _anthropic_text(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    return "\n".join(b.get("text", "") for b in blocks or [] if isinstance(b, dict) and b.get("type") == "text")


def _anthropic_image(block: dict[str, Any]) -> dict[str, Any]:
    src = block.get("source", {})
    if src.get("type") == "base64":
        url = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
    else:
        url = src.get("url", "")
    return {"type": "image_url", "image_url": {"url": url}}


def anthropic_to_openai(req: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = req.get("system")
    if system:
        messages.append({"role": "system", "content": _anthropic_text(system)})

    for msg in req.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif kind == "image":
                parts.append(_anthropic_image(block))
            elif kind == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {"name": block.get("name"),
                                 "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)},
                })
            elif kind == "tool_result":
                result = block.get("content")
                text = _anthropic_text(result) if not isinstance(result, str) else result
                if block.get("is_error"):
                    text = f"[error] {text}"
                tool_results.append({"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": text})
            # thinking, redacted_thinking, documents...: se ignoran en esta fase

        # Los resultados de herramientas van antes que el texto que los acompaña.
        messages.extend(tool_results)
        if role == "assistant":
            out: dict[str, Any] = {"role": "assistant",
                                   "content": "\n".join(p["text"] for p in parts if p["type"] == "text") or None}
            if tool_calls:
                out["tool_calls"] = tool_calls
            if out["content"] is not None or tool_calls:
                messages.append(out)
        elif parts:
            only_text = all(p["type"] == "text" for p in parts)
            messages.append({"role": role,
                             "content": "\n".join(p["text"] for p in parts) if only_text else parts})

    out_req: dict[str, Any] = {"messages": messages}
    if "max_tokens" in req:
        out_req["max_tokens"] = req["max_tokens"]
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("stop_sequences", "stop")):
        if req.get(src) is not None:
            out_req[dst] = req[src]
    if req.get("tools"):
        out_req["tools"] = [
            {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t.get("input_schema", {"type": "object"})}}
            for t in req["tools"] if t.get("name") and "input_schema" in t  # ignora herramientas de servidor
        ]
        if not out_req["tools"]:
            del out_req["tools"]
    choice = req.get("tool_choice")
    if choice and "tools" in out_req:
        kind = choice.get("type")
        out_req["tool_choice"] = {"auto": "auto", "any": "required", "none": "none"}.get(kind) or \
            {"type": "function", "function": {"name": choice.get("name")}}
    return out_req


def openai_to_anthropic(resp: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        content.append({"type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": fn.get("name"), "input": _json_args(fn.get("arguments"))})
    usage = resp.get("usage") or {}
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _ANTHROPIC_STOP.get(choice.get("finish_reason") or "stop", "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
    }


async def openai_stream_to_anthropic(chunks: AsyncIterator[dict[str, Any]], model: str,
                                     stats: StreamStats) -> AsyncIterator[bytes]:
    yield _sse("message_start", {"type": "message_start", "message": {
        "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}}})

    index = -1                      # índice del bloque de contenido abierto
    text_open = False
    tool_blocks: dict[int, int] = {}  # índice de tool_call de OpenAI -> índice de bloque
    finish = "stop"

    def close_current() -> bytes | None:
        return _sse("content_block_stop", {"type": "content_block_stop", "index": index}) if index >= 0 else None

    async for chunk in chunks:
        stats.see(chunk)
        for choice in chunk.get("choices", []) or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if not text_open:
                    if (c := close_current()):
                        yield c
                    index += 1
                    text_open = True
                    yield _sse("content_block_start", {"type": "content_block_start", "index": index,
                                                       "content_block": {"type": "text", "text": ""}})
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                   "delta": {"type": "text_delta", "text": delta["content"]}})
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                fn = tc.get("function") or {}
                if i not in tool_blocks:
                    if (c := close_current()):
                        yield c
                    text_open = False
                    index += 1
                    tool_blocks[i] = index
                    yield _sse("content_block_start", {"type": "content_block_start", "index": index,
                                                       "content_block": {"type": "tool_use",
                                                                         "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                                                                         "name": fn.get("name", ""), "input": {}}})
                if fn.get("arguments"):
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": tool_blocks[i],
                                                       "delta": {"type": "input_json_delta",
                                                                 "partial_json": fn["arguments"]}})
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]

    if (c := close_current()):
        yield c
    yield _sse("message_delta", {"type": "message_delta",
                                 "delta": {"stop_reason": _ANTHROPIC_STOP.get(finish, "end_turn"), "stop_sequence": None},
                                 "usage": {"output_tokens": stats.completion_tokens or stats.chars // 4}})
    yield _sse("message_stop", {"type": "message_stop"})


# ======================================================================= Gemini

_GEMINI_STOP = {"stop": "STOP", "length": "MAX_TOKENS", "tool_calls": "STOP", "content_filter": "SAFETY"}


def _lower_types(schema: Any) -> Any:
    """Gemini admite tipos en mayúsculas (OBJECT, STRING); OpenAI los quiere en minúsculas."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.lower()
            elif k in ("nullable", "propertyOrdering"):
                continue
            else:
                out[k] = _lower_types(v)
        return out
    if isinstance(schema, list):
        return [_lower_types(v) for v in schema]
    return schema


def gemini_to_openai(req: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = req.get("systemInstruction") or req.get("system_instruction")
    if system:
        text = "\n".join(p.get("text", "") for p in system.get("parts", []))
        if text:
            messages.append({"role": "system", "content": text})

    pending: dict[str, list[str]] = {}   # nombre de función -> ids de llamadas sin respuesta
    for content in req.get("contents", []):
        role = "assistant" if content.get("role") == "model" else "user"
        parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        for part in content.get("parts", []):
            if "text" in part and not part.get("thought"):
                parts.append({"type": "text", "text": part["text"]})
            elif "inlineData" in part or "inline_data" in part:
                data = part.get("inlineData") or part.get("inline_data")
                mime = data.get("mimeType") or data.get("mime_type", "image/png")
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data.get('data', '')}"}})
            elif "functionCall" in part:
                call = part["functionCall"]
                cid = call.get("id") or f"call_{uuid.uuid4().hex[:16]}"
                pending.setdefault(call.get("name", ""), []).append(cid)
                tool_calls.append({"id": cid, "type": "function", "function": {
                    "name": call.get("name"), "arguments": json.dumps(call.get("args", {}), ensure_ascii=False)}})
            elif "functionResponse" in part:
                resp = part["functionResponse"]
                name = resp.get("name", "")
                cid = resp.get("id") or (pending.get(name) or [f"call_{name}"]).pop(0)
                messages.append({"role": "tool", "tool_call_id": cid,
                                 "content": json.dumps(resp.get("response", {}), ensure_ascii=False)})
        if role == "assistant":
            out: dict[str, Any] = {"role": "assistant",
                                   "content": "\n".join(p["text"] for p in parts if p["type"] == "text") or None}
            if tool_calls:
                out["tool_calls"] = tool_calls
            if out["content"] is not None or tool_calls:
                messages.append(out)
        elif parts:
            only_text = all(p["type"] == "text" for p in parts)
            messages.append({"role": "user", "content": "\n".join(p["text"] for p in parts) if only_text else parts})

    out_req: dict[str, Any] = {"messages": messages}
    gen = req.get("generationConfig") or req.get("generation_config") or {}
    for src, dst in (("temperature", "temperature"), ("topP", "top_p"),
                     ("maxOutputTokens", "max_tokens"), ("stopSequences", "stop")):
        if gen.get(src) is not None:
            out_req[dst] = gen[src]
    tools = []
    for tool in req.get("tools", []) or []:
        for fd in tool.get("functionDeclarations") or tool.get("function_declarations") or []:
            tools.append({"type": "function", "function": {
                "name": fd["name"], "description": fd.get("description", ""),
                "parameters": _lower_types(fd.get("parameters") or fd.get("parametersJsonSchema")
                                           or {"type": "object", "properties": {}})}})
    if tools:
        out_req["tools"] = tools
    return out_req


def _gemini_usage(prompt: int, completion: int) -> dict[str, int]:
    return {"promptTokenCount": prompt, "candidatesTokenCount": completion, "totalTokenCount": prompt + completion}


def openai_to_gemini(resp: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    parts: list[dict[str, Any]] = []
    if msg.get("content"):
        parts.append({"text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        parts.append({"functionCall": {"id": tc.get("id"), "name": fn.get("name"), "args": _json_args(fn.get("arguments"))}})
    usage = resp.get("usage") or {}
    return {
        "candidates": [{"content": {"role": "model", "parts": parts or [{"text": ""}]},
                        "finishReason": _GEMINI_STOP.get(choice.get("finish_reason") or "stop", "STOP"), "index": 0}],
        "usageMetadata": _gemini_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
        "modelVersion": model,
    }


async def openai_stream_to_gemini(chunks: AsyncIterator[dict[str, Any]], model: str,
                                  stats: StreamStats) -> AsyncIterator[bytes]:
    calls: dict[int, dict[str, str]] = {}
    finish = "stop"
    async for chunk in chunks:
        stats.see(chunk)
        for choice in chunk.get("choices", []) or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield _sse(None, {"candidates": [{"content": {"role": "model", "parts": [{"text": delta["content"]}]},
                                                  "index": 0}], "modelVersion": model})
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["id"] = tc.get("id") or slot["id"]
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]

    parts = [{"functionCall": {"id": c["id"] or None, "name": c["name"], "args": _json_args(c["arguments"])}}
             for _, c in sorted(calls.items())]
    prompt = stats.prompt_tokens or 0
    completion = stats.completion_tokens or stats.chars // 4
    yield _sse(None, {"candidates": [{"content": {"role": "model", "parts": parts or [{"text": ""}]},
                                      "finishReason": _GEMINI_STOP.get(finish, "STOP"), "index": 0}],
                      "usageMetadata": _gemini_usage(prompt, completion), "modelVersion": model})


# ======================================================================= OpenAI (paso directo)

async def openai_stream_passthrough(chunks: AsyncIterator[dict[str, Any]], stats: StreamStats) -> AsyncIterator[bytes]:
    async for chunk in chunks:
        stats.see(chunk)
        yield _sse(None, chunk)
    yield b"data: [DONE]\n\n"


def now_ms() -> int:
    return int(time.time() * 1000)
