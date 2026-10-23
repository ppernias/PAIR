"""Backends simulados para probar PAIR sin LM Studio ni OpenAI."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from pair.app import create_app
from pair.config import Backend, Config, Images

RULES = __import__("yaml").safe_load((Path(__file__).parent.parent / "rules.yaml").read_text(encoding="utf-8"))


class FakeBackend:
    """Imita un servidor OpenAI-compatible. Guarda las peticiones recibidas."""

    def __init__(self, name: str, model: str, reply: str = "hola", tool_call: dict | None = None, down: bool = False):
        self.name, self.model, self.reply, self.tool_call, self.down = name, model, reply, tool_call, down
        self.requests: list[dict] = []

    def _message(self) -> dict:
        msg = {"role": "assistant", "content": None if self.tool_call else f"{self.reply} desde {self.name}"}
        if self.tool_call:
            msg["tool_calls"] = [{"id": "call_1", "type": "function", "function": {
                "name": self.tool_call["name"], "arguments": json.dumps(self.tool_call["args"])}}]
        return msg

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("caído", request=request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": self.model}]})
        body = json.loads(request.content)
        self.requests.append(body)
        if request.url.path.endswith("/images/generations"):
            return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "Q0xPVUQ="}]})
        finish = "tool_calls" if self.tool_call else "stop"
        if not body.get("stream"):
            return httpx.Response(200, json={
                "id": "x", "object": "chat.completion", "model": body["model"],
                "choices": [{"index": 0, "message": self._message(), "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}})
        chunks = []
        if self.tool_call:
            args = json.dumps(self.tool_call["args"])
            chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": self.tool_call["name"], "arguments": args[:5]}}]}}]})
            chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": args[5:]}}]}}]})
        else:
            for word in f"{self.reply} desde {self.name}".split(" "):
                chunks.append({"choices": [{"index": 0, "delta": {"content": word + " "}}]})
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        chunks.append({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 50}})
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})


class FakeFlux:
    """Imita el generador local: POST /generar devuelve los bytes de una imagen."""

    def __init__(self, down: bool = False):
        self.down = down
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("caído", request=request)
        self.requests.append(json.loads(request.content))
        return httpx.Response(200, content=b"PNGLOCAL", headers={"content-type": "image/png"})


@pytest.fixture
def make_client(tmp_path):
    def _make(local: FakeBackend | None = None, cloud: FakeBackend | None = None, mode: str = "equilibrado",
              flux: FakeFlux | None = None):
        local = local or FakeBackend("local", "qwen-local")
        cloud = cloud or FakeBackend("cloud", "gpt-test")
        flux = flux or FakeFlux()
        cfg = Config(
            mode=mode,
            local=Backend("lmstudio", "http://local.test/v1", max_context_tokens=16000),
            cloud=Backend("openai", "https://cloud.test/v1", "gpt-test",
                          price_input_per_mtok=1.0, price_output_per_mtok=10.0),
            log_path=str(tmp_path / "pair.db"),
            rules=RULES,
            images=Images(local_url="http://flux.test", cloud_model="gpt-image-test", cloud_price_per_image=0.04),
        )
        app = create_app(cfg, local_transport=httpx.MockTransport(local.handler),
                         cloud_transport=httpx.MockTransport(cloud.handler),
                         image_transport=httpx.MockTransport(flux.handler))
        client = TestClient(app)
        client.flux = flux
        return client, local, cloud
    return _make
