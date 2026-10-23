"""Proxy HTTP de PAIR: tres puertas de entrada (OpenAI, Anthropic, Gemini), un router."""

from __future__ import annotations

import copy
import json
import time
import unicodedata
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import translate as tr
from .backends import BackendError, Client
from .config import Config
from .router import CLOUD, LOCAL, Decision, decide, estimate_tokens, last_user_text
from .store import Store


class Pair:
    """Decide, llama al backend (con fallback) y registra."""

    def __init__(self, cfg: Config, local_transport: httpx.AsyncBaseTransport | None = None,
                 cloud_transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self.clients = {
            LOCAL: Client(cfg.local, local_transport),
            CLOUD: Client(cfg.cloud, cloud_transport, is_cloud=True),
        }
        self.store = Store(cfg.log_path, cfg.cloud.price_input_per_mtok, cfg.cloud.price_output_per_mtok,
                           cfg.capture_prompts)

    async def close(self) -> None:
        for c in self.clients.values():
            await c.close()

    def route(self, req: dict[str, Any]) -> tuple[Decision, list[str], str]:
        prompt = last_user_text(req)          # antes de quitar etiquetas
        decision = decide(req, self.cfg.rules, self.cfg.mode, self.cfg.local.max_context_tokens)
        other = CLOUD if decision.target == LOCAL else LOCAL
        order = [decision.target] if decision.sensitive else [decision.target, other]
        return decision, order, prompt

    async def call(self, endpoint: str, req: dict[str, Any], stream: bool):
        """Devuelve (decision, target, model, resultado, fell_back, prompt, t0).

        `resultado` es un dict (sin stream) o un iterador de chunks (con stream).
        """
        t0 = time.monotonic()
        decision, order, prompt = self.route(req)
        last_error: BackendError | None = None
        for i, target in enumerate(order):
            client = self.clients[target]
            try:
                model = await client.model()
                body = copy.deepcopy(req)
                result = await (client.stream(body) if stream else client.complete(body))
                return decision, target, model, result, i > 0, prompt, t0
            except BackendError as e:
                last_error = e
                if not e.retriable:
                    break
        self.store.record(endpoint=endpoint, target=order[-1], decided=decision.target, reason=decision.reason,
                          model=None, sensitive=decision.sensitive, fell_back=len(order) > 1, stream=stream,
                          prompt_tokens=estimate_tokens(req), completion_tokens=0,
                          latency_ms=int((time.monotonic() - t0) * 1000), error=str(last_error))
        assert last_error is not None
        raise last_error

    def log(self, endpoint: str, req: dict[str, Any], decision: Decision, target: str, model: str,
            fell_back: bool, stream: bool, prompt: str, t0: float,
            prompt_tokens: int | None, completion_tokens: int | None, completion_chars: int = 0) -> None:
        self.store.record(
            endpoint=endpoint, target=target, decided=decision.target, reason=decision.reason, model=model,
            sensitive=decision.sensitive, fell_back=fell_back, stream=stream,
            prompt_tokens=prompt_tokens if prompt_tokens is not None else estimate_tokens(req),
            completion_tokens=completion_tokens if completion_tokens is not None else completion_chars // 4,
            latency_ms=int((time.monotonic() - t0) * 1000), prompt=prompt,
        )


def _route_header(decision: Decision, target: str, model: str, fell_back: bool) -> dict[str, str]:
    value = f"{target}; {decision.reason}" + ("; fallback" if fell_back else "")
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return {"X-PAIR-Route": ascii_value, "X-PAIR-Model": model}


def _error(endpoint: str, err: BackendError) -> JSONResponse:
    msg = f"PAIR no pudo completar la petición: {err}"
    if endpoint == "anthropic":
        body = {"type": "error", "error": {"type": "api_error", "message": msg}}
    elif endpoint == "gemini":
        body = {"error": {"code": 502, "message": msg, "status": "UNAVAILABLE"}}
    else:
        body = {"error": {"message": msg, "type": "pair_backend_error"}}
    status = err.status if err.status and 400 <= err.status < 500 else 502
    return JSONResponse(body, status_code=status)


DEBUG_FILE = Path("~/.pair/last_request.json").expanduser()


def describe(req: dict[str, Any]) -> str:
    """Resumen de la estructura de una petición: rol y tipo/tamaño de cada parte."""
    def part(p: Any) -> str:
        if isinstance(p, str):
            return f"texto({len(p)})"
        if isinstance(p, dict):
            kind = p.get("type") or next(iter(p), "?")
            return f"{kind}{sorted(k for k in p if k != 'type')}({len(json.dumps(p, ensure_ascii=False))})"
        return type(p).__name__

    lines = []
    for m in req.get("messages") or req.get("contents") or []:
        content = m.get("content", m.get("parts"))
        items = [part(content)] if isinstance(content, str) else [part(p) for p in content or []]
        lines.append(f"  {m.get('role', '?')}: {', '.join(items) or 'vacío'}")
    extras = sorted(k for k in req if k not in ("messages", "contents"))
    return "\n".join(lines + [f"  otros campos: {', '.join(extras)}"])


def create_app(cfg: Config, debug: bool = False, **transports: Any) -> FastAPI:
    pair = Pair(cfg, **transports)

    def inspect(endpoint: str, req: dict[str, Any]) -> None:
        if not debug:
            return
        DEBUG_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEBUG_FILE.write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[debug] petición {endpoint} (completa en {DEBUG_FILE}):\n{describe(req)}", flush=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await pair.close()

    app = FastAPI(title="PAIR — Personal AI Router", version="0.0.1", lifespan=lifespan)
    app.state.pair = pair

    async def handle(endpoint: str, canonical: dict[str, Any], stream: bool,
                     to_response: Callable[[dict[str, Any], str], dict[str, Any]],
                     to_stream: Callable[..., AsyncIterator[bytes]]):
        try:
            decision, target, model, result, fell_back, prompt, t0 = await pair.call(endpoint, canonical, stream)
        except BackendError as e:
            return _error(endpoint, e)
        headers = _route_header(decision, target, model, fell_back)

        if not stream:
            usage = result.get("usage") or {}
            pair.log(endpoint, canonical, decision, target, model, fell_back, False, prompt, t0,
                     usage.get("prompt_tokens"), usage.get("completion_tokens"),
                     len(((result.get("choices") or [{}])[0].get("message") or {}).get("content") or ""))
            return JSONResponse(to_response(result, model), headers=headers)

        stats = tr.StreamStats()

        async def body() -> AsyncIterator[bytes]:
            try:
                async for piece in to_stream(result, stats):
                    yield piece
            finally:
                pair.log(endpoint, canonical, decision, target, model, fell_back, True, prompt, t0,
                         stats.prompt_tokens, stats.completion_tokens, stats.chars)

        return StreamingResponse(body(), media_type="text/event-stream", headers=headers)

    # ------------------------------------------------------------ OpenAI
    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        req = await request.json()
        inspect("openai", req)
        return await handle("openai", req, bool(req.get("stream")),
                            lambda resp, model: resp,
                            lambda chunks, stats: tr.openai_stream_passthrough(chunks, stats))

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "pair", "object": "model", "owned_by": "pair"}]}

    # ------------------------------------------------------------ Anthropic
    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        req = await request.json()
        inspect("anthropic", req)
        requested = req.get("model", "pair")
        return await handle("anthropic", tr.anthropic_to_openai(req), bool(req.get("stream")),
                            lambda resp, model: tr.openai_to_anthropic(resp, requested),
                            lambda chunks, stats: tr.openai_stream_to_anthropic(chunks, requested, stats))

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        req = await request.json()
        return {"input_tokens": estimate_tokens(tr.anthropic_to_openai(req))}

    # ------------------------------------------------------------ Gemini
    @app.post("/v1beta/models/{model_action}")
    async def gemini(model_action: str, request: Request):
        model_name, _, action = model_action.partition(":")
        req = await request.json()
        if action == "countTokens":
            inner = req.get("generateContentRequest") or req
            return {"totalTokens": estimate_tokens(tr.gemini_to_openai(inner))}
        stream = action == "streamGenerateContent"
        inspect("gemini", req)
        if action not in ("generateContent", "streamGenerateContent"):
            return JSONResponse({"error": {"code": 404, "message": f"acción no soportada: {action}"}}, status_code=404)
        return await handle("gemini", tr.gemini_to_openai(req), stream,
                            lambda resp, model: tr.openai_to_gemini(resp, model_name),
                            lambda chunks, stats: tr.openai_stream_to_gemini(chunks, model_name, stats))

    # ------------------------------------------------------------ PAIR
    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": cfg.mode}

    @app.get("/pair/stats")
    async def stats(days: float = 30):
        return pair.store.stats(days)

    return app
