"""Proxy HTTP de PAIR: tres puertas de entrada (OpenAI, Anthropic, Gemini), un router."""

from __future__ import annotations

import copy
import base64
import json
import re
import uuid
import time
import unicodedata
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import translate as tr
from .backends import BackendError, Client
from .config import Config
from .images import ImageError, to_openai as images_to_openai, ImageService
from .router import CLOUD, LOCAL, Decision, decide, estimate_tokens, last_user_text, strip_tags
from .store import Store


class Pair:
    """Decide, llama al backend (con fallback) y registra."""

    def __init__(self, cfg: Config, local_transport: httpx.AsyncBaseTransport | None = None,
                 cloud_transport: httpx.AsyncBaseTransport | None = None,
                 image_transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self.images = ImageService(cfg.images, cfg.cloud, cfg.rules, cfg.mode, image_transport, cloud_transport)
        self.clients = {
            LOCAL: Client(cfg.local, local_transport),
            CLOUD: Client(cfg.cloud, cloud_transport, is_cloud=True),
        }
        self.store = Store(cfg.log_path, cfg.cloud.price_input_per_mtok, cfg.cloud.price_output_per_mtok,
                           cfg.capture_prompts)

    async def close(self) -> None:
        for c in self.clients.values():
            await c.close()
        await self.images.close()

    async def route(self, req: dict[str, Any]) -> tuple[Decision, list[str], str]:
        prompt = last_user_text(req)          # antes de quitar etiquetas
        context = await self.clients[LOCAL].context_tokens()
        decision = decide(req, self.cfg.rules, self.cfg.mode, context)
        other = CLOUD if decision.target == LOCAL else LOCAL
        order = [decision.target] if decision.sensitive else [decision.target, other]
        return decision, order, prompt

    async def call(self, endpoint: str, req: dict[str, Any], stream: bool):
        """Devuelve (decision, target, model, resultado, fell_back, prompt, t0).

        `resultado` es un dict (sin stream) o un iterador de chunks (con stream).
        """
        t0 = time.monotonic()
        decision, order, prompt = await self.route(req)
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
            prompt_tokens: int | None, completion_tokens: int | None, completion_chars: int = 0,
            reasoning_tokens: int = 0) -> None:
        self.store.record(
            endpoint=endpoint, target=target, decided=decision.target, reason=decision.reason, model=model,
            sensitive=decision.sensitive, fell_back=fell_back, stream=stream,
            prompt_tokens=prompt_tokens if prompt_tokens is not None else estimate_tokens(req),
            completion_tokens=completion_tokens if completion_tokens is not None else completion_chars // 4,
            latency_ms=int((time.monotonic() - t0) * 1000), prompt=prompt, reasoning_tokens=reasoning_tokens,
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


_EMBEDDED_IMAGE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]+\)")


def _drop_embedded_images(req: dict[str, Any]) -> None:
    """Quita de los mensajes las imágenes incrustadas como texto en base64.

    Un cliente puede reenviarlas en el historial o en peticiones auxiliares (como la
    del título de la conversación); son megas que ningún modelo necesita leer y que
    dispararían el tamaño y el coste de la petición.
    """
    for m in req.get("messages", []):
        content = m.get("content")
        if isinstance(content, str) and "data:image/" in content:
            m["content"] = _EMBEDDED_IMAGE.sub("[imagen generada]", content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "data:image/" in str(part.get("text", "")):
                    part["text"] = _EMBEDDED_IMAGE.sub("[imagen generada]", part["text"])


IMAGES_DIR = Path("~/.pair/images").expanduser()
_GENERATED_LINK = re.compile(r"!\[[^\]]*\]\(https?://[^)\s]+/imagenes/([\w.-]+\.png)\)")


def _attach_last_generated_image(req: dict[str, Any]) -> None:
    """Si en la conversación hay imágenes que generó PAIR, adjunta la más reciente al último
    mensaje del usuario, para que el modelo la vea de verdad (en el historial solo hay un enlace).

    Así "descríbela" o "¿qué le cambiarías?" se responden mirando la imagen y no imaginándola.
    """
    messages = req.get("messages", [])
    if not messages or messages[-1].get("role") != "user":
        return
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in content):
        return                                       # ya trae sus propias imágenes
    for m in reversed(messages[:-1]):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), str):
            continue
        names = _GENERATED_LINK.findall(m["content"])
        path = IMAGES_DIR / names[-1] if names else None
        if path and path.is_file():
            b64 = base64.b64encode(path.read_bytes()).decode()
            parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            last["content"] = parts + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
            return


def _save_image(b64: str) -> str:
    """Guarda una imagen generada y devuelve su nombre de fichero."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.png"
    (IMAGES_DIR / name).write_bytes(base64.b64decode(b64))
    return name


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
                     len(((result.get("choices") or [{}])[0].get("message") or {}).get("content") or ""),
                     (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            return JSONResponse(to_response(result, model), headers=headers)

        stats = tr.StreamStats()

        async def body() -> AsyncIterator[bytes]:
            try:
                async for piece in to_stream(result, stats):
                    yield piece
            finally:
                pair.log(endpoint, canonical, decision, target, model, fell_back, True, prompt, t0,
                         stats.prompt_tokens, stats.completion_tokens, stats.chars, stats.reasoning_tokens)

        return StreamingResponse(body(), media_type="text/event-stream", headers=headers)

    # ------------------------------------------------------------ OpenAI
    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        req = await request.json()
        inspect("openai", req)
        _drop_embedded_images(req)
        # Imagen pedida desde el chat: modelo de imagen (p. ej. "pair-image") o etiqueta @imagen.
        image_tags = (cfg.rules.get("tags", {}) or {}).get("image", ["@imagen", "@image"])
        if "image" in str(req.get("model", "")).lower() or strip_tags(req, image_tags):
            return await image_in_chat(req, bool(req.get("stream")), str(request.base_url))
        _attach_last_generated_image(req)
        return await handle("openai", req, bool(req.get("stream")),
                            lambda resp, model: resp,
                            lambda chunks, stats: tr.openai_stream_passthrough(chunks, stats))

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "pair", "object": "model", "owned_by": "pair"},
                                           {"id": "pair-image", "object": "model", "owned_by": "pair"}]}

    # ------------------------------------------------------------ Imágenes (formato OpenAI)
    async def make_images(req: dict[str, Any]):
        """Genera y registra. Devuelve (resultado, None) o (None, error)."""
        t0 = time.monotonic()
        try:
            result = await pair.images.generate(req)
        except ImageError as e:
            pair.store.record(endpoint="images", target="local", decided="local", reason="error", model=None,
                              sensitive=False, fell_back=False, stream=False, prompt_tokens=0, completion_tokens=0,
                              latency_ms=int((time.monotonic() - t0) * 1000), error=e.detail, cloud_cost=0.0)
            return None, e
        pair.store.record(endpoint="images", target=result.target, decided=result.decision.target,
                          reason=result.decision.reason, model=result.model, sensitive=result.decision.sensitive,
                          fell_back=result.fell_back, stream=False, prompt_tokens=0, completion_tokens=0,
                          latency_ms=int((time.monotonic() - t0) * 1000), cloud_cost=result.cloud_cost)
        return result, None

    @app.post("/v1/images/generations")
    async def images(request: Request):
        req = await request.json()
        inspect("images", req)
        result, error = await make_images(req)
        if error:
            status = error.status if error.status and 400 <= error.status < 500 else 502
            return JSONResponse({"error": {"message": f"PAIR no pudo generar la imagen: {error.detail}",
                                           "type": "pair_backend_error"}}, status_code=status)
        return JSONResponse(images_to_openai(result, req.get("response_format")),
                            headers=_route_header(result.decision, result.target, result.model, result.fell_back))

    @app.get("/imagenes/{name}")
    async def image_file(name: str):
        path = IMAGES_DIR / name
        if not re.fullmatch(r"[\w.-]+\.png", name) or not path.is_file():
            return JSONResponse({"error": "imagen no encontrada"}, status_code=404)
        return FileResponse(path, media_type="image/png")

    async def image_in_chat(req: dict[str, Any], stream: bool, base_url: str):
        """Genera una imagen pedida desde un chat. La guarda en ~/.pair/images y responde con
        un enlace a ella, de modo que en la conversación solo viaja una URL corta."""
        result, error = await make_images({"prompt": last_user_text(req)})
        if error:
            content, headers = f"No he podido generar la imagen: {error.detail}", {}
        else:
            links = [f"{base_url.rstrip('/')}/imagenes/{_save_image(b64)}" for b64 in result.images]
            content = "\n\n".join(f"![imagen]({url})" for url in links)
            headers = _route_header(result.decision, result.target, result.model, result.fell_back)
        model = req.get("model", "pair")
        if not stream:
            return JSONResponse({"id": "pair-img", "object": "chat.completion", "created": int(time.time()),
                                 "model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}],
                                 "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                                headers=headers)

        async def body() -> AsyncIterator[bytes]:
            base = {"id": "pair-img", "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
            for delta, finish in (({"role": "assistant", "content": content}, None), ({}, "stop")):
                chunk = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(body(), media_type="text/event-stream", headers=headers)

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
