"""Clientes para los backends, ambos con API compatible con OpenAI.

- local: LM Studio (http://localhost:1234/v1)
- cloud: OpenAI (o cualquier endpoint compatible)
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import Backend

# Parámetros que algunos modelos de razonamiento rechazan; si la API se queja
# de alguno, se reintenta una vez sin él.
_DROPPABLE = ("temperature", "top_p", "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs")


class BackendError(Exception):
    def __init__(self, backend: str, status: int | None, detail: str):
        super().__init__(f"{backend}: {status} {detail}")
        self.backend = backend
        self.status = status
        self.detail = detail

    @property
    def retriable(self) -> bool:
        """¿Tiene sentido probar con otro backend? (caído, timeout o error 5xx)."""
        return self.status is None or self.status >= 500


class Client:
    def __init__(self, backend: Backend, transport: httpx.AsyncBaseTransport | None = None, is_cloud: bool = False):
        self.backend = backend
        self.is_cloud = is_cloud
        headers = {"Content-Type": "application/json"}
        if backend.api_key:
            headers["Authorization"] = f"Bearer {backend.api_key}"
        self._http = httpx.AsyncClient(
            base_url=backend.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(backend.timeout_s, connect=5.0),
            transport=transport,
        )
        root = backend.base_url.rstrip("/")
        self._lms_url = (root[:-3] if root.endswith("/v1") else root) + "/api/v0/models"
        self._lms_cache: tuple[float, list[dict[str, Any]] | None] = (0.0, None)

    async def close(self) -> None:
        await self._http.aclose()

    async def _lmstudio(self) -> list[dict[str, Any]] | None:
        """Estado de los modelos según la API propia de LM Studio (None si no es LM Studio).

        Se consulta como mucho cada 30 s, para notar cambios hechos en LM Studio sin
        añadir latencia a cada petición.
        """
        if self.is_cloud:
            return None
        ts, data = self._lms_cache
        if time.monotonic() - ts < 30:
            return data
        try:
            r = await self._http.get(self._lms_url, timeout=3.0)
            data = r.json().get("data") if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            data = None
        self._lms_cache = (time.monotonic(), data)
        return data

    async def model(self) -> str:
        """Modelo a usar: el configurado; si no hay, el que esté cargado en LM Studio,
        y si no se sabe, el primero que ofrezca el servidor."""
        if self.backend.model:
            return self.backend.model
        for m in await self._lmstudio() or []:
            if m.get("state") == "loaded" and m.get("type") != "embeddings":
                return m["id"]
        try:
            r = await self._http.get("/models")
            r.raise_for_status()
            models = [m["id"] for m in r.json().get("data", []) if "embed" not in m["id"]]
        except httpx.HTTPError as e:
            raise BackendError(self.backend.name, None, f"no responde: {e}") from e
        if not models:
            raise BackendError(self.backend.name, None, "no hay ningún modelo cargado")
        return models[0]

    async def context_tokens(self) -> int:
        """Contexto útil del modelo local: el que tiene cargado LM Studio (con un 10 % de
        margen para la respuesta) o, si no se puede saber, `max_context_tokens`."""
        try:
            model = await self.model()
        except BackendError:
            return self.backend.max_context_tokens
        for m in await self._lmstudio() or []:
            if m.get("id") == model and m.get("loaded_context_length"):
                return int(m["loaded_context_length"] * 0.9)
        return self.backend.max_context_tokens

    def _prepare(self, req: dict[str, Any], model: str, stream: bool) -> dict[str, Any]:
        body = {k: v for k, v in req.items() if not k.startswith("_")}
        body["model"] = model
        body["stream"] = stream
        if self.is_cloud and "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
        if stream and self.is_cloud:
            body["stream_options"] = {"include_usage": True}
        else:
            body.pop("stream_options", None)
        return body

    async def _post(self, body: dict[str, Any], stream: bool) -> httpx.Response:
        try:
            request = self._http.build_request("POST", "/chat/completions", json=body)
            r = await self._http.send(request, stream=stream)
        except httpx.HTTPError as e:
            raise BackendError(self.backend.name, None, f"no responde: {e}") from e
        if r.status_code >= 400:
            text = (await r.aread()).decode("utf-8", "replace")
            await r.aclose()
            dropped = [p for p in _DROPPABLE if p in body and p in text]
            if r.status_code == 400 and dropped:
                for p in dropped:
                    body.pop(p)
                return await self._post(body, stream)
            raise BackendError(self.backend.name, r.status_code, text[:500])
        return r

    async def complete(self, req: dict[str, Any]) -> dict[str, Any]:
        body = self._prepare(req, await self.model(), stream=False)
        r = await self._post(body, stream=False)
        return r.json()

    async def stream(self, req: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Abre el stream y devuelve un iterador de chunks ya parseados.

        La conexión se establece aquí (antes de devolver), de modo que un
        backend caído se detecta a tiempo de hacer fallback.
        """
        body = self._prepare(req, await self.model(), stream=True)
        r = await self._post(body, stream=True)

        async def chunks() -> AsyncIterator[dict[str, Any]]:
            try:
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue
            finally:
                await r.aclose()

        return chunks()
