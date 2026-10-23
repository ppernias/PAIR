"""Generación de imágenes: el generador local (FLUX en el Mac Studio) o OpenAI.

Expone el formato de OpenAI (`POST /v1/images/generations`) y aplica la misma idea
que con el texto: local por defecto, nube cuando se pide o cuando el local falla.
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Backend, Images
from .router import CLOUD, LOCAL, Decision, is_sensitive, strip_tags


@dataclass
class ImageResult:
    decision: Decision
    target: str
    model: str
    images: list[str]          # PNG en base64
    fell_back: bool
    cloud_cost: float


class ImageError(Exception):
    def __init__(self, status: int | None, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail

    @property
    def retriable(self) -> bool:
        return self.status is None or self.status >= 500


def _size(size: str | None) -> tuple[int, int] | None:
    """'1536x1024' -> (1536, 1024), ajustado a lo que admite FLUX (múltiplos de 16, máx. 2048)."""
    m = re.fullmatch(r"(\d+)x(\d+)", size or "")
    if not m:
        return None
    w, h = (min(2048, max(256, int(v) // 16 * 16)) for v in m.groups())
    return w, h


class ImageService:
    def __init__(self, images: Images, cloud: Backend, rules: dict[str, Any], mode: str,
                 local_transport: httpx.AsyncBaseTransport | None = None,
                 cloud_transport: httpx.AsyncBaseTransport | None = None):
        self.cfg, self.cloud, self.rules, self.mode = images, cloud, rules, mode
        timeout = httpx.Timeout(images.timeout_s, connect=5.0)
        self._local = httpx.AsyncClient(base_url=images.local_url.rstrip("/"), timeout=timeout,
                                        transport=local_transport) if images.local_url else None
        headers = {"Authorization": f"Bearer {cloud.api_key}"} if cloud.api_key else {}
        self._cloud = httpx.AsyncClient(base_url=cloud.base_url.rstrip("/"), headers=headers, timeout=timeout,
                                        transport=cloud_transport)
        self._lock = asyncio.Lock()      # el generador local atiende de una en una

    async def close(self) -> None:
        if self._local:
            await self._local.aclose()
        await self._cloud.aclose()

    def decide(self, req: dict[str, Any]) -> Decision:
        """Quita etiquetas del prompt y decide. Modifica `req["prompt"]`."""
        wrapper = {"messages": [{"role": "user", "content": req.get("prompt", "")}]}
        tags = self.rules.get("tags", {}) or {}
        sensitive = is_sensitive(wrapper, self.rules) is not None
        local_tag = strip_tags(wrapper, tags.get("local", []))
        cloud_tag = strip_tags(wrapper, tags.get("cloud", [])) if not local_tag else None
        req["prompt"] = wrapper["messages"][0]["content"]
        if not self._local:
            if sensitive:
                return Decision(LOCAL, "contenido sensible y sin generador local", True)
            return Decision(CLOUD, "sin generador local")
        if local_tag:
            return Decision(LOCAL, "etiqueta @local", sensitive)
        if sensitive:
            return Decision(LOCAL, "contenido sensible", True)
        if cloud_tag:
            return Decision(CLOUD, f"etiqueta {cloud_tag}")
        if self.mode == "calidad":
            return Decision(CLOUD, "modo calidad")
        return Decision(LOCAL, "imagen: local por defecto")

    async def _generate_local(self, req: dict[str, Any]) -> list[str]:
        if not self._local:
            raise ImageError(None, "no hay generador local configurado")
        body: dict[str, Any] = {"prompt": req["prompt"], "referencias": [], "mejorar": True}
        if size := _size(req.get("size")):
            body["ancho"], body["alto"] = size
        out = []
        for _ in range(max(1, int(req.get("n") or 1))):
            async with self._lock:
                try:
                    r = await self._local.post("/generar", json=body)
                except httpx.HTTPError as e:
                    raise ImageError(None, f"generador local no responde: {e}") from e
            if r.status_code >= 400:
                raise ImageError(r.status_code, r.text[:300])
            out.append(base64.b64encode(r.content).decode())
        return out

    async def _generate_cloud(self, req: dict[str, Any]) -> list[str]:
        body = {k: v for k, v in req.items() if k in ("prompt", "n", "size", "quality", "background", "output_format")}
        body["model"] = self.cfg.cloud_model
        try:
            r = await self._cloud.post("/images/generations", json=body)
        except httpx.HTTPError as e:
            raise ImageError(None, f"nube no responde: {e}") from e
        if r.status_code >= 400:
            raise ImageError(r.status_code, r.text[:300])
        images = []
        for item in r.json().get("data", []):
            if item.get("b64_json"):
                images.append(item["b64_json"])
            elif item.get("url"):
                img = await self._cloud.get(item["url"])
                images.append(base64.b64encode(img.content).decode())
        return images

    async def generate(self, req: dict[str, Any]) -> ImageResult:
        decision = self.decide(req)
        order = [decision.target]
        if not decision.sensitive and self._local:
            order.append(CLOUD if decision.target == LOCAL else LOCAL)
        error: ImageError | None = None
        for i, target in enumerate(order):
            try:
                if target == LOCAL:
                    images = await self._generate_local(req)
                    model = "flux-local"
                else:
                    images = await self._generate_cloud(req)
                    model = self.cfg.cloud_model
                cost = self.cfg.cloud_price_per_image * len(images)
                return ImageResult(decision, target, model, images, i > 0, cost)
            except ImageError as e:
                error = e
                if not e.retriable:
                    break
        assert error is not None
        raise error


def to_openai(result: ImageResult, response_format: str | None) -> dict[str, Any]:
    data = []
    for b64 in result.images:
        item = {"b64_json": b64}
        if response_format == "url":
            item = {"url": f"data:image/png;base64,{b64}"}
        data.append(item)
    return {"created": int(time.time()), "data": data}
