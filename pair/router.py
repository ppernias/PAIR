"""Motor de enrutado: decide si una petición va al modelo local o a la nube.

Trabaja siempre sobre el formato canónico (chat completions de OpenAI).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

LOCAL = "local"
CLOUD = "cloud"


@dataclass
class Decision:
    target: str          # "local" | "cloud"
    reason: str          # regla que produjo la decisión, legible
    sensitive: bool = False

    @property
    def header(self) -> str:
        return f"{self.target}; {self.reason}"


# ---------------------------------------------------------------- utilidades

def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def last_user_text(req: dict[str, Any]) -> str:
    for msg in reversed(req.get("messages", [])):
        if msg.get("role") == "user":
            text = _text_of(msg.get("content"))
            if text:
                return text
    return ""


def all_text(req: dict[str, Any]) -> str:
    parts = [_text_of(m.get("content")) for m in req.get("messages", [])]
    if req.get("tools"):
        parts.append(json.dumps(req["tools"], ensure_ascii=False))
    return "\n".join(parts)


def has_images(req: dict[str, Any]) -> bool:
    for msg in req.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        ):
            return True
    return False


def estimate_tokens(req: dict[str, Any]) -> int:
    """Aproximación barata: ~4 caracteres por token."""
    return len(all_text(req)) // 4


def strip_tags(req: dict[str, Any], tags: list[str]) -> str | None:
    """Quita la primera etiqueta encontrada del último mensaje del usuario.

    Devuelve la etiqueta quitada, o None. Modifica `req` en el sitio.
    """
    for msg in reversed(req.get("messages", [])):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        for tag in tags:
            pattern = re.compile(rf"(^|\s){re.escape(tag)}(?=\s|$)", re.IGNORECASE)
            if isinstance(content, str) and pattern.search(content):
                msg["content"] = pattern.sub(" ", content).strip()
                return tag
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and pattern.search(part.get("text", "")):
                        part["text"] = pattern.sub(" ", part["text"]).strip()
                        return tag
        return None
    return None


def _matches_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE) is not None


# ---------------------------------------------------------------- decisión

def is_sensitive(req: dict[str, Any], rules: dict[str, Any]) -> str | None:
    text = all_text(req)
    for pattern in rules.get("sensitive_patterns", []) or []:
        if re.search(pattern, text):
            return pattern
    return None


def decide(req: dict[str, Any], rules: dict[str, Any], mode: str, local_max_context: int = 0) -> Decision:
    """Aplica la cascada de reglas. Puede modificar `req` (quita etiquetas)."""
    tags = rules.get("tags", {}) or {}

    # 1. Privacidad: lo sensible nunca sale del equipo (ni siquiera con @frontera).
    sensitive = is_sensitive(req, rules)

    # 2. Etiquetas explícitas.
    if strip_tags(req, tags.get("local", [])):
        return Decision(LOCAL, "etiqueta @local", bool(sensitive))
    cloud_tag = strip_tags(req, tags.get("cloud", []))
    if sensitive:
        return Decision(LOCAL, "contenido sensible", True)
    if cloud_tag:
        return Decision(CLOUD, f"etiqueta {cloud_tag}")

    # 3. Capacidades que el local no cubre.
    force = rules.get("force_cloud_if", {}) or {}
    if force.get("images") and has_images(req):
        return Decision(CLOUD, "incluye imágenes")
    tokens = estimate_tokens(req)
    if force.get("over_local_context") and local_max_context and tokens > local_max_context:
        return Decision(CLOUD, f"contexto de ~{tokens} tokens supera el local")

    # 4. Palabras clave.
    text = last_user_text(req)
    keywords = rules.get("keywords", {}) or {}
    for phrase in keywords.get("cloud", []) or []:
        if _matches_phrase(text, phrase):
            return Decision(CLOUD, f"palabra clave '{phrase}'")
    for phrase in keywords.get("local", []) or []:
        if _matches_phrase(text, phrase):
            return Decision(LOCAL, f"palabra clave '{phrase}'")

    # 5. Modo por defecto.
    default = (rules.get("default_by_mode", {}) or {}).get(mode)
    if default == LOCAL:
        return Decision(LOCAL, f"modo {mode}")
    if default == CLOUD:
        return Decision(CLOUD, f"modo {mode}")
    if isinstance(default, dict) and "local_if_prompt_tokens_below" in default:
        limit = int(default["local_if_prompt_tokens_below"])
        if tokens < limit:
            return Decision(LOCAL, f"modo {mode}: ~{tokens} tokens < {limit}")
        return Decision(CLOUD, f"modo {mode}: ~{tokens} tokens >= {limit}")
    return Decision(CLOUD, "sin regla aplicable")
