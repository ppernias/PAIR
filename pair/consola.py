"""Formato de las líneas de registro en el Terminal: verde = local, rojo = frontera."""

from __future__ import annotations

import os
import sys

_VERDE, _ROJO, _AMARILLO, _FIN = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def _con_color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def linea(when: str, endpoint: str, target: str, reason: str, model: str | None,
          prompt_tokens: int | None, completion_tokens: int | None, latency_ms: int | None,
          fell_back: bool, error: str | None) -> str:
    flag = " ↺" if fell_back else ""
    tail = f"  ERROR {error[:60]}" if error else f"  {prompt_tokens}+{completion_tokens} tok  {latency_ms} ms"
    texto = f"{when}  {endpoint:<9} {target:<5}{flag}  {reason}  [{model}]{tail}"
    if not _con_color():
        return texto
    color = _AMARILLO if error else (_VERDE if target == "local" else _ROJO)
    return f"{color}{texto}{_FIN}"
