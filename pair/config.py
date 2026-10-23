"""Carga de config.yaml y rules.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Backend:
    name: str
    base_url: str
    model: str = ""
    api_key_env: str = ""
    timeout_s: float = 120
    max_context_tokens: int = 0
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8787
    mode: str = "equilibrado"
    local: Backend = field(default_factory=lambda: Backend("lmstudio", "http://localhost:1234/v1"))
    cloud: Backend = field(default_factory=lambda: Backend("openai", "https://api.openai.com/v1", "gpt-5", "OPENAI_API_KEY"))
    log_path: str = "~/.pair/pair.db"
    capture_prompts: bool = True
    rules: dict[str, Any] = field(default_factory=dict)


MODES = ("ahorro", "equilibrado", "calidad")


def _backend(data: dict[str, Any], default: Backend) -> Backend:
    merged = {**default.__dict__, **(data or {})}
    return Backend(**{k: v for k, v in merged.items() if k in Backend.__dataclass_fields__})


def load(config_path: str | Path = "config.yaml") -> Config:
    path = Path(config_path)
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = Config()
    server = data.get("server", {})
    log = data.get("log", {})
    mode = data.get("mode", base.mode)
    if mode not in MODES:
        raise ValueError(f"mode debe ser uno de {MODES}, no '{mode}'")

    rules_path = Path(data.get("rules_path", "rules.yaml"))
    if not rules_path.is_absolute():
        rules_path = path.parent / rules_path
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8")) if rules_path.exists() else {}

    return Config(
        host=server.get("host", base.host),
        port=int(server.get("port", base.port)),
        mode=mode,
        local=_backend(data.get("local", {}), base.local),
        cloud=_backend(data.get("cloud", {}), base.cloud),
        log_path=log.get("path", base.log_path),
        capture_prompts=bool(log.get("capture_prompts", base.capture_prompts)),
        rules=rules or {},
    )
