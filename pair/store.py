"""Registro local en SQLite: decisiones, costes y banco de peticiones."""

from __future__ import annotations

import csv
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    endpoint TEXT NOT NULL,          -- openai | anthropic | gemini
    target TEXT NOT NULL,            -- local | cloud (donde se resolvió)
    decided TEXT NOT NULL,           -- local | cloud (lo que decidió el router)
    reason TEXT NOT NULL,
    model TEXT,
    sensitive INTEGER NOT NULL DEFAULT 0,
    fell_back INTEGER NOT NULL DEFAULT 0,
    stream INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,                   -- coste real
    cloud_cost_usd REAL,             -- coste si hubiera ido a la nube
    latency_ms INTEGER,
    error TEXT
);
CREATE TABLE IF NOT EXISTS bench (
    request_id INTEGER PRIMARY KEY REFERENCES requests(id),
    prompt TEXT NOT NULL,
    label TEXT                       -- a rellenar a mano: local | cloud
);
"""


class Store:
    def __init__(self, path: str, price_in: float, price_out: float, capture_prompts: bool = True):
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(p, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.price_in = price_in
        self.price_out = price_out
        self.capture_prompts = capture_prompts

    def cloud_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price_in + completion_tokens * self.price_out) / 1_000_000

    def record(
        self,
        *,
        endpoint: str,
        target: str,
        decided: str,
        reason: str,
        model: str | None,
        sensitive: bool,
        fell_back: bool,
        stream: bool,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        prompt: str = "",
        error: str | None = None,
    ) -> int:
        cloud_cost = self.cloud_cost(prompt_tokens, completion_tokens)
        cost = cloud_cost if target == "cloud" else 0.0
        cur = self.db.execute(
            "INSERT INTO requests (ts, endpoint, target, decided, reason, model, sensitive, fell_back, stream,"
            " prompt_tokens, completion_tokens, cost_usd, cloud_cost_usd, latency_ms, error)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), endpoint, target, decided, reason, model, int(sensitive), int(fell_back), int(stream),
             prompt_tokens, completion_tokens, cost, cloud_cost, latency_ms, error),
        )
        rid = cur.lastrowid
        if self.capture_prompts and prompt and not sensitive and not error:
            self.db.execute("INSERT INTO bench (request_id, prompt) VALUES (?, ?)", (rid, prompt))
        self.db.commit()
        return rid

    def stats(self, days: float = 30) -> dict[str, Any]:
        since = time.time() - days * 86400
        row = self.db.execute(
            "SELECT COUNT(*), SUM(target='local'), SUM(fell_back), SUM(sensitive),"
            " COALESCE(SUM(cost_usd),0), COALESCE(SUM(cloud_cost_usd),0), SUM(error IS NOT NULL)"
            " FROM requests WHERE ts >= ?",
            (since,),
        ).fetchone()
        total, local, fell_back, sensitive, cost, all_cloud, errors = (v or 0 for v in row)
        return {
            "days": days,
            "requests": total,
            "local": local,
            "local_pct": round(100 * local / total, 1) if total else 0.0,
            "fell_back": fell_back,
            "sensitive": sensitive,
            "errors": errors,
            "cost_usd": round(cost, 4),
            "all_cloud_usd": round(all_cloud, 4),
            "saved_usd": round(all_cloud - cost, 4),
            "saved_pct": round(100 * (all_cloud - cost) / all_cloud, 1) if all_cloud else 0.0,
        }

    def recent(self, n: int = 20) -> list[tuple]:
        return self.db.execute(
            "SELECT datetime(ts,'unixepoch','localtime'), endpoint, target, reason, model,"
            " prompt_tokens, completion_tokens, latency_ms, fell_back, error"
            " FROM requests ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()

    def export_bench(self, out: str) -> int:
        rows = self.db.execute(
            "SELECT b.request_id, datetime(r.ts,'unixepoch','localtime'), r.target, r.reason, b.prompt, b.label"
            " FROM bench b JOIN requests r ON r.id = b.request_id ORDER BY b.request_id"
        ).fetchall()
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "fecha", "decision_pair", "motivo", "prompt", "etiqueta (local|cloud)"])
            w.writerows(rows)
        return len(rows)
