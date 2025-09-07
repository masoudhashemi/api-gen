from __future__ import annotations
from typing import Any, Dict, Iterable, List
import os
import orjson
import hashlib


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "ab") as f:
        for rec in records:
            f.write(orjson.dumps(rec))
            f.write(b"\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    res: List[Dict[str, Any]] = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            res.append(orjson.loads(line))
    return res


def hash_actions(actions: List[Dict[str, Any]]) -> str:
    norm = [
        {"tool_name": a["tool_name"], "arguments": a.get("arguments", {})}
        for a in actions
    ]
    s = orjson.dumps(norm)
    return hashlib.sha256(s).hexdigest()

