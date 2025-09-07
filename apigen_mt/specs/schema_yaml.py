from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple
import yaml
import logging

log = logging.getLogger("apigen_mt.schemas")

SCHEMAS_DIR = os.getenv("APIGEN_SCHEMAS_DIR", "configs/schemas")


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _write_yaml(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    log.info("Wrote YAML: %s", path)


def load_tools(domain: str) -> Dict[str, Any]:
    path = os.path.join(SCHEMAS_DIR, f"{domain.lower()}.yaml")
    data = _load_yaml(path)
    if "tools" not in data:
        data["tools"] = []
    return data


def write_tools(domain: str, tools_data: Dict[str, Any]):
    path = os.path.join(SCHEMAS_DIR, f"{domain.lower()}.yaml")
    _write_yaml(path, tools_data)


def merge_tools(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {t.get("name"): t for t in existing if t.get("name")}
    for t in new:
        name = t.get("name")
        if not name:
            continue
        if name in by_name:
            # Optionally merge fields; keep existing as source of truth
            # but update description/deps/schema if missing
            cur = by_name[name]
            for k in ("description", "write", "deps", "schema"):
                if k not in cur or cur[k] in (None, [], {}):
                    cur[k] = t.get(k, cur.get(k))
        else:
            by_name[name] = t
    log.info("Merged tools: existing=%d new=%d result=%d", len(existing), len(new), len(by_name))
    return list(by_name.values())


def split_domain_yaml(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split combined domain YAML into sampling vs tools docs."""
    sampling = {
        k: v
        for k, v in data.items()
        if k in ("domain", "personas", "policies", "domain_rows", "examples")
    }
    tools = {"domain": data.get("domain"), "tools": data.get("tools", [])}
    return sampling, tools
