from __future__ import annotations
import os
from typing import Any, Dict
import yaml
import logging

log = logging.getLogger("apigen_mt.sampling")


DEFAULT_DIR = os.getenv("APIGEN_SAMPLERS_DIR", "configs/sampling")


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    log.debug("Loaded YAML: %s", path)
    return data


def load_global() -> Dict[str, Any]:
    return _load_yaml(os.path.join(DEFAULT_DIR, "global.yaml"))


def load_domain(domain: str) -> Dict[str, Any]:
    return _load_yaml(os.path.join(DEFAULT_DIR, f"{domain.lower()}.yaml"))


def sampler_bundle(domain: str) -> Dict[str, Any]:
    g = load_global()
    d = load_domain(domain)
    # Merge with domain taking precedence
    personas = d.get("personas", g.get("personas", []))
    policies = d.get("policies", [])
    domain_rows = d.get("domain_rows", {})
    examples = d.get("examples", [])
    log.info("Sampler bundle | domain=%s personas=%d policies=%d examples=%d", domain, len(personas or []), len(policies or []), len(examples or []))
    return {
        "personas": personas,
        "policies": policies,
        "domain_rows": domain_rows,
        "examples": examples,
    }
