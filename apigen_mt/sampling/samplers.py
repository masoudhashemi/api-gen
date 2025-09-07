from __future__ import annotations
import random
from typing import Any, Dict, List

from apigen_mt.sampling.yaml_loader import sampler_bundle


def sample_persona(rng: random.Random, domain: str) -> Dict[str, Any]:
    personas = sampler_bundle(domain).get("personas", [])
    if not personas:
        return {"id": "p_default", "style": "neutral", "tone": "helpful"}
    return rng.choice(personas)


def sample_policies(domain: str) -> List[str]:
    return sampler_bundle(domain).get("policies", [])


def sample_domain_rows(domain: str, rng: random.Random) -> Dict[str, Any]:
    return sampler_bundle(domain).get("domain_rows", {})


def sample_examples(domain: str) -> List[str]:
    return sampler_bundle(domain).get("examples", [])


def build_task_context(api_graph_summary: str, domain: str, rng: random.Random, tool_specs: Dict[str, Any] | None = None, skeleton: List[str] | None = None, tool_catalog: List[Dict[str, Any]] | None = None) -> str:
    persona = sample_persona(rng, domain)
    policies = sample_policies(domain)
    rows = sample_domain_rows(domain, rng)
    examples = sample_examples(domain)
    # Derive a compact map of allowed IDs from rows
    valid_ids: Dict[str, List[str]] = {}
    try:
        for tbl, recs in (rows or {}).items():
            for r in recs or []:
                if isinstance(r, dict):
                    for k, v in r.items():
                        if isinstance(k, str) and k.endswith("_id") and isinstance(v, str):
                            valid_ids.setdefault(k, [])
                            if v not in valid_ids[k]:
                                valid_ids[k].append(v)
    except Exception:
        valid_ids = {}
    ctx = (
        f"Persona: {persona}\nDomain: {domain}\nPolicies: {policies}\nData: {rows}\n"
        f"APIs: {api_graph_summary}\nToolSchemas: {tool_specs or {}}\n"
        f"ToolCatalog: {[{k: t.get(k) for k in ('name','description','write','deps')} for t in (tool_catalog or [])]}\n"
        f"ValidIDs: {valid_ids}\n"
        f"SkeletonHint: {skeleton or []}\n"
        f"Guidance: {examples}\n"
        "Generate a realistic multi-step task using available tools."
    )
    return ctx
