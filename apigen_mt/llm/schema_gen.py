from __future__ import annotations
from typing import Any, Dict
import yaml

from apigen_mt.llm.lite import LLMClient
from apigen_mt.sampling.yaml_loader import load_domain as load_sampling_domain
from apigen_mt.specs.schema_yaml import load_tools as load_tools_yaml, split_domain_yaml


DOMAIN_SPEC_PROMPT = """
You are a domain schema generator. Produce a YAML spec for a new domain.
The YAML should contain keys: domain (str), personas (list of {id, style, tone}),
policies (list of strings), domain_rows (object with small arrays of rows),
examples (list of strings), and tools (list of tools).
Each tool: { name, description, write (bool), deps (list of tool names), schema (JSON Schema for arguments) }.
Keep arrays small (2-3 items). Ensure tool schemas are strict and deps realistic.
If existing items are provided, DO NOT duplicate them. Prefer adding complementary items.
"""


def generate_domain_yaml(domain_name: str, *, hints: Dict[str, Any] | None = None) -> str:
    hints = hints or {}
    llm = LLMClient()
    # Include existing YAML to avoid duplicates
    existing_sampling = load_sampling_domain(domain_name)
    existing_tools = load_tools_yaml(domain_name)
    user = (
        f"Domain: {domain_name}\nHints: {hints}\n"
        f"ExistingSampling:\n{yaml.safe_dump(existing_sampling, sort_keys=False)}\n"
        f"ExistingTools:\n{yaml.safe_dump(existing_tools, sort_keys=False)}\n"
        "Output: YAML only."
    )
    content = llm.chat([
        {"role": "system", "content": DOMAIN_SPEC_PROMPT},
        {"role": "user", "content": user},
    ])
    if llm.mock:
        # Deterministic stub for offline dev
        if domain_name.lower() == "support":
            spec = {
                "domain": "support",
                "personas": [
                    {"id": "p_direct", "style": "concise", "tone": "neutral"},
                    {"id": "p_patient", "style": "detailed", "tone": "polite"},
                ],
                "policies": [
                    "Only the ticket owner can change status.",
                    "Do not reveal private customer data.",
                ],
                "domain_rows": {
                    "tickets": [
                        {"ticket_id": "t_100", "status": "open"},
                        {"ticket_id": "t_200", "status": "pending"},
                    ],
                    "customers": [
                        {"customer_id": "c_001", "name": "Lena"},
                        {"customer_id": "c_002", "name": "Omar"},
                    ],
                },
                "examples": [
                    "Update ticket t_100 to pending and add a comment.",
                ],
                "tools": [
                    {
                        "name": "get_ticket",
                        "description": "Fetch a ticket by ID.",
                        "write": False,
                        "deps": [],
                        "schema": {"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"]},
                    },
                    {
                        "name": "update_ticket_status",
                        "description": "Update a ticket's status.",
                        "write": True,
                        "deps": ["get_ticket"],
                        "schema": {"type": "object", "properties": {"ticket_id": {"type": "string"}, "status": {"type": "string"}}, "required": ["ticket_id", "status"]},
                    },
                    {
                        "name": "add_comment",
                        "description": "Add a comment to a ticket.",
                        "write": True,
                        "deps": ["get_ticket"],
                        "schema": {"type": "object", "properties": {"ticket_id": {"type": "string"}, "comment": {"type": "string"}}, "required": ["ticket_id", "comment"]},
                    },
                ],
            }
            # Merge with existing by simply returning full spec; caller should merge
            return yaml.safe_dump(spec, sort_keys=False)
        # Generic minimal stub
        return yaml.safe_dump({
            "domain": domain_name,
            "personas": [{"id": "p_default", "style": "neutral", "tone": "helpful"}],
            "policies": ["Follow domain rules."],
            "domain_rows": {},
            "examples": [],
            "tools": [],
        }, sort_keys=False)
    return content


def augment_personas_yaml(yaml_text: str, n: int = 3) -> str:
    llm = LLMClient()
    prompt = (
        "Given this YAML, add N more varied personas (id, style, tone) under 'personas'.\n"
        "Return YAML only."
    )
    content = llm.chat([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"N={n}\nYAML:\n{yaml_text}"},
    ])
    if llm.mock:
        data = yaml.safe_load(yaml_text) or {}
        ps = data.get("personas", [])
        base_idx = len(ps)
        for i in range(n):
            ps.append({"id": f"p_auto_{base_idx+i}", "style": "varied", "tone": "varied"})
        data["personas"] = ps
        return yaml.safe_dump(data, sort_keys=False)
    return content
