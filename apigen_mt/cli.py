from __future__ import annotations
import argparse
import os
from typing import Any, Dict

from apigen_mt.pipeline.phase1 import generate_blueprints
from apigen_mt.pipeline.phase2 import rollout_blueprints
from apigen_mt.storage.io import read_jsonl
from apigen_mt.logging_utils import setup_logging


def main():
    p = argparse.ArgumentParser(prog="apigen-mt")
    p.add_argument("--log-level", default=os.getenv("APIGEN_LOG_LEVEL", "INFO"), help="DEBUG, INFO, WARNING, ERROR")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("gen-blueprints")
    p1.add_argument("--domain", default="retail")
    p1.add_argument("--count", type=int, default=5)
    p1.add_argument("--max-attempts", type=int, default=5)
    p1.add_argument("--seed", type=int, default=7)
    p1.add_argument("--out", default="data/blueprints.jsonl")
    # Accept both hyphen and underscore variants for convenience
    import argparse as _argparse
    p1.add_argument("--best-of", dest="best_of", type=int, default=3, help="Number of proposals to sample per attempt; pick best candidate")
    p1.add_argument("--best_of", dest="best_of", type=int, help=_argparse.SUPPRESS)
    p1.add_argument("--skip-committee", dest="skip_committee", action="store_true", help="Skip committee review to improve acceptance during bootstrap")
    p1.add_argument("--skip_committee", dest="skip_committee", action="store_true", help=_argparse.SUPPRESS)
    p1.add_argument("--min-correctness", dest="min_correctness", type=int, default=3)
    p1.add_argument("--min_correctness", dest="min_correctness", type=int, help=_argparse.SUPPRESS)
    p1.add_argument("--min-completeness", dest="min_completeness", type=int, default=3)
    p1.add_argument("--min_completeness", dest="min_completeness", type=int, help=_argparse.SUPPRESS)
    p1.add_argument("--force-skeleton", dest="force_skeleton", action="store_true", default=True, help="Enforce using the graph skeleton in prompts (default: True)")
    p1.add_argument("--force_skeleton", dest="force_skeleton", action="store_true", help=_argparse.SUPPRESS)
    p1.add_argument("--recombine", action="store_true", help="After generating blueprints, recombine them to create more complex ones")

    p3 = sub.add_parser("roll-out")
    p3.add_argument("--domain", default="retail")
    p3.add_argument("--trials", type=int, default=3)
    p3.add_argument("--agent", choices=["replay", "llm", "natural"], default="replay")
    p3.add_argument("--in", dest="inp", default="data/blueprints.jsonl")
    p3.add_argument("--out", default="data/trajectories.jsonl")

    p4 = sub.add_parser("gen-domain-spec")
    p4.add_argument("--name", required=True)
    p4.add_argument("--out", default=None, help="Defaults to configs/sampling/<name>.yaml")
    p4.add_argument("--hint", action="append", default=[], help="key=value hint; can repeat")

    p5 = sub.add_parser("augment-personas")
    p5.add_argument("--domain", required=True)
    p5.add_argument("--n", type=int, default=3)
    p5.add_argument("--path", default=None, help="Path to domain YAML; defaults to configs/sampling/<domain>.yaml")

    args = p.parse_args()
    setup_logging(args.log_level)

    if args.cmd == "gen-blueprints":
        res = generate_blueprints(domain=args.domain, count=args.count, max_attempts=args.max_attempts, seed=args.seed, out_path=args.out,
                                  best_of=args.best_of, skip_committee=args.skip_committee, min_correctness=args.min_correctness,
                                  min_completeness=args.min_completeness, force_skeleton=args.force_skeleton, recombine=args.recombine)
        print(f"generated: {len(res)} blueprints -> {args.out}")
    elif args.cmd == "roll-out":
        bps = read_jsonl(args.inp)
        res = rollout_blueprints(bps, domain=args.domain, trials=args.trials, agent_mode=args.agent, out_path=args.out)
        print(f"kept {len(res)} successful trajectories -> {args.out}")
    elif args.cmd == "gen-domain-spec":
        from apigen_mt.llm.schema_gen import generate_domain_yaml
        from apigen_mt.specs.schema_yaml import load_tools as load_tools_yaml, write_tools as write_tools_yaml, merge_tools, split_domain_yaml
        from apigen_mt.sampling.yaml_loader import load_domain as load_sampling_yaml
        hints = {}
        for kv in args.hint:
            if "=" in kv:
                k, v = kv.split("=", 1)
                hints[k] = v
        yml = generate_domain_yaml(args.name, hints=hints)
        import yaml as _yaml
        data = _yaml.safe_load(yml) or {}
        sampling_doc, tools_doc = split_domain_yaml(data)

        # Merge sampling YAML
        samp_path = args.out or f"configs/sampling/{args.name.lower()}.yaml"
        os.makedirs(os.path.dirname(samp_path), exist_ok=True)
        # Load existing sampling to merge and dedup
        existing_sampling = load_sampling_yaml(args.name)
        # Personas: dedup by id
        ex_p = {p.get("id"): p for p in existing_sampling.get("personas", []) if isinstance(p, dict)}
        for p in sampling_doc.get("personas", []) or []:
            pid = p.get("id") if isinstance(p, dict) else None
            if pid and pid not in ex_p:
                ex_p[pid] = p
        existing_sampling["personas"] = list(ex_p.values())
        # Policies/examples: set-dedup
        for key in ("policies", "examples"):
            vals = set(existing_sampling.get(key, []) or [])
            for v in sampling_doc.get(key, []) or []:
                vals.add(v)
            existing_sampling[key] = sorted(vals)
        # Domain rows: append-dedup by dict repr
        dr = existing_sampling.get("domain_rows", {}) or {}
        for tbl, rows in (sampling_doc.get("domain_rows", {}) or {}).items():
            cur = dr.get(tbl, [])
            seen = {repr(x) for x in cur}
            for r in rows or []:
                if repr(r) not in seen:
                    cur.append(r)
                    seen.add(repr(r))
            dr[tbl] = cur
        existing_sampling["domain_rows"] = dr
        with open(samp_path, "w", encoding="utf-8") as f:
            _yaml.safe_dump(existing_sampling, f, sort_keys=False)
        print(f"updated sampling YAML -> {samp_path}")

        # Merge tools YAML
        existing_tools_doc = load_tools_yaml(args.name)
        merged_tools = merge_tools(existing_tools_doc.get("tools", []), tools_doc.get("tools", []))
        existing_tools_doc["domain"] = args.name
        existing_tools_doc["tools"] = merged_tools
        write_tools_yaml(args.name, existing_tools_doc)
        print(f"updated tools YAML -> configs/schemas/{args.name.lower()}.yaml")
    elif args.cmd == "augment-personas":
        import yaml
        from apigen_mt.llm.schema_gen import augment_personas_yaml
        path = args.path or f"configs/sampling/{args.domain.lower()}.yaml"
        if not os.path.exists(path):
            raise SystemExit(f"domain yaml not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            yml = f.read()
        updated = augment_personas_yaml(yml, n=args.n)
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
        data = yaml.safe_load(updated) or {}
        print(f"personas now: {len(data.get('personas', []))} in {path}")


if __name__ == "__main__":
    main()
