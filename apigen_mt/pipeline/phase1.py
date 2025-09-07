from __future__ import annotations
import json
import random
import hashlib
from typing import Any, Dict, List, Tuple
import itertools

from apigen_mt.env.registry import env_factory
from apigen_mt.graph.api_graph import build_api_graph, graph_summary
from apigen_mt.graph.walk import propose_skeleton
from apigen_mt.llm.lite import LLMClient
from apigen_mt.policies.registry import policy_suite_for
from apigen_mt.schemas import Action, ExecStep, ExecTrace, TaskConfig, Metadata, PolicyChecks, ReviewScores
from apigen_mt.sampling.samplers import build_task_context
from apigen_mt.storage.io import write_jsonl, hash_actions
import jsonschema
from apigen_mt.specs.schema_yaml import load_tools as load_tools_yaml
import logging

log = logging.getLogger("apigen_mt.phase1")


class ValidationCache:
    """Simple in-memory cache for expensive validation results."""

    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size

    def _make_key(self, operation: str, *args) -> str:
        """Create a cache key from operation and arguments."""
        key_data = [operation] + [str(arg) for arg in args]
        return hashlib.md5("|".join(key_data).encode()).hexdigest()

    def get(self, operation: str, *args) -> Any:
        """Get cached result if available."""
        key = self._make_key(operation, *args)
        return self.cache.get(key)

    def set(self, operation: str, result: Any, *args) -> None:
        """Cache a result."""
        key = self._make_key(operation, *args)
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = result

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()


def _format_check(payload: Dict[str, Any], env_tools: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not isinstance(payload.get("instruction"), str):
        errs.append("instruction must be string")
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        errs.append("actions must be non-empty list")
    else:
        for i, a in enumerate(actions):
            if not isinstance(a, dict):
                errs.append(f"action[{i}] not dict")
                continue
            tn = a.get("tool_name")
            if tn not in env_tools:
                errs.append(f"unknown tool: {tn}")
            if "arguments" not in a:
                errs.append(f"action[{i}] missing arguments")
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not all(isinstance(o, str) for o in outputs):
        errs.append("outputs must be list[str]")
    return (len(errs) == 0, errs)


def _order_check(actions: List[Action], env) -> Tuple[bool, List[str]]:
    """Ensure actions respect declared tool dependencies."""
    errs: List[str] = []
    seen = set()
    for idx, a in enumerate(actions):
        tool = env.tools.get(a.tool_name)
        if not tool:
            errs.append(f"unknown tool: {a.tool_name}")
            continue
        missing = [d for d in tool.deps if d not in seen]
        if missing:
            errs.append(f"order: {a.tool_name} requires {missing} before step {idx}")
        seen.add(a.tool_name)
    return (len(errs) == 0, errs)


def _exec_check(actions: List[Action], env, cache: ValidationCache | None = None) -> Tuple[ExecTrace, bool, List[str]]:
    # Create cache key from actions
    actions_key = json.dumps([{"tool_name": a.tool_name, "arguments": a.arguments} for a in actions], sort_keys=True)

    # Check cache first
    if cache:
        cached_result = cache.get("exec_check", actions_key)
        if cached_result:
            return cached_result

    errs: List[str] = []
    pre = env.snapshot()
    steps: List[ExecStep] = []
    for a in actions:
        ok = True
        res = None
        err = None
        try:
            # Validate arguments via JSON Schema
            schema = env.tools[a.tool_name].schema
            jsonschema.validate(instance=a.arguments, schema=schema)
            res = env.execute(a.tool_name, a.arguments)
        except Exception as e:
            ok = False
            err = str(e)
            errs.append(f"exec:{a.tool_name}:{err}")
        steps.append(ExecStep(tool_name=a.tool_name, arguments=a.arguments, ok=ok, result=res, error=err))
        if not ok:
            break
    post = env.snapshot()
    diff = env.diff(pre, post)
    result = ExecTrace(steps=[s.model_dump() for s in steps], pre_state=pre, post_state=post, diff_patch=diff), (len(errs) == 0), errs

    # Cache the result
    if cache:
        cache.set("exec_check", result, actions_key)

    return result


def _policy_check(trace: ExecTrace, domain: str, cache: ValidationCache | None = None) -> Tuple[bool, List[str]]:
    # Create cache key from trace
    trace_key = json.dumps(trace.model_dump(), sort_keys=True)

    # Check cache first
    if cache:
        cached_result = cache.get("policy_check", domain, trace_key)
        if cached_result:
            return cached_result

    suite = policy_suite_for(domain)
    if suite is None:
        result = True, []
    else:
        res = suite.run(trace.model_dump())
        result = res.passed, res.failures

    # Cache the result
    if cache:
        cache.set("policy_check", result, domain, trace_key)

    return result


def committee_review(llm: LLMClient, task_payload: Dict[str, Any], trace: ExecTrace, *, min_correctness: int = 3, min_completeness: int = 3) -> Tuple[bool, Dict[str, int], List[str]]:
    """Enhanced committee review with detailed evaluation criteria."""

    # Build comprehensive evaluation context
    eval_context = {
        "task": {
            "instruction": task_payload.get("instruction", ""),
            "actions": task_payload.get("actions", []),
            "outputs": task_payload.get("outputs", [])
        },
        "execution": {
            "trace_summary": {
                "total_steps": len(trace.steps),
                "successful_steps": sum(1 for s in trace.steps if s.ok),
                "failed_steps": sum(1 for s in trace.steps if not s.ok),
                "state_changes": len(trace.diff_patch)
            },
            "final_state": trace.post_state
        }
    }

    detailed_prompt = f"""
You are an expert evaluator of API interaction tasks. Evaluate this proposed task on multiple dimensions:

TASK DETAILS:
- Instruction: {eval_context['task']['instruction']}
- Number of Actions: {len(eval_context['task']['actions'])}
- Expected Outputs: {len(eval_context['task']['outputs'])}

EXECUTION RESULTS:
- Total Steps: {eval_context['execution']['trace_summary']['total_steps']}
- Successful Steps: {eval_context['execution']['trace_summary']['successful_steps']}
- Failed Steps: {eval_context['execution']['trace_summary']['failed_steps']}
- State Changes: {eval_context['execution']['trace_summary']['state_changes']}

EVALUATION CRITERIA (Rate 0-5 for each):

1. CORRECTNESS: Are the actions logically sound and properly sequenced?
   - Do actions respect API dependencies?
   - Are arguments valid and complete?
   - Does the sequence make business sense?

2. COMPLETENESS: Does the task cover all necessary steps?
   - Are there missing prerequisite actions?
   - Are all required outputs achievable?
   - Is the task self-contained?

3. SATISFACTION: How well does this task serve a realistic user need?
   - Is the scenario plausible and common?
   - Does it provide clear value to the user?
   - Are the outputs meaningful and actionable?

4. CREATIVITY: How novel and varied is this task?
   - Does it combine APIs in interesting ways?
   - Is it different from typical CRUD operations?
   - Does it demonstrate complex workflow understanding?

5. REALISM: How realistic is this task scenario?
   - Do the arguments use plausible values?
   - Is the user intent clearly expressed?
   - Could this be a real customer request?

PROVIDE SPECIFIC FEEDBACK:
- What works well about this task?
- What could be improved?
- Any specific issues with the action sequence?
- Suggestions for making it more realistic?

Format your response as JSON with keys: correctness, completeness, satisfaction, creativity, realism, strengths, weaknesses, suggestions
"""

    judges = llm.judge_committee(detailed_prompt, k=3)

    # Enhanced aggregation with more metrics
    agg = {"correctness": 0, "completeness": 0, "satisfaction": 0, "creativity": 0, "realism": 0}
    detailed_notes = []

    for i, j in enumerate(judges):
        # Aggregate scores
        for k in agg:
            if k in j:
                agg[k] += int(j.get(k, 0))

        # Collect detailed feedback
        feedback_parts = []
        if j.get("strengths"):
            feedback_parts.append(f"Strengths: {j['strengths']}")
        if j.get("weaknesses"):
            feedback_parts.append(f"Issues: {j['weaknesses']}")
        if j.get("suggestions"):
            feedback_parts.append(f"Suggestions: {j['suggestions']}")

        detailed_notes.append(f"Judge {i+1}: {'; '.join(feedback_parts)}")

    # Average the scores
    for k in agg:
        agg[k] = round(agg[k] / len(judges))

    # Enhanced passing criteria
    passed = (agg["correctness"] >= min_correctness and
              agg["completeness"] >= min_completeness and
              agg["realism"] >= 3)  # Require minimum realism

    return passed, agg, detailed_notes


def reflect_context(base_ctx: str, errs: List[str]) -> str:
    return base_ctx + "\nFEEDBACK:" + "; ".join(errs) + "\nPlease regenerate correcting these issues."


import itertools
from apigen_mt.storage.io import hash_actions

def recombine_blueprints(
    base_blueprints: List[Dict[str, Any]],
    *,
    domain: str,
    llm: LLMClient,
    env,
    min_correctness: int = 3,
    min_completeness: int = 3,
) -> List[Dict[str, Any]]:
    """Compose longer tasks from validated blueprints."""
    log.info("Starting blueprint recombination for %d candidates...", len(base_blueprints))
    recombined: List[Dict[str, Any]] = []
    
    # 1. Group by persona
    by_persona: Dict[str, List[Dict[str, Any]]] = {}
    for bp in base_blueprints:
        persona_id = bp["metadata"]["persona_id"]
        by_persona.setdefault(persona_id, []).append(bp)

    base_state = env.snapshot()

    for persona_id, bps in by_persona.items():
        if len(bps) < 2:
            continue
        
        # 2. Create combinations (pairs)
        for bp1, bp2 in itertools.combinations(bps, 2):
            log.info("Attempting to combine two blueprints for persona %s", persona_id)
            env.restore(base_state)

            # 3. Concatenate actions and outputs
            t1 = TaskConfig(**bp1)
            t2 = TaskConfig(**bp2)
            combined_actions = t1.actions + t2.actions
            combined_outputs = t1.outputs + t2.outputs

            # 4. Re-run execution and policy checks
            trace, ok_exec, errs = _exec_check(combined_actions, env)
            if not ok_exec:
                log.warning("Recombination failed: combined execution check failed: %s", errs)
                continue

            ok_policy, perrs = _policy_check(trace, domain)
            if not ok_policy:
                log.warning("Recombination failed: combined policy check failed: %s", perrs)
                continue

            # 5. Synthesize a new instruction
            instruction_prompt = f'''Given the following two user requests:
1. '{t1.instruction}'
2. '{t2.instruction}'

And the combined plan of actions to fulfill them:
{[a.model_dump() for a in combined_actions]}

Write a single, coherent user instruction that naturally asks for both outcomes.
For example, if one task is 'book a flight' and the other is 'rent a car', a good combined instruction would be 'I need to book a flight to LA and then rent a car there for 5 days.'
The instruction should be a single sentence or two short, related sentences.
'''
            new_instruction = llm.chat([{"role": "user", "content": instruction_prompt}]).strip()
            log.info("Synthesized new instruction: %s", new_instruction)

            # 6. Re-run alignment validation
            new_payload = {
                "instruction": new_instruction,
                "actions": [a.model_dump() for a in combined_actions],
                "outputs": combined_outputs,
            }
            passed, scores, notes = committee_review(llm, new_payload, trace, min_correctness=min_correctness, min_completeness=min_completeness)
            if not passed:
                log.info("Recombination failed: committee rejected new combined task. Scores: %s, Notes: %s", scores, notes)
                continue

            # 7. Create and store the new blueprint
            meta = Metadata(
                persona_id=persona_id,
                domain=domain,
                policy_checks=PolicyChecks(passed=True, failures=[]),
                review_scores=ReviewScores(**scores),
                recombined_from = [hash_actions([a.model_dump() for a in t1.actions]), hash_actions([a.model_dump() for a in t2.actions])]
            )

            cfg = TaskConfig(
                instruction=new_instruction,
                actions=combined_actions,
                outputs=combined_outputs,
                diff_patch=trace.diff_patch,
                metadata=meta,
            )
            recombined.append(cfg.model_dump())
            log.info("Successfully recombined blueprint. New task has %d actions.", len(combined_actions))

    log.info("Finished blueprint recombination. Created %d new blueprints.", len(recombined))
    return recombined

# --- Planner-first candidates (grounded, domain-aware) ---
def _retail_planner_candidates(env, rng: random.Random, k: int = 3) -> List[Dict[str, Any]]:
    """Produce grounded candidate payloads for the retail domain without relying on LLM.

    Each payload has: instruction(str), actions(list[{tool_name, arguments}]), outputs(list[str])
    """
    state = env.snapshot() or {}
    users = list((state.get("users") or {}).items())
    orders = list((state.get("orders") or {}).items())
    rng.shuffle(users)
    rng.shuffle(orders)

    cands: List[Dict[str, Any]] = []

    # Pattern A: Update address (read before write)
    if users:
        uid, u = users[0]
        new_addr = f"{rng.randint(10, 999)} New Lane, Springfield"
        actions = [
            {"tool_name": "get_user_info", "arguments": {"user_id": uid}},
            {"tool_name": "update_address", "arguments": {"user_id": uid, "address": new_addr}},
        ]
        instruction = f"Update {u.get('name','the user')}'s shipping address to '{new_addr}'."
        outputs = [
            f"Fetched {u.get('name','user')}'s user info.",
            f"{u.get('name','User')}'s shipping address has been updated to {new_addr}.",
        ]
        cands.append({"instruction": instruction, "actions": actions, "outputs": outputs})

    # Pattern B: Cancel an active order (processing/shipped)
    for oid, o in orders:
        if o.get("status") in ("processing", "shipped"):
            uid = o.get("user_id")
            actions = [
                {"tool_name": "list_orders", "arguments": {"user_id": uid}},
                {"tool_name": "get_order", "arguments": {"order_id": oid}},
                {"tool_name": "cancel_order", "arguments": {"user_id": uid, "order_id": oid, "reason": "changed mind"}},
            ]
            instruction = f"Cancel order {oid} for the correct account and confirm status."
            outputs = [f"Order {oid} cancelled"]
            cands.append({"instruction": instruction, "actions": actions, "outputs": outputs})
            break

    # Pattern C: Refund a delivered order (partial)
    for oid, o in orders:
        if o.get("status") == "delivered":
            total = float(o.get("total", 0.0))
            amount = round(min(50.0, max(10.0, total * 0.4)), 2)
            actions = [
                {"tool_name": "get_order", "arguments": {"order_id": oid}},
                {"tool_name": "refund_order", "arguments": {"order_id": oid, "amount": amount}},
            ]
            instruction = f"Process a ${amount} refund for the delivered order {oid}."
            outputs = [f"Processed a ${amount} refund for order {oid}."]
            cands.append({"instruction": instruction, "actions": actions, "outputs": outputs})
            break

    return cands[:k]

def generate_blueprints(*, domain: str = "retail", count: int = 5, max_attempts: int = 5, seed: int = 7, out_path: str = "data/blueprints.jsonl", best_of: int = 3, skip_committee: bool = False, min_correctness: int = 3, min_completeness: int = 3, force_skeleton: bool = True, recombine: bool = False):
    log.info("Phase1 start | domain=%s count=%s attempts=%s seed=%s best_of=%s skip_committee=%s", domain, count, max_attempts, seed, best_of, skip_committee)
    rng = random.Random(seed)
    env = env_factory(domain)

    g = build_api_graph(env)
    gsum = graph_summary(g)
    llm = LLMClient()

    # Initialize validation cache for expensive operations
    validation_cache = ValidationCache(max_size=50)

    # Progress tracking
    progress = {
        "total_candidates": count,
        "accepted": 0,
        "rejected": 0,
        "total_attempts": 0,
        "validation_failures": {
            "format": 0,
            "order": 0,
            "execution": 0,
            "policy": 0,
            "committee": 0
        },
        "cache_hits": 0,
        "cache_misses": 0
    }

    # Ensure each blueprint candidate starts from the same clean state
    base_state = env.snapshot()
    accepted: List[Dict[str, Any]] = []

    for candidate_idx in range(count):
        env.restore(base_state)
        candidate_num = candidate_idx + 1
        log.info("Generating candidate blueprint (%d/%d)", candidate_num, count)
        # Provide tool schemas and a skeleton hint to the LLM
        tool_specs = {name: t.schema for name, t in env.tools.items()}
        tool_catalog = load_tools_yaml(domain).get("tools", [])
        skeleton = propose_skeleton(g, rng)
        log.debug("Graph summary: %s", gsum)
        log.debug("Skeleton hint: %s", skeleton)
        ctx = build_task_context(gsum, domain, rng, tool_specs=tool_specs, skeleton=skeleton if force_skeleton else [], tool_catalog=tool_catalog)

        # 0) Planner-first attempt (domain-aware) to reduce invalid candidates
        if domain.lower() == "retail":
            log.info("Trying planner-first grounded candidates (best_of=%d)", best_of)
            planned = _retail_planner_candidates(env, rng, k=best_of)
            for p in planned:
                # Validate and score like LLM proposals
                ok_fmt, fe = _format_check(p, env.tools)
                if not ok_fmt:
                    progress["validation_failures"]["format"] += 1
                    continue
                actions = [Action(**a) for a in p["actions"]]
                ok_order, oerrs = _order_check(actions, env)
                if not ok_order:
                    progress["validation_failures"]["order"] += 1
                    continue
                actions_key = json.dumps([{"tool_name": a.tool_name, "arguments": a.arguments} for a in actions], sort_keys=True)
                pre_cached = bool(validation_cache.get("exec_check", actions_key)) if validation_cache else False
                trace, ok_exec, errs = _exec_check(actions, env, validation_cache)
                if validation_cache:
                    if pre_cached:
                        progress["cache_hits"] += 1
                    else:
                        progress["cache_misses"] += 1
                if not ok_exec:
                    progress["validation_failures"]["execution"] += 1
                    env.restore(trace.pre_state)
                    continue
                trace_key = json.dumps(trace.model_dump(), sort_keys=True)
                pre_cached_pol = bool(validation_cache.get("policy_check", domain, trace_key)) if validation_cache else False
                ok_policy, perrs = _policy_check(trace, domain, validation_cache)
                if validation_cache:
                    if pre_cached_pol:
                        progress["cache_hits"] += 1
                    else:
                        progress["cache_misses"] += 1
                if not ok_policy:
                    progress["validation_failures"]["policy"] += 1
                    env.restore(trace.pre_state)
                    continue
                if not skip_committee:
                    passed, scores, notes = committee_review(llm, p, trace, min_correctness=min_correctness, min_completeness=min_completeness)
                    if not passed:
                        progress["validation_failures"]["committee"] += 1
                        env.restore(trace.pre_state)
                        continue
                else:
                    scores = {"correctness": 5, "completeness": 5, "satisfaction": 4, "creativity": 3, "realism": 4}

                # Accept planner blueprint
                meta = Metadata(
                    persona_id="auto",
                    domain=domain,
                    policy_checks=PolicyChecks(passed=True, failures=[]),
                    review_scores=ReviewScores(**scores),
                )
                cfg = TaskConfig(
                    instruction=p["instruction"],
                    actions=actions,
                    outputs=p["outputs"],
                    diff_patch=trace.diff_patch,
                    metadata=meta,
                )
                # dedup
                act_hash = hash_actions([a.model_dump() for a in actions])
                if any(hash_actions([Action(**x).model_dump() for x in b["actions"]]) == act_hash for b in accepted):
                    progress["rejected"] += 1
                    env.restore(trace.pre_state)
                    continue
                accepted.append(cfg.model_dump())
                progress["accepted"] += 1
                progress["total_attempts"] += 1
                success_rate = progress["accepted"] / candidate_num if candidate_num > 0 else 0
                log.info("Accepted planner blueprint (%d/%d): actions=%d outputs=%d diff_keys=%d | Success rate: %.1f%%",
                        progress["accepted"], count, len(cfg.actions), len(cfg.outputs), len(cfg.diff_patch), success_rate * 100)
                env.restore(trace.pre_state)
                break
            if progress["accepted"] >= candidate_num:
                # move to next candidate
                continue

        # 1) LLM proposal attempts
        for _ in range(max_attempts):
            log.info("LLM proposing task (best_of=%d)", best_of)
            payload = None
            thought = ""
            fmt_errs: List[str] = []
            # Try best_of candidates and pick the best by successful execution/policy and fewer steps
            candidates: List[Tuple[Dict[str, Any], ExecTrace]] = []
            for i in range(best_of):
                t, p = llm.propose_task(ctx)
                ok_fmt, fe = _format_check(p, env.tools)
                if not ok_fmt:
                    fmt_errs = fe
                    continue
                actions_i = [Action(**a) for a in p["actions"]]
                ok_order, oerrs = _order_check(actions_i, env)
                if not ok_order:
                    continue
                trace_i, ok_exec_i, errs_i = _exec_check(actions_i, env, validation_cache)
                if not ok_exec_i:
                    env.restore(trace_i.pre_state)
                    continue
                ok_policy_i, perrs_i = _policy_check(trace_i, domain, validation_cache)
                if not ok_policy_i:
                    env.restore(trace_i.pre_state)
                    continue
                candidates.append((p, trace_i))
                env.restore(trace_i.pre_state)
            # pick best by (fewest failed steps==0, min total steps, min writes)
            if candidates:
                def _score(item: Tuple[Dict[str, Any], ExecTrace]) -> Tuple[int, int, int]:
                    _p, tr = item
                    steps = len(tr.steps)
                    # ExecTrace.steps is a list of ExecStep models
                    writes = 0
                    for s in tr.steps:
                        tool = env.tools.get(getattr(s, "tool_name", None))
                        if tool and tool.write:
                            writes += 1
                    return (steps, writes, 0)
                payload, trace = sorted(candidates, key=_score)[0]
            else:
                payload = None
            if payload is None:
                log.warning("Format check failed across best_of candidates: %s", fmt_errs)
                progress["validation_failures"]["format"] += 1
                ctx = reflect_context(ctx, fmt_errs)
                continue
            log.debug("LLM thought: %s", (thought or "")[:200])

            actions = [Action(**a) for a in payload["actions"]]
            ok_order, oerrs = _order_check(actions, env)
            if not ok_order:
                log.warning("Order check failed; attempting auto-reorder: %s", oerrs)
                progress["validation_failures"]["order"] += 1
                # naive reorder: move deps earlier if present
                name_to_action = {a.tool_name: a for a in actions}
                ordered: List[Action] = []
                placed = set()
                def place(tool_name: str):
                    if tool_name in placed:
                        return
                    tool = env.tools.get(tool_name)
                    if not tool:
                        return
                    for d in tool.deps:
                        if d in name_to_action:
                            place(d)
                    if tool_name in name_to_action:
                        ordered.append(name_to_action[tool_name])
                        placed.add(tool_name)
                for a in actions:
                    place(a.tool_name)
                # append any leftover in original order
                for a in actions:
                    if a.tool_name not in placed:
                        ordered.append(a)
                        placed.add(a.tool_name)
                actions = ordered
                ok_order2, oerrs2 = _order_check(actions, env)
                if not ok_order2:
                    log.warning("Auto-reorder did not satisfy deps: %s", oerrs2)
                    ctx = reflect_context(ctx, oerrs)
                    continue
            # Check cache for execution results (pre-check)
            actions_key = json.dumps([{"tool_name": a.tool_name, "arguments": a.arguments} for a in actions], sort_keys=True)
            pre_cached = bool(validation_cache.get("exec_check", actions_key)) if validation_cache else False
            trace, ok_exec, errs = _exec_check(actions, env, validation_cache)
            if validation_cache:
                if pre_cached:
                    progress["cache_hits"] += 1
                else:
                    progress["cache_misses"] += 1

            if not ok_exec:
                log.warning("Execution check failed: %s", errs)
                progress["validation_failures"]["execution"] += 1
                ctx = reflect_context(ctx, errs)
                # Reset env to pre state for next attempt
                env.restore(trace.pre_state)
                continue

            trace_key = json.dumps(trace.model_dump(), sort_keys=True)
            pre_cached_pol = bool(validation_cache.get("policy_check", domain, trace_key)) if validation_cache else False
            ok_policy, perrs = _policy_check(trace, domain, validation_cache)
            if validation_cache:
                if pre_cached_pol:
                    progress["cache_hits"] += 1
                else:
                    progress["cache_misses"] += 1

            if not ok_policy:
                log.warning("Policy check failed: %s", perrs)
                progress["validation_failures"]["policy"] += 1
                ctx = reflect_context(ctx, perrs)
                env.restore(trace.pre_state)
                continue

            if not skip_committee:
                passed, scores, notes = committee_review(llm, payload, trace, min_correctness=min_correctness, min_completeness=min_completeness)
                if not passed:
                    log.info("Committee rejected: scores=%s notes=%s", scores, notes)
                    progress["validation_failures"]["committee"] += 1
                    ctx = reflect_context(ctx, notes)
                    env.restore(trace.pre_state)
                    continue
            else:
                scores = {"correctness": 5, "completeness": 5, "satisfaction": 4, "creativity": 3, "realism": 4}

            # Accepted
            progress["accepted"] += 1
            progress["total_attempts"] += 1

            meta = Metadata(
                persona_id="auto",
                domain=domain,
                policy_checks=PolicyChecks(passed=True, failures=[]),
                review_scores=ReviewScores(**scores),
            )
            cfg = TaskConfig(
                instruction=payload["instruction"],
                actions=actions,
                outputs=payload["outputs"],
                diff_patch=trace.diff_patch,
                metadata=meta,
            )
            # de-duplicate by actions
            act_hash = hash_actions([a.model_dump() for a in actions])
            if any(hash_actions([Action(**x).model_dump() for x in b["actions"]]) == act_hash for b in accepted):
                log.info("Duplicate blueprint actions; skipping")
                progress["rejected"] += 1
                env.restore(trace.pre_state)
                continue
            accepted.append(cfg.model_dump())

            # Progress summary
            success_rate = progress["accepted"] / candidate_num if candidate_num > 0 else 0
            log.info("Accepted blueprint (%d/%d): actions=%d outputs=%d diff_keys=%d | Success rate: %.1f%%",
                    progress["accepted"], count, len(cfg.actions), len(cfg.outputs), len(cfg.diff_patch),
                    success_rate * 100)

            # Reset env to the pre-state for next blueprint candidate
            env.restore(trace.pre_state)
            break

    if recombine and accepted:
        recombined_bps = recombine_blueprints(
            accepted,
            domain=domain,
            llm=llm,
            env=env,
            min_correctness=min_correctness,
            min_completeness=min_completeness,
        )
        accepted.extend(recombined_bps)

    if accepted:
        write_jsonl(out_path, accepted)
        log.info("Wrote %d blueprints -> %s", len(accepted), out_path)

    # Final progress summary
    total_attempts = sum(progress["validation_failures"].values()) + progress["accepted"]
    success_rate = progress["accepted"] / count if count > 0 else 0
    cache_total = progress["cache_hits"] + progress["cache_misses"]
    cache_hit_rate = progress["cache_hits"] / cache_total if cache_total > 0 else 0

    log.info("Phase 1 Complete | Success rate: %.1f%% (%d/%d) | Total attempts: %d",
             success_rate * 100, progress["accepted"], count, total_attempts)
    log.info("Validation failures: Format=%d, Order=%d, Execution=%d, Policy=%d, Committee=%d",
             progress["validation_failures"]["format"],
             progress["validation_failures"]["order"],
             progress["validation_failures"]["execution"],
             progress["validation_failures"]["policy"],
             progress["validation_failures"]["committee"])
    log.info("Cache performance: %.1f%% hit rate (%d/%d)",
             cache_hit_rate * 100, progress["cache_hits"], cache_total)

    return accepted
