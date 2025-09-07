from __future__ import annotations
import os
import json
import re
import logging
from typing import Any, Dict, List, Tuple

log = logging.getLogger("apigen_mt.llm")

import litellm
litellm._turn_on_debug()

class LLMClient:
    def __init__(self):
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.mock = os.getenv("MOCK_LLM", "false").lower() in ("1", "true", "yes")
        self._client = None
        self._last_error: str | None = None
        self._timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "15"))
        if not self.mock:
            try:
                from litellm import completion
                self._completion = completion
            except Exception as e:
                # Fallback to mock if LiteLLM import fails
                self._last_error = f"LiteLLM import failed: {e}"
                self.mock = True
                log.warning(self._last_error)

    def chat(self, messages: List[Dict[str, Any]], *, response_format: str | None = None, temperature: float = 0.2) -> str:
        # Validate messages before sending to API
        validated_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                # Ensure assistant messages have content or tool_calls
                if "content" not in msg and "tool_calls" not in msg:
                    msg = dict(msg)  # Make a copy
                    msg["content"] = ""
                elif "content" in msg and msg["content"] is None:
                    msg = dict(msg)  # Make a copy
                    msg["content"] = ""
            validated_messages.append(msg)

        if self.mock:
            # Deterministic mock: echo the last user message with a placeholder
            last = next((m for m in reversed(validated_messages) if m["role"] in ("user", "system")), {"content": ""})
            return f"MOCK[{response_format or 'text'}]: {last.get('content','')}"
        try:
            res = self._completion(
                model=self.model,
                messages=validated_messages,
                temperature=temperature,
                max_retries=0,
                request_timeout=self._timeout,
            )
            return res["choices"][0]["message"]["content"]
        except Exception as e:
            # Network/provider failure; fallback to mock mode for robustness
            self._last_error = f"LLM call failed: {e}"
            self.mock = True
            log.warning(self._last_error)
            last = next((m for m in reversed(validated_messages) if m["role"] in ("user", "system")), {"content": ""})
            return f"MOCK[{response_format or 'text'}]: {last.get('content','')}"

    def chat_tools(self, messages: List[Dict[str, Any]], *, tools: List[Dict[str, Any]], tool_choice: str = "auto", temperature: float = 0.4) -> Dict[str, Any]:
        """Tool-aware chat. Returns a dict describing either a message or a tool_call.

        Return schema:
          {"type": "message", "content": str}
          or
          {"type": "tool_call", "name": str, "arguments": dict, "tool_call_id": str, "raw_assistant_message": {...}}
        """
        if self.mock:
            # Minimal mock: just reply conversationally without tools
            content = self.chat(messages, temperature=temperature)
            return {"type": "message", "content": content}

        # Validate like chat()
        validated_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "assistant" and "content" not in msg and "tool_calls" not in msg:
                msg = dict(msg)
                msg["content"] = ""
            validated_messages.append(msg)

        try:
            res = self._completion(
                model=self.model,
                messages=validated_messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False,
                temperature=temperature,
                max_retries=0,
                request_timeout=self._timeout,
            )
            msg = res.choices[0].message
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                import json as _json
                args = _json.loads(tc.function.arguments or "{}")
                raw_assistant_message = {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                    ],
                }
                return {
                    "type": "tool_call",
                    "name": tc.function.name,
                    "arguments": args,
                    "tool_call_id": tc.id,
                    "raw_assistant_message": raw_assistant_message,
                }
            else:
                return {"type": "message", "content": (msg.content or "").strip()}
        except Exception as e:
            # Harden against network errors: switch to mock and produce a simple message
            self._last_error = f"LLM tool-call failed: {e}"
            self.mock = True
            log.warning(self._last_error)
            content = self.chat(messages, temperature=temperature)
            return {"type": "message", "content": content}

    def judge_committee(self, prompt: str, k: int = 3) -> List[Dict[str, int]]:
        judges: List[Dict[str, int]] = []
        for i in range(k):
            content = self.chat([
                {"role": "system", "content": "You are a careful data judge. Score 0-5 for each metric as JSON."},
                {"role": "user", "content": prompt},
            ], response_format="json")
            if self.mock:
                judges.append({"correctness": 4, "completeness": 4, "satisfaction": 4, "creativity": 3})
            else:
                # best-effort JSON parsing without hard dependency
                import json
                try:
                    judges.append(json.loads(content))
                except Exception:
                    judges.append({"correctness": 3, "completeness": 3, "satisfaction": 3, "creativity": 3})
        return judges

    def propose_task(self, context: str) -> Tuple[str, Dict[str, Any]]:
        # Returns thought(str), payload(dict with instruction/actions/outputs)
        system = (
            "Propose a realistic multi-step task as STRICT JSON only. "
            "Keys: thought(str), instruction(str), actions(array of {tool_name, arguments}), outputs(array of str). "
            "Rules: (1) Use ONLY tool names provided in the context. "
            "(2) Arguments MUST conform to the provided JSON Schemas exactly (types/required keys). "
            "(3) Use ONLY IDs and values that appear in the Data section; do not invent IDs. "
            "(4) Respect the dependency graph. If a SkeletonHint is present, follow that order for write tools. "
            "(5) outputs must be concise natural-language strings. "
            "(6) Do NOT include any text before/after the JSON and do NOT add extra keys."
        )
        content = self.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": context},
        ], response_format="json")
        if self.mock:
            # Domain-aware mock for offline dev
            if "Domain: airline" in context:
                payload = {
                    "instruction": "Please change my seat and then cancel my booking if needed.",
                    "actions": [
                        {"tool_name": "get_reservation", "arguments": {"reservation_id": "r_100"}},
                        {"tool_name": "change_seat", "arguments": {"reservation_id": "r_100", "passenger_id": "p_001", "seat": "12B"}},
                        {"tool_name": "cancel_reservation", "arguments": {"reservation_id": "r_100", "passenger_id": "p_001", "reason": "change of plans"}},
                    ],
                    "outputs": ["Seat updated to 12B", "Reservation r_100 cancelled"],
                }
            else:
                payload = {
                    "instruction": "Help me cancel my recent order and confirm refund status.",
                    "actions": [
                        {"tool_name": "list_orders", "arguments": {"user_id": "u_001"}},
                        {"tool_name": "get_order", "arguments": {"order_id": "o_100"}},
                        {"tool_name": "cancel_order", "arguments": {"user_id": "u_001", "order_id": "o_100", "reason": "changed mind"}},
                    ],
                    "outputs": ["Order o_100 cancelled", "No refund issued"],
                }
            return "mock-thought", payload
        # Try to parse strict JSON; fall back to extracting the first JSON object
        try:
            data = json.loads(content)
        except Exception:
            # Attempt to extract JSON object from text
            m = re.search(r"\{[\s\S]*\}$", content.strip())
            data = None
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    data = None
            if data is None:
                # Final fallback: switch to mock stub to keep pipeline running
                self.mock = True
                log.warning("Propose task: invalid JSON from LLM; switching to mock. Content preview: %s", content[:200])
                return self.propose_task(context)
        thought = data.get("thought", "")
        payload = {k: v for k, v in data.items() if k != "thought"}
        log.debug("Propose task parsed: instruction_len=%d actions=%d outputs=%d", len(payload.get("instruction", "")), len(payload.get("actions", [])), len(payload.get("outputs", [])))
        return thought, payload

    # Optional helper used by natural agent for summarization
    def summarize(self, bullet_points: List[str]) -> str:
        if self.mock:
            return "\n".join(bullet_points)
        prompt = (
            "Write a concise, helpful summary for the user. Include these points explicitly and keep it friendly.\n"
            + "\n".join(f"- {b}" for b in bullet_points)
        )
        return self.chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ])
