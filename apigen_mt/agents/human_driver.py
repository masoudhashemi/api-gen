from __future__ import annotations
from typing import Any, Dict, List
import random
import json
from apigen_mt.llm.lite import LLMClient


def derive_valid_ids(rows: Dict[str, Any]) -> Dict[str, List[str]]:
    ids: Dict[str, List[str]] = {}
    for tbl, recs in (rows or {}).items():
        for r in recs or []:
            if isinstance(r, dict):
                for k, v in r.items():
                    if isinstance(k, str) and k.endswith("_id") and isinstance(v, (str, int)):
                        v_str = str(v)
                        ids.setdefault(k, [])
                        if v_str not in ids[k]:
                            ids[k].append(v_str)
    return ids


class HumanDriver:
    """An enhanced Human LM driver that generates natural, varied responses using LLM.

    Strategy:
    1. Uses LLM to generate natural responses based on persona and context
    2. Reveals information incrementally and naturally
    3. Adapts tone and style based on persona
    4. Maintains conversation coherence
    """

    def __init__(self, persona: Dict[str, Any], blueprint, domain_rows: Dict[str, Any], rng: random.Random):
        self.persona = persona or {"id": "p_default", "style": "neutral", "tone": "helpful"}
        self.blueprint = blueprint
        self.rows = domain_rows or {}
        self.valid_ids = derive_valid_ids(self.rows)
        self.rng = rng
        self.llm = LLMClient()
        self.next_action_idx = 0
        self.conversation_history = []
        self.revealed_info = set()
        self.facts: Dict[str, str] = {}

        # Seed stable facts from blueprint actions to ensure consistency
        try:
            for a in (self.blueprint.actions or []):
                for k, v in (a.arguments or {}).items():
                    if isinstance(v, (str, int)):
                        self.facts.setdefault(k, str(v))
        except Exception:
            pass

        # Seed user_id from instruction if a known user name is mentioned
        try:
            users = (self.rows or {}).get("users") or []
            instr_l = (self.blueprint.instruction or "").lower()
            for u in users:
                name = str(u.get("name", ""))
                uid = str(u.get("user_id", ""))
                if name and uid and name.lower() in instr_l:
                    self.facts.setdefault("user_id", uid)
                    self.facts.setdefault("name", name)
                    break
        except Exception:
            pass
        # If user_id known from blueprint actions, derive name from rows
        try:
            if "user_id" in self.facts:
                uid = self.facts["user_id"]
                for u in (self.rows or {}).get("users", []) or []:
                    if str(u.get("user_id", "")) == uid and u.get("name"):
                        self.facts.setdefault("name", str(u.get("name")))
                        break
        except Exception:
            pass

    def _arg_from_blueprint(self, key: str) -> str | None:
        # Look at next action first, then search subsequent actions
        actions = self.blueprint.actions
        for i in range(self.next_action_idx, min(self.next_action_idx + 3, len(actions))):
            v = actions[i].arguments.get(key)
            if v is not None:
                return str(v)
        for a in actions:
            v = a.arguments.get(key)
            if v is not None:
                return str(v)
        return None

    def _arg_from_rows(self, key: str) -> str | None:
        vals = self.valid_ids.get(key)
        if vals:
            return self.rng.choice(vals)
        return None

    def _build_persona_prompt(self) -> str:
        """Build a detailed persona prompt for the LLM."""
        persona = self.persona
        style = persona.get("style", "neutral")
        tone = persona.get("tone", "helpful")

        prompt = f"You are a {style} person with a {tone} tone. "

        if style == "professional":
            prompt += "You communicate clearly and efficiently. "
        elif style == "casual":
            prompt += "You use conversational language and contractions. "
        elif style == "formal":
            prompt += "You use proper grammar and formal language. "

        if tone == "polite":
            prompt += "You are very courteous and use phrases like 'please' and 'thank you'. "
        elif tone == "impatient":
            prompt += "You are direct and want things done quickly. "
        elif tone == "helpful":
            prompt += "You are cooperative and provide information willingly. "

        prompt += "Respond naturally to the assistant's messages while revealing only the information that's directly relevant to their questions."

        return prompt

    def _extract_relevant_info(self, requested_slots: List[str]) -> Dict[str, str]:
        """Extract relevant information that should be revealed."""
        info = {}
        for key in requested_slots:
            if key in self.revealed_info:
                continue  # Don't repeat already revealed info

            # Prefer known facts first, then blueprint. Avoid random user_id.
            val: str | None = None
            if key in self.facts:
                val = self.facts[key]
            elif key == "user_id":
                val = self._arg_from_blueprint("user_id")
                if not val:
                    # try map by mentioned name in instruction
                    v_name = self.facts.get("name")
                    if v_name:
                        users = (self.rows or {}).get("users") or []
                        for u in users:
                            if str(u.get("name", "")).lower() == v_name.lower():
                                uid = str(u.get("user_id", ""))
                                if uid:
                                    val = uid
                                    break
                # do not fall back to random for user_id
            elif key == "order_id":
                val = self._arg_from_blueprint("order_id")
            elif key == "name":
                # derive name from known user_id
                if "name" in self.facts:
                    val = self.facts["name"]
                elif "user_id" in self.facts:
                    uid = self.facts["user_id"]
                    for u in (self.rows or {}).get("users", []) or []:
                        if str(u.get("user_id", "")) == uid and u.get("name"):
                            val = str(u.get("name"))
                            break
                else:
                    # fallback: pick any known name deterministically
                    us = (self.rows or {}).get("users", []) or []
                    if us:
                        val = str(us[0].get("name", ""))
            else:
                val = self._arg_from_blueprint(key) or self._arg_from_rows(key)
            if val is not None:
                info[key] = val
                self.revealed_info.add(key)
                self.facts.setdefault(key, val)

        return info

    def reply(self, last_assistant: str, requested_slots: List[str]) -> str:
        """Generate a natural response using LLM."""
        if self.llm.mock:
            # Fallback to original logic for mock mode
            return self._fallback_reply(last_assistant, requested_slots)

        # Update facts from assistant text if explicit IDs are mentioned
        try:
            import re as _re
            if last_assistant:
                m_uid = _re.search(r"u_\d+", last_assistant, flags=_re.IGNORECASE)
                if m_uid:
                    self.facts.setdefault("user_id", m_uid.group(0))
                m_oid = _re.search(r"o_\d+", last_assistant, flags=_re.IGNORECASE)
                if m_oid:
                    self.facts.setdefault("order_id", m_oid.group(0))
                m_name = _re.search(r"name[^A-Za-z0-9]*is[^A-Za-z0-9]*([A-Z][a-zA-Z]+)", last_assistant)
                if m_name:
                    self.facts.setdefault("name", m_name.group(1))
        except Exception:
            pass

        # Extract relevant information
        relevant_info = self._extract_relevant_info(requested_slots)

        # Build conversation context
        context = f"""
Original request: {self.blueprint.instruction}

Assistant's last message: {last_assistant}

Conversation history:
{chr(10).join(f"- {msg}" for msg in self.conversation_history[-3:])}

Requested information: {', '.join(requested_slots) if requested_slots else 'None specific'}
Available information to share: {json.dumps(relevant_info, indent=2)}
"""

        system_prompt = self._build_persona_prompt()

        user_prompt = f"""
{context}

Generate a natural, conversational response as the user. Follow these guidelines:
1. Respond naturally and in character based on your persona
2. Only reveal information that's directly relevant to the assistant's questions
3. Don't volunteer extra information unless it helps the conversation flow
4. Keep responses concise but natural (1-2 sentences typically)
5. If no specific information is requested, acknowledge and ask for clarification or next steps
6. Use your persona's tone and style consistently

Response should be just the spoken text, no quotes or extra formatting.
"""

        try:
            response = self.llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.7)

            # Clean up response
            response = response.strip()
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]

            # Update conversation history
            self.conversation_history.append(f"Assistant: {last_assistant}")
            self.conversation_history.append(f"User: {response}")

            return response

        except Exception as e:
            # Fallback on error
            print(f"LLM error in HumanDriver: {e}")
            return self._fallback_reply(last_assistant, requested_slots)

    def _fallback_reply(self, last_assistant: str, requested_slots: List[str]) -> str:
        """Fallback method using original logic."""
        parts = []
        provided = False
        for key in requested_slots:
            # Prefer facts and blueprint; avoid random for user_id
            if key in self.facts:
                val = self.facts[key]
            elif key == "user_id":
                val = self._arg_from_blueprint("user_id")
            elif key == "name":
                # derive name from user_id or rows
                uid = self.facts.get("user_id") or self._arg_from_blueprint("user_id")
                val = None
                if uid:
                    for u in (self.rows or {}).get("users", []) or []:
                        if str(u.get("user_id", "")) == uid and u.get("name"):
                            val = str(u.get("name"))
                            break
                if not val:
                    us = (self.rows or {}).get("users", []) or []
                    if us:
                        val = str(us[0].get("name", ""))
            else:
                val = self._arg_from_blueprint(key) or self._arg_from_rows(key)

        # Confirmation handling: if assistant asks to confirm proceeding
        lower_last = (last_assistant or "").lower()
        if any(k in lower_last for k in ["should i proceed", "go ahead", "confirm", "proceed with", "do you want me to", "are you sure"]):
            if not requested_slots:
                return "Yes, please proceed."
            if val is not None:
                provided = True
                if key.endswith("_id"):
                    parts.append(f"my {key} is {val}")
                elif key == "reason":
                    parts.append("I changed my mind")
                else:
                    parts.append(f"{key} is {val}")

        if not provided:
            parts.append("Could you proceed?")

        tone = (self.persona.get("tone") or "helpful").lower()
        response = "; ".join(parts)

        if tone == "polite":
            response = f"Sure — {response}"
        elif tone == "impatient":
            response = response  # Keep direct

        return response
