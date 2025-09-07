from __future__ import annotations
from typing import Any, Dict, List
import json

from apigen_mt.llm.lite import LLMClient


def to_tool_defs(tool_catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert YAML tool catalog entries to OpenAI tools format."""
    tools = []
    for t in tool_catalog:
        tools.append({
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description", ""),
                "parameters": t.get("schema", {"type": "object", "properties": {}}),
            },
        })
    return tools

class AssistantAgent:
    """A conversational agent that decides between talking to the user and calling a tool."""

    def __init__(self, tool_catalog: List[Dict[str, Any]], llm: LLMClient):
        self.tools = to_tool_defs(tool_catalog)
        self.llm = llm

    def get_next_turn(self, messages: List[Dict[str, Any]], goal: str) -> Dict[str, Any]:
        """Makes a conversational decision about the next action."""

        system_prompt = f'''You are a helpful and professional assistant.

Your goal is to assist the user with their request: "{goal}"

Follow these guidelines:
1.  **Be Conversational**: Do not just execute a plan. Engage the user in a natural, step-by-step dialogue.
2.  **Clarify and Confirm**: Ask for any missing information needed for a tool. Before executing a state-changing action (e.g., booking, cancelling, refunding), confirm with the user first. For example, say "I found the order for $55. Are you sure you want to proceed with the refund?"
3.  **Use Tools Incrementally**: Use your available tools one at a time to gather information or perform actions.
4.  **Think Step-by-Step**: Do not try to solve the entire problem at once. Take one logical step at a time, checking in with the user as needed.
5.  **Summarize**: When the goal is complete, provide a clear and friendly summary of what was done.
6.  **Verify Before Writing**: Prefer calling a read/lookup tool (like get_user_info or get_order) to validate IDs and context before any write/change tool (like update_address, cancel_order, refund_order).
7.  **One Tool At A Time**: Propose at most one tool call per turn. If input is missing, ask a short, specific question.
8.  **Respect Source of Truth**: When a write tool requires a user_id for an order (e.g., cancel_order), use the user_id from the latest get_order result for that order, even if the user text provided something else.
9.  **Elicit Missing Fields First**: For address updates, ask for the user's name (or ID) and the new address explicitly before any tool call; for cancellations/refunds, ask for order ID or enough info to look it up. Only call write tools after clear confirmation from the user.
'''
        
        llm_messages = [{"role": "system", "content": system_prompt}] + messages

        if self.llm.mock:
            # Simple mock logic for offline testing
            return {"type": "message", "content": "This is a mock response. In a real scenario, I would ask a question or call a tool."}

        # Delegate to LLM client for tool-aware calls with timeouts + fallback
        res = self.llm.chat_tools(llm_messages, tools=self.tools, tool_choice="auto", temperature=0.4)
        return res
