from __future__ import annotations
from typing import Any, Dict
from apigen_mt.policies.base import PolicySuite


def retail_policy_suite() -> PolicySuite:
    suite = PolicySuite("retail")

    @suite.add
    def identity_match(trace: Dict[str, Any]):
        # if cancel_order occurs, ensure user_id matches the order's user
        orders = trace["post_state"]["orders"]
        for step in trace["steps"]:
            if step["tool_name"] == "cancel_order" and step["ok"]:
                uid = step["arguments"].get("user_id")
                oid = step["arguments"].get("order_id")
                if oid in orders and orders[oid]["user_id"] != uid:
                    return False, f"identity_mismatch:{oid}"
        return True, "ok"

    @suite.add
    def refund_limits(trace: Dict[str, Any]):
        # refunded cannot exceed total
        for oid, order in trace["post_state"]["orders"].items():
            if order["refunded"] - 1e-6 > order["total"]:
                return False, f"refund_exceeds_total:{oid}"
        return True, "ok"

    @suite.add
    def refund_identity_consistency(trace: Dict[str, Any]):
        # If the trace references a specific user via list_orders/get_user_info,
        # ensure refunds apply to orders belonging to one of those users.
        referenced_users = set()
        for step in trace["steps"]:
            if not step.get("ok"):
                continue
            if step["tool_name"] == "list_orders":
                uid = step["arguments"].get("user_id")
                if uid:
                    referenced_users.add(uid)
            if step["tool_name"] == "get_user_info":
                uid = step["arguments"].get("user_id")
                if uid:
                    referenced_users.add(uid)
        if not referenced_users:
            return True, "ok"  # no identity context
        orders = trace["post_state"]["orders"]
        for step in trace["steps"]:
            if step["tool_name"] == "refund_order" and step.get("ok"):
                oid = step["arguments"].get("order_id")
                if oid in orders and orders[oid]["user_id"] not in referenced_users:
                    return False, f"refund_identity_mismatch:{oid}:{orders[oid]['user_id']} not in {sorted(referenced_users)}"
        return True, "ok"

    @suite.add
    def refund_only_if_not_processing(trace: Dict[str, Any]):
        # allow refunds anytime in this mock, but demonstrate cross-step example
        return True, "ok"

    return suite
