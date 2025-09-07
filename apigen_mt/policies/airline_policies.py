from __future__ import annotations
from typing import Any, Dict
from apigen_mt.policies.base import PolicySuite


def airline_policy_suite() -> PolicySuite:
    suite = PolicySuite("airline")

    @suite.add
    def identity_match(trace: Dict[str, Any]):
        resv = trace["post_state"]["reservations"]
        for step in trace["steps"]:
            if step["tool_name"] in ("change_seat", "cancel_reservation") and step["ok"]:
                rid = step["arguments"].get("reservation_id")
                pid = step["arguments"].get("passenger_id")
                if rid in resv and resv[rid]["passenger_id"] != pid:
                    return False, f"identity_mismatch:{rid}"
        return True, "ok"

    @suite.add
    def refund_limits(trace: Dict[str, Any]):
        for rid, r in trace["post_state"]["reservations"].items():
            if r["refunded"] - 1e-6 > r["fare"]:
                return False, f"refund_exceeds_fare:{rid}"
        return True, "ok"

    @suite.add
    def no_cancel_after_flown(trace: Dict[str, Any]):
        for rid, r in trace["post_state"]["reservations"].items():
            if r["status"] == "flown":
                # ensure no cancel_reservation step attempted successfully
                for step in trace["steps"]:
                    if step["tool_name"] == "cancel_reservation" and step["ok"] and step["arguments"].get("reservation_id") == rid:
                        return False, f"cancel_after_flown:{rid}"
        return True, "ok"

    return suite

