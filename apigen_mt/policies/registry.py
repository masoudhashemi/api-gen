from __future__ import annotations


def policy_suite_for(domain: str):
    domain = domain.lower()
    if domain == "retail":
        from apigen_mt.policies.retail_policies import retail_policy_suite
        return retail_policy_suite()
    if domain == "airline":
        from apigen_mt.policies.airline_policies import airline_policy_suite
        return airline_policy_suite()
    return None

