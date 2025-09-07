from __future__ import annotations
from typing import Any, Dict, List


class PolicyResult:
    def __init__(self, passed: bool, failures: List[str] | None = None):
        self.passed = passed
        self.failures = failures or []


class PolicySuite:
    def __init__(self, name: str):
        self.name = name
        self._checks = []

    def add(self, fn):
        self._checks.append(fn)
        return fn

    def run(self, trace: Dict[str, Any]) -> PolicyResult:
        failures: List[str] = []
        for check in self._checks:
            ok, msg = check(trace)
            if not ok:
                failures.append(msg)
        return PolicyResult(passed=(len(failures) == 0), failures=failures)

