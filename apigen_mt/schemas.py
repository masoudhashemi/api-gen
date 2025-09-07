from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Action(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class PolicyChecks(BaseModel):
    passed: bool
    failures: List[str] = Field(default_factory=list)


class ReviewScores(BaseModel):
    correctness: int = 0
    completeness: int = 0
    satisfaction: int = 0
    creativity: int = 0


class Metadata(BaseModel):
    persona_id: str
    domain: str
    policy_checks: PolicyChecks
    review_scores: ReviewScores
    recombined_from: Optional[List[str]] = None


class TaskConfig(BaseModel):
    instruction: str
    actions: List[Action]
    outputs: List[str]
    diff_patch: Dict[str, Any] = Field(default_factory=dict)
    metadata: Metadata


class ExecStep(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    ok: bool
    result: Any = None
    error: Optional[str] = None


class ExecTrace(BaseModel):
    steps: List[ExecStep] = Field(default_factory=list)
    pre_state: Dict[str, Any] = Field(default_factory=dict)
    post_state: Dict[str, Any] = Field(default_factory=dict)
    diff_patch: Dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    role: str  # "user" | "assistant" | "tool"
    content: Any = None
    tool_call: Optional[Dict[str, Any]] = None


class TrajectoryEval(BaseModel):
    state_match: bool
    output_match: bool


class Trajectory(BaseModel):
    blueprint_id: str
    turns: List[Turn]
    final_response: str
    eval: TrajectoryEval

