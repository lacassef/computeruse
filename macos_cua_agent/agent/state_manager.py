from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ActionResult:
    success: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    image_b64: str
    timestamp: float
    changed_since_last: bool = False
    note: str = ""


class StateManager:
    """Tracks loop state, history, and termination criteria."""

    def __init__(
        self,
        max_steps: int = 50,
        max_failures: int = 5,
        max_wall_clock_seconds: Optional[int] = None,
    ) -> None:
        self.max_steps = max_steps
        self.max_failures = max_failures
        self.max_wall_clock_seconds = max_wall_clock_seconds

        self.history: List[str] = []
        self.actions: List[Dict[str, Any]] = []
        self.observations: List[Observation] = []
        self.failure_count = 0
        self.steps = 0
        self.started_at = time.time()

    def record_observation(self, image_b64: str, changed: bool, note: str = "") -> Observation:
        obs = Observation(image_b64=image_b64, timestamp=time.time(), changed_since_last=changed, note=note)
        self.observations.append(obs)
        self.history.append(f"observation@{obs.timestamp}")
        return obs

    def record_action(self, action: Dict[str, Any], result: ActionResult) -> None:
        self.actions.append(action)
        self.history.append(f"action:{action.get('type', 'unknown')}:{result.success}")
        self.steps += 1
        if not result.success:
            self.failure_count += 1

    def should_halt(self) -> bool:
        if self.max_steps and self.steps >= self.max_steps:
            return True
        if self.max_failures and self.failure_count >= self.max_failures:
            return True
        if self.max_wall_clock_seconds and (time.time() - self.started_at) >= self.max_wall_clock_seconds:
            return True
        return False

    def summary(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "failures": self.failure_count,
            "history": list(self.history),
            "actions": list(self.actions),
            "observations": len(self.observations),
            "runtime_seconds": time.time() - self.started_at,
        }

