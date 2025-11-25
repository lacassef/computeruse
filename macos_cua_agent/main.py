from __future__ import annotations

import time
from pathlib import Path

from macos_cua_agent.agent.cognitive_core import CognitiveCore
from macos_cua_agent.agent.state_manager import StateManager
from macos_cua_agent.drivers.action_engine import ActionEngine
from macos_cua_agent.drivers.vision_pipeline import VisionPipeline
from macos_cua_agent.policies.policy_engine import PolicyEngine
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import configure_logging, get_logger


def run() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__, level=settings.log_level)

    policy_path = Path(__file__).resolve().parent / "policies" / "safety_rules.yaml"

    vision = VisionPipeline(settings)
    policy_engine = PolicyEngine(str(policy_path))
    action_engine = ActionEngine(settings, policy_engine)
    cognitive_core = CognitiveCore(settings)
    state = StateManager(
        max_steps=settings.max_steps,
        max_failures=settings.max_failures,
        max_wall_clock_seconds=settings.max_wall_clock_seconds,
    )

    current_frame = vision.capture_base64()
    state.record_observation(current_frame, changed=True, note="initial capture")

    while not state.should_halt():
        action = cognitive_core.propose_action(current_frame, state.history)

        # Noop ends the loop gracefully.
        if action.get("type") == "noop":
            logger.info("Noop action requested; stopping loop.")
            break

        result = action_engine.execute(action)
        state.record_action(action, result)

        time.sleep(settings.verify_delay_ms / 1000)
        next_frame = vision.capture_base64()
        changed = vision.has_changed(current_frame, next_frame)
        state.record_observation(next_frame, changed)

        if not changed:
            logger.info("No UI change detected after action: %s", action)

        current_frame = next_frame

    logger.info("Session finished: %s", state.summary())


if __name__ == "__main__":
    run()

