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


def _read_prompt() -> str | None:
    try:
        return input("\nEnter a prompt (blank to quit): ").strip() or None
    except (EOFError, KeyboardInterrupt):
        return None


def _run_session(
    user_prompt: str,
    settings: Settings,
    vision: VisionPipeline,
    action_engine: ActionEngine,
    cognitive_core: CognitiveCore,
    logger,
) -> None:
    state = StateManager(
        max_steps=settings.max_steps,
        max_failures=settings.max_failures,
        max_wall_clock_seconds=settings.max_wall_clock_seconds,
    )

    # Seed history with the user's instruction so the model sees recent context.
    state.history.append(f"user_prompt:{user_prompt}")

    current_frame = vision.capture_base64()
    state.record_observation(current_frame, changed=True, note=f"initial capture for: {user_prompt}")

    last_action_sig: str | None = None
    repeat_without_change = 0
    repeat_same_action = 0
    repeat_info_for_model: dict | None = None

    try:
        while not state.should_halt():
            action = cognitive_core.propose_action(
                current_frame, state.history, user_prompt=user_prompt, repeat_info=repeat_info_for_model
            )

            # Noop ends the loop gracefully.
            if action.get("type") == "noop":
                logger.info("Noop action requested; stopping loop. Reason: %s", action.get("reason"))
                break

            result = action_engine.execute(action)
            state.record_action(action, result)

            # Allow UI to settle; certain hotkeys (e.g., Spotlight) need a longer pause.
            base_delay = settings.verify_delay_ms / 1000
            extra_delay = 0.0
            if action.get("type") == "key":
                keys = [k.lower() for k in action.get("keys") or []]
                if "space" in keys and ("cmd" in keys or "command" in keys):
                    extra_delay = 0.8  # Spotlight animation/stabilization
            time.sleep(base_delay + extra_delay)
            next_frame = vision.capture_base64()
            changed = vision.has_changed(current_frame, next_frame)
            state.record_observation(next_frame, changed)

            if not changed:
                logger.info("No UI change detected after action: %s", action)

            action_sig = repr(action)
            if action_sig == last_action_sig:
                repeat_same_action += 1
                if repeat_same_action >= 5:
                    logger.info("Breaking loop: repeated identical action %s times.", repeat_same_action)
                    break
            else:
                repeat_same_action = 0
            if not changed and action_sig == last_action_sig:
                repeat_without_change += 1
                if repeat_without_change >= 3:
                    logger.info("Breaking loop: repeated identical action without UI change.")
                    break
            else:
                repeat_without_change = 0
            last_action_sig = action_sig
            repeat_info_for_model = {"count": repeat_same_action, "action": last_action_sig}

            current_frame = next_frame
    except KeyboardInterrupt:
        logger.info("Session cancelled by user.")

    logger.info("Session finished: %s", state.summary())


def run() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__, level=settings.log_level)
    if not settings.enable_hid:
        logger.warning("ENABLE_HID is false; actions will run in dry-run mode (no real input).")

    policy_path = Path(__file__).resolve().parent / "policies" / "safety_rules.yaml"

    vision = VisionPipeline(settings)
    policy_engine = PolicyEngine(str(policy_path))
    action_engine = ActionEngine(settings, policy_engine)
    cognitive_core = CognitiveCore(settings)

    while True:
        user_prompt = _read_prompt()
        if not user_prompt:
            logger.info("No prompt provided; exiting.")
            return
        logger.info("Starting session for prompt: %s", user_prompt)
        _run_session(
            user_prompt=user_prompt,
            settings=settings,
            vision=vision,
            action_engine=action_engine,
            cognitive_core=cognitive_core,
            logger=logger,
        )


if __name__ == "__main__":
    run()
