from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from macos_cua_agent.agent.cognitive_core import CognitiveCore
from macos_cua_agent.agent.state_manager import ActionResult, StateManager
from macos_cua_agent.drivers.action_engine import ActionEngine
from macos_cua_agent.drivers.vision_pipeline import VisionPipeline
from macos_cua_agent.memory.memory_manager import Episode, MemoryManager
from macos_cua_agent.orchestrator.planner_client import PlannerClient
from macos_cua_agent.orchestrator.planning import Plan
from macos_cua_agent.orchestrator.reflection import Reflector
from macos_cua_agent.policies.policy_engine import PolicyEngine
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger


class Orchestrator:
    """Coordinates planning, execution, and memory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        policy_path = Path(__file__).resolve().parent.parent / "policies" / "safety_rules.yaml"

        self.vision = VisionPipeline(settings)
        self.policy_engine = PolicyEngine(str(policy_path))
        self.action_engine = ActionEngine(settings, self.policy_engine)
        self.cognitive_core = CognitiveCore(settings)
        self.memory = MemoryManager(settings)
        self.planner = PlannerClient(settings)
        self.reflector = Reflector(settings)

        if not settings.enable_hid:
            self.logger.warning("ENABLE_HID is false; actions will run in dry-run mode (no real input).")

    def run_task(self, user_prompt: str) -> dict:
        prior_episodes = self.memory.list_episodes()
        prior_semantic = self.memory.search_semantic(user_prompt, top_k=5)
        plan = self.planner.make_plan(user_prompt, prior_episodes, prior_semantic)
        self.logger.info("Plan %s created with %s steps", plan.id, len(plan.steps))
        summary = self._run_session(user_prompt=user_prompt, plan=plan)
        return summary

    def _run_session(self, user_prompt: str, plan: Optional[Plan] = None) -> dict:
        state = StateManager(
            max_steps=self.settings.max_steps,
            max_failures=self.settings.max_failures,
            max_wall_clock_seconds=self.settings.max_wall_clock_seconds,
        )

        if plan:
            serialized_plan = "; ".join([f"{s.id}:{s.description}({s.status})" for s in plan.steps])
            state.history.append(f"plan_init:{serialized_plan}")

        state.history.append(f"user_prompt:{user_prompt}")

        current_frame = self.vision.capture_base64()
        state.record_observation(current_frame, changed=True, note=f"initial capture for: {user_prompt}")

        last_action_sig: str | None = None
        repeat_without_change = 0
        repeat_same_action = 0
        repeat_info_for_model: dict | None = None
        hotkey_counts: dict[tuple[str, ...], int] = {}
        hint_injected = False

        try:
            while not state.should_halt():
                action = self.cognitive_core.propose_action(
                    current_frame,
                    state.history,
                    user_prompt=user_prompt,
                    repeat_info=repeat_info_for_model,
                    plan=plan,
                    current_step=plan.current_step() if plan else None,
                )

                if action.get("type") == "noop":
                    self.logger.info("Noop action requested; stopping loop. Reason: %s", action.get("reason"))
                    break

                if action.get("type") == "key":
                    combo = tuple(sorted([k.lower() for k in action.get("keys") or []]))
                    count = hotkey_counts.get(combo, 0)
                    if count >= 2:
                        self.logger.info("Skipping hotkey %s; already executed %s times", "+".join(combo), count)
                        result = ActionResult(success=False, reason="hotkey deduped")
                        state.record_action(action, result)
                        repeat_info_for_model = {"count": repeat_same_action, "action": repr(action)}
                        continue
                    hotkey_counts[combo] = count + 1

                result = self.action_engine.execute(action)
                state.record_action(action, result)

                base_delay = self.settings.verify_delay_ms / 1000
                extra_delay = 0.0
                if action.get("type") == "key":
                    keys = [k.lower() for k in action.get("keys") or []]
                    if "space" in keys and ("cmd" in keys or "command" in keys):
                        extra_delay = 0.8
                time.sleep(base_delay + extra_delay)
                next_frame = self.vision.capture_base64()
                changed = self.vision.has_changed(current_frame, next_frame)
                state.record_observation(next_frame, changed)

                if not changed:
                    self.logger.info("No UI change detected after action: %s", action)

                if plan and plan.current_step():
                    step_completed = False
                    if self.reflector.available:
                        step_completed = self.reflector.is_step_complete(
                            plan.current_step(), state.history, next_frame, changed
                        )
                    elif result.success and changed:
                        step_completed = True
                    if step_completed:
                        finished_id = plan.current_step().id if plan.current_step() else None
                        prev_step = plan.current_step()
                        plan.advance()
                        state.history.append(
                            f"plan_step_completed:{finished_id if finished_id is not None else 'unknown'}"
                        )
                        self.logger.info("Advanced plan to step index %s", plan.current_step_index)
                        if not plan.current_step():
                            self.logger.info("Plan completed; stopping loop.")
                            break

                action_sig = repr(action)
                pending_break = False
                if action_sig == last_action_sig:
                    repeat_same_action += 1
                    if repeat_same_action >= 5:
                        pending_break = True
                else:
                    repeat_same_action = 0
                if not changed and action_sig == last_action_sig:
                    repeat_without_change += 1
                    if repeat_without_change >= 3:
                        pending_break = True
                else:
                    repeat_without_change = 0

                if pending_break and self.reflector.available and not hint_injected:
                    hint = self.reflector.suggest_hint(plan.current_step() if plan else None, state.history, next_frame)
                    if hint:
                        state.history.append(f"reflector_hint:{hint}")
                        repeat_info_for_model = {
                            "count": repeat_same_action,
                            "action": action_sig,
                            "hint": hint,
                        }
                        self.logger.info("Injected reflector hint to unblock: %s", hint)
                        hint_injected = True
                        pending_break = False

                if pending_break:
                    if repeat_same_action >= 5:
                        reason = f"repeat_same_action:{repeat_same_action}"
                        state.record_stuck(reason)
                        self.logger.info("Breaking loop: repeated identical action %s times.", repeat_same_action)
                    else:
                        reason = "repeat_without_change"
                        state.record_stuck(reason)
                        self.logger.info("Breaking loop: repeated identical action without UI change.")
                    break
                last_action_sig = action_sig
                repeat_info_for_model = {"count": repeat_same_action, "action": last_action_sig}

                current_frame = next_frame
        except KeyboardInterrupt:
            self.logger.info("Session cancelled by user.")

        summary = state.summary()
        if plan:
            summary["plan"] = plan.to_dict()
        self.logger.info("Session finished: %s", summary)
        self._persist_episode(user_prompt, state, plan)
        return summary

    def _persist_episode(self, user_prompt: str, state: StateManager, plan: Optional[Plan]) -> None:
        episode_id = plan.id if plan else f"session-{int(state.started_at)}"
        outcome = "success"
        if state.failure_count > 0:
            outcome = "mixed"
        if plan and plan.current_step():
            outcome = "incomplete"
        summary = self.planner.summarize_episode(user_prompt, state.history, plan)

        log_path = self.memory.logs_dir / f"{episode_id}.log"
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                for line in state.history:
                    handle.write(f"{line}\n")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to write episode log: %s", exc)
            log_path = None

        episode = Episode(
            id=episode_id,
            created_at=state.started_at,
            user_prompt=user_prompt,
            plan=plan.to_dict() if plan else {},
            outcome=outcome,
            summary=summary,
            tags=["desktop", "cua"],
            raw_log_path=str(log_path) if log_path else None,
        )
        try:
            self.memory.save_episode(episode)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to persist episode: %s", exc)
