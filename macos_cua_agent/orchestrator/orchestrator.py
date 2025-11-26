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
from macos_cua_agent.orchestrator.planning import Plan, Step
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
        self.policy_engine = PolicyEngine(str(policy_path), settings)
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
        hint_count = 0
        max_hints = 3
        plan_revision_count = 0
        max_plan_revisions = 3
        global_hotkeys = {("cmd", "space"), ("command", "space"), ("cmd", "tab"), ("command", "tab")}

        try:
            while not state.should_halt():
                if plan and plan_revision_count < max_plan_revisions and self._should_replan(
                    plan, state, repeat_same_action, repeat_without_change
                ):
                    revised_plan = self.planner.revise_plan(plan, state.history, current_frame)
                    if revised_plan:
                        if revised_plan.to_dict() != plan.to_dict():
                            plan_revision_count += 1
                            self.logger.info("Plan revised (auto); new current step index %s", revised_plan.current_step_index)
                            state.history.append(
                                f"plan_revised:auto:step_index={revised_plan.current_step_index}"
                            )
                        plan = revised_plan
                        repeat_same_action = 0
                        repeat_without_change = 0
                        last_action_sig = None
                        repeat_info_for_model = None

                current_step = plan.current_step() if plan else None
                loop_state = self._format_loop_state(plan, state, repeat_same_action, repeat_without_change)
                action = self.cognitive_core.propose_action(
                    current_frame,
                    state.history,
                    user_prompt=user_prompt,
                    repeat_info=repeat_info_for_model,
                    plan=plan,
                    current_step=current_step,
                    loop_state=loop_state,
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
                if action.get("execution") == "shell" and result.metadata:
                    stdout = (result.metadata.get("stdout") or "").strip()
                    stderr = (result.metadata.get("stderr") or "").strip()
                    if stdout:
                        state.history.append(f"shell_stdout:{stdout[:500]}")
                    if stderr:
                        state.history.append(f"shell_stderr:{stderr[:500]}")

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
                else:
                    # UI changed; allow future hotkeys to be considered fresh.
                    hotkey_counts.clear()

                if plan and current_step:
                    step_completed = False
                    if self.reflector.available:
                        step_completed = self.reflector.is_step_complete(
                            current_step, state.history, next_frame, changed
                        )
                    elif not self.settings.strict_step_completion:
                        step_completed = self._heuristic_step_complete(current_step, action, result, changed)
                    if step_completed:
                        finished_id = current_step.id if current_step else None
                        plan.advance()
                        hotkey_counts.clear()
                        state.history.append(
                            f"plan_step_completed:{finished_id if finished_id is not None else 'unknown'}"
                        )
                        self.logger.info("Advanced plan to step index %s", plan.current_step_index)
                        if not plan.current_step():
                            self.logger.info("Plan completed; stopping loop.")
                            break

                action_sig = repr(action)
                is_wait = action.get("type") == "wait"
                pending_break = False
                break_reason = ""
                if not is_wait:
                    if action_sig == last_action_sig:
                        repeat_same_action += 1
                        if repeat_same_action >= 5:
                            pending_break = True
                            break_reason = f"repeat_same_action:{repeat_same_action}"
                    else:
                        repeat_same_action = 0
                    if not changed and action_sig == last_action_sig:
                        repeat_without_change += 1
                        if repeat_without_change >= 3:
                            pending_break = True
                            break_reason = break_reason or "repeat_without_change"
                    else:
                        repeat_without_change = 0
                else:
                    repeat_same_action = 0
                    repeat_without_change = 0

                if action.get("type") == "key":
                    combo = tuple(sorted([k.lower() for k in action.get("keys") or []]))
                    if combo in global_hotkeys and not changed:
                        state.history.append(f"global_hotkey_no_effect:{'+'.join(combo)}")
                        repeat_info_for_model = {
                            "count": repeat_same_action,
                            "action": action_sig,
                            "hint": "Global hotkey had no visible effect; prefer clicking the visible app or window.",
                        }

                if pending_break:
                    if plan and current_step:
                        plan.fail_current(break_reason or "stuck")
                        state.history.append(f"plan_step_failed:{current_step.id}:{break_reason or 'stuck'}")
                    state.record_stuck(break_reason or "stuck")

                    hint = ""
                    plan_changed = False
                    if self.reflector.available and hint_count < max_hints:
                        hint = self.reflector.suggest_hint(plan.current_step() if plan else None, state.history, next_frame)
                        if hint:
                            hint_count += 1
                            state.history.append(f"reflector_hint:{hint}")
                            repeat_info_for_model = {
                                "count": repeat_same_action,
                                "action": action_sig,
                                "hint": hint,
                            }
                            self.logger.info("Injected reflector hint to unblock: %s", hint)
                            pending_break = False

                    if plan and plan_revision_count < max_plan_revisions:
                        revised_plan = self.planner.revise_plan(plan, state.history, next_frame)
                        if revised_plan.to_dict() != plan.to_dict():
                            plan_revision_count += 1
                            state.history.append(
                                f"plan_revised:stuck:step_index={revised_plan.current_step_index}"
                            )
                            self.logger.info(
                                "Plan revised after stuck; new current step index %s", revised_plan.current_step_index
                            )
                            plan_changed = True
                        plan = revised_plan
                        if plan_changed:
                            pending_break = False

                    repeat_same_action = 0
                    repeat_without_change = 0
                    last_action_sig = None

                    if pending_break:
                        self.logger.info("Breaking loop: %s", break_reason or "stuck")
                        break

                    repeat_info_for_model = repeat_info_for_model or {"count": 0, "action": action_sig}
                    current_frame = next_frame
                    continue

                last_action_sig = action_sig
                if not repeat_info_for_model or "hint" not in repeat_info_for_model:
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

    def _should_replan(
        self, plan: Optional[Plan], state: StateManager, repeat_same_action: int, repeat_without_change: int
    ) -> bool:
        if not plan or not plan.current_step():
            return False
        if repeat_same_action >= 3 or repeat_without_change >= 2:
            return True
        if state.failure_count >= 3:
            return True
        if plan.current_step().status == "failed":
            return True
        return False

    def _format_loop_state(
        self, plan: Optional[Plan], state: StateManager, repeat_same_action: int, repeat_without_change: int
    ) -> dict:
        current_step = plan.current_step() if plan else None
        return {
            "current_step_id": current_step.id if current_step else None,
            "current_step_status": current_step.status if current_step else None,
            "failure_count": state.failure_count,
            "steps_taken": state.steps,
            "repeat_same_action": repeat_same_action,
            "repeat_without_change": repeat_without_change,
        }

    def _heuristic_step_complete(self, step: Step, action: dict, result: ActionResult, changed: bool) -> bool:
        """Conservative fallback when reflection is disabled."""
        if not result.success or not changed:
            return False
        if action.get("type") in {"wait", "noop", "capture_only"}:
            return False
        if step.status == "failed":
            return False
        # Require a direct UI interaction before auto-completing.
        if action.get("type") in {"left_click", "double_click", "right_click", "type", "scroll", "key", "mouse_move"}:
            return True
        return False

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
