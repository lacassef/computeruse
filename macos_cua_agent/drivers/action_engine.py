from __future__ import annotations

import time

from macos_cua_agent.agent.state_manager import ActionResult
from macos_cua_agent.drivers.hid_driver import HIDDriver
from macos_cua_agent.drivers.semantic_driver import SemanticDriver
from macos_cua_agent.drivers.shell_driver import ShellDriver
from macos_cua_agent.policies.policy_engine import PolicyEngine
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger


class ActionEngine:
    """Routes actions through policy evaluation, semantic path, or HID injection."""

    def __init__(self, settings: Settings, policy_engine: PolicyEngine) -> None:
        self.settings = settings
        self.policy_engine = policy_engine
        self.hid_driver = HIDDriver(settings)
        self.semantic_driver = SemanticDriver(settings)
        self.shell_driver = ShellDriver(settings)
        self.logger = get_logger(__name__, level=settings.log_level)

    def execute(self, action: dict) -> ActionResult:
        decision = self.policy_engine.evaluate(action)
        if not decision.allowed:
            self.logger.warning("Action blocked by policy: %s", decision.reason)
            return ActionResult(success=False, reason=decision.reason)
        if decision.hitl_required:
            self.logger.warning("Action requires human confirmation: %s", action)
            return ActionResult(success=False, reason="human confirmation required")

        # Special-cased actions for loop control.
        if action.get("type") in ("noop", "capture_only"):
            return ActionResult(success=True, reason=action.get("reason", "noop"))
        if action.get("type") == "wait":
            seconds = float(action.get("seconds", 1))
            time.sleep(seconds)
            return ActionResult(success=True, reason=f"waited {seconds} seconds")

        self.logger.info("Executing action via %s: %s", action.get("execution", "hid"), action)
        execution_path = action.get("execution", "hid")
        if execution_path == "semantic" and self.settings.enable_semantic:
            result = self.semantic_driver.execute(action)
        elif execution_path == "shell":
            result = self.shell_driver.execute(action)
        else:
            result = self._execute_hid(action)

        self.logger.info("Action result: success=%s reason=%s", result.success, result.reason)
        return result

    def _execute_hid(self, action: dict) -> ActionResult:
        action_type = action.get("type")
        x = action.get("x")
        y = action.get("y")

        if action_type == "mouse_move" and x is not None and y is not None:
            return self.hid_driver.move(x, y)
        if action_type == "left_click" and x is not None and y is not None:
            return self.hid_driver.left_click(x, y)
        if action_type == "right_click" and x is not None and y is not None:
            return self.hid_driver.right_click(x, y)
        if action_type == "double_click" and x is not None and y is not None:
            return self.hid_driver.double_click(x, y)
        if action_type == "scroll":
            clicks = action.get("clicks", 0)
            return self.hid_driver.scroll(int(clicks))
        if action_type == "type":
            return self.hid_driver.type_text(action.get("text", ""))
        if action_type == "key":
            keys = action.get("keys") or []
            return self.hid_driver.press_keys(keys)

        self.logger.warning("Unknown HID action: %s", action)
        return ActionResult(success=False, reason="unknown action")
