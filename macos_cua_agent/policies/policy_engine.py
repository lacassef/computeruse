from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger

DEFAULT_RULES = {
    "blocked_actions": ["shell_command"],
    "blocked_bundle_ids": ["com.apple.keychainaccess"],
    "hitl_actions": ["erase_disk", "format_disk"],
    "allowed_shell_basenames": [],
    "blocked_shell_basenames": [],
}


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""
    hitl_required: bool = False


class PolicyEngine:
    """Evaluates proposed actions against configured safety rules."""

    def __init__(self, rules_path: str, settings: Settings | None = None) -> None:
        self.logger = get_logger(__name__)
        self.rules = self._load_rules(rules_path)
        self._apply_overrides_from_settings(settings)

    def _load_rules(self, rules_path: str) -> Dict[str, Any]:
        if not os.path.exists(rules_path):
            self.logger.info("safety_rules.yaml missing; using defaults.")
            return dict(DEFAULT_RULES)
        try:
            import yaml  # type: ignore
        except Exception as exc:
            self.logger.warning("PyYAML unavailable (%s); using defaults.", exc)
            return dict(DEFAULT_RULES)

        with open(rules_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        merged = dict(DEFAULT_RULES)
        merged.update(loaded)
        return merged

    def _apply_overrides_from_settings(self, settings: Settings | None) -> None:
        if not settings:
            return

        allow_env = settings.shell_allowed_commands
        if allow_env:
            allowlist: List[str] = [cmd.strip() for cmd in allow_env.split(",") if cmd.strip()]
            if allowlist:
                self.rules["allowed_shell_basenames"] = allowlist
                self.logger.info("Applied shell allowlist from env: %s", allowlist)

    def evaluate(self, action: Dict[str, Any]) -> PolicyDecision:
        action_type = action.get("type") or action.get("action")
        bundle_id = action.get("bundle_id") or action.get("app")

        if action_type in self.rules.get("blocked_actions", []):
            return PolicyDecision(allowed=False, reason=f"action blocked: {action_type}")
        if action_type == "sandbox_shell":
            cmd_raw = action.get("cmd") or action.get("command") or ""
            basename = ""
            if isinstance(cmd_raw, str):
                parts = cmd_raw.strip().split()
                basename = parts[0] if parts else ""
            elif isinstance(cmd_raw, (list, tuple)):
                basename = str(cmd_raw[0]) if cmd_raw else ""

            if basename in self.rules.get("blocked_shell_basenames", []):
                return PolicyDecision(allowed=False, reason=f"shell command blocked: {basename}")

            allowed = self.rules.get("allowed_shell_basenames")
            if allowed and basename not in allowed:
                return PolicyDecision(allowed=False, reason=f"shell command not allowlisted: {basename}")
        if bundle_id and bundle_id in self.rules.get("blocked_bundle_ids", []):
            return PolicyDecision(allowed=False, reason=f"bundle blocked: {bundle_id}")
        if action_type in self.rules.get("hitl_actions", []):
            return PolicyDecision(allowed=True, hitl_required=True, reason="human confirmation required")
        return PolicyDecision(allowed=True)
