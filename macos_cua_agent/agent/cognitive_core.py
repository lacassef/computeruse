from __future__ import annotations

from typing import Any, Dict, List, Optional

from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger


class CognitiveCore:
    """Minimal cognitive core that can call Anthropic or fall back to stubbed actions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.client = self._build_client()

    def _build_client(self) -> Optional[Any]:
        if not self.settings.use_anthropic:
            self.logger.info("Anthropic disabled; running in deterministic stub mode.")
            return None
        try:
            import anthropic  # type: ignore
        except ImportError:
            self.logger.warning("anthropic package not installed; falling back to stubbed actions.")
            return None

        if not self.settings.anthropic_api_key:
            self.logger.warning("ANTHROPIC_API_KEY missing; falling back to stubbed actions.")
            return None

        return anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

    def propose_action(self, observation_b64: str, history: List[str]) -> Dict[str, Any]:
        """Return the next action as a dict with at least a `type` field."""
        if not self.client:
            # Deterministic stub: request a screenshot refresh after first loop.
            action_type = "noop" if history else "capture_only"
            return {
                "type": action_type,
                "reason": "Cognitive core running without Anthropic client.",
            }

        try:
            response = self._call_anthropic(observation_b64, history)
            parsed_action = self._parse_response(response)
            if parsed_action:
                return parsed_action
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.exception("Anthropic call failed; falling back to noop.", exc_info=exc)

        return {"type": "noop", "reason": "Failed to generate action"}

    def _call_anthropic(self, observation_b64: str, history: List[str]) -> Any:
        """Send a computer-use request to Anthropic's API."""
        prompt = self._build_prompt(history)
        content = [
            {"type": "text", "text": prompt},
            {"type": "input_image", "input_image": observation_b64},
        ]

        # The beta header and tool type are centralized in settings to manage schema drift.
        response = self.client.beta.tools.messages.create(  # type: ignore[attr-defined]
            model=self.settings.anthropic_model,
            max_tokens=512,
            beta=[self.settings.beta_header],
            tools=[{"type": self.settings.tool_type}],
            messages=[{"role": "user", "content": content}],
        )
        return response

    def _build_prompt(self, history: List[str]) -> str:
        return (
            "You are controlling a macOS desktop. "
            "Propose the next concrete UI action using the computer-use tool only. "
            f"Recent events: {history[-10:]}"
        )

    def _parse_response(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract the first tool use block from the response."""
        content = getattr(response, "content", None)
        if not content:
            return None

        for block in content:
            if getattr(block, "type", "") == "tool_use":
                try:
                    # The block input should already be a structured dict.
                    parsed = dict(block.input)  # type: ignore[attr-defined]
                    parsed.setdefault("type", parsed.get("action", "noop"))
                    return parsed
                except Exception:
                    continue
        return None
