from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger
from macos_cua_agent.utils.macos_integration import get_display_info

# OpenRouter exposes an OpenAI-compatible tool-calling API. We define our own
# computer tool schema so Claude Opus 4.5 can drive local actions.
COMPUTER_TOOL = {
    "type": "function",
    "function": {
        "name": "computer",
        "description": (
            "Control the macOS desktop: move and click the mouse, type, press hotkeys, "
            "scroll, wait, or request a screenshot."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "move_mouse",
                        "left_click",
                        "right_click",
                        "double_click",
                        "scroll",
                        "type",
                        "hotkey",
                        "wait",
                        "screenshot",
                    ],
                },
                "x": {"type": "number", "description": "X coordinate in logical pixels."},
                "y": {"type": "number", "description": "Y coordinate in logical pixels."},
                "scroll_y": {
                    "type": "number",
                    "description": "Vertical scroll amount (positive up, negative down).",
                },
                "text": {"type": "string", "description": "Text to type."},
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hotkey combo, e.g. ['ctrl','s'].",
                },
                "seconds": {
                    "type": "number",
                    "description": "Seconds to wait for the 'wait' action.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


class CognitiveCore:
    """Calls Claude Opus 4.5 via OpenRouter with a custom computer tool."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.display = get_display_info()
        self.client = self._build_client()

    def _build_client(self) -> Optional[Any]:
        if not self.settings.use_openrouter:
            self.logger.info("OpenRouter disabled; running in deterministic stub mode.")
            return None
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            self.logger.warning("openai package not installed; falling back to stubbed actions.")
            return None

        if not self.settings.openrouter_api_key:
            self.logger.warning("OPENROUTER_API_KEY missing; running stub mode.")
            return None

        return OpenAI(base_url=self.settings.openrouter_base_url, api_key=self.settings.openrouter_api_key)

    def propose_action(
        self,
        observation_b64: str,
        history: List[str],
        user_prompt: Optional[str] = None,
        repeat_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the next action as a dict with at least a `type` field."""
        if not self.client:
            action_type = "noop" if history else "capture_only"
            return {
                "type": action_type,
                "reason": "Cognitive core running without OpenRouter client.",
            }

        try:
            response = self._call_openrouter(
                observation_b64, history, user_prompt=user_prompt, repeat_info=repeat_info
            )
            parsed_action = self._parse_tool_call(response)
            if parsed_action:
                return parsed_action
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.exception("OpenRouter call failed; falling back to noop.", exc_info=exc)

        return {"type": "noop", "reason": "Failed to generate action"}

    def _call_openrouter(
        self,
        observation_b64: str,
        history: List[str],
        user_prompt: Optional[str],
        repeat_info: Optional[Dict[str, Any]],
    ) -> Any:
        """Send a vision + tool-calling request to OpenRouter."""
        system_prompt = (
            "You can control the user's macOS desktop using the `computer` tool. "
            "Be brief, avoid destructive actions, and prefer precise coordinates. "
            "Avoid repeating the same hotkey if the UI does not change; if stuck, return noop. "
            f"Display size: {self.display.logical_width}x{self.display.logical_height} logical px. "
            f"Recent events: {history[-10:]}"
        )
        if repeat_info and repeat_info.get("count", 0) >= 2:
            system_prompt += (
                f" Warning: last action repeated {repeat_info['count']} times "
                f"({repeat_info.get('action')}); choose a different next action."
            )

        mime = "image/png" if self.settings.encode_format.lower() == "png" else "image/jpeg"

        # OpenRouter accepts OpenAI-style content with base64 image_url.
        task_hint = f"User request: {user_prompt}" if user_prompt else "No explicit user task provided."
        content = [
            {
                "type": "text",
                "text": f"{task_hint}\n\nPlan your next action, then call the computer tool once.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{observation_b64}"},
            },
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        return self.client.chat.completions.create(
            model=self.settings.openrouter_model,
            messages=messages,
            tools=[COMPUTER_TOOL],
            tool_choice="auto",
        )

    def _parse_tool_call(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract the first tool call and map it to the local action schema."""
        choices = getattr(response, "choices", [])
        if not choices:
            return None
        message = choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            # No tool call means the model replied with text; treat as noop.
            return {"type": "noop", "reason": "model returned text"}

        first = tool_calls[0]
        args_raw = first.function.arguments if hasattr(first, "function") else "{}"
        try:
            args = json.loads(args_raw or "{}")
        except json.JSONDecodeError:
            return {"type": "noop", "reason": f"bad tool args: {args_raw}"}

        return self._map_tool_args(args)

    def _map_tool_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = args.get("action")
        if action == "move_mouse":
            x = args.get("x")
            y = args.get("y")
            if x is None or y is None:
                return {"type": "noop", "reason": "move_mouse missing coordinates"}
            return {"type": "mouse_move", "x": float(x), "y": float(y)}
        if action == "left_click":
            return {"type": "left_click", "x": float(args.get("x", 0)), "y": float(args.get("y", 0))}
        if action == "right_click":
            return {"type": "right_click", "x": float(args.get("x", 0)), "y": float(args.get("y", 0))}
        if action == "double_click":
            return {"type": "double_click", "x": float(args.get("x", 0)), "y": float(args.get("y", 0))}
        if action == "scroll":
            return {"type": "scroll", "clicks": int(args.get("scroll_y", 0))}
        if action == "type":
            return {"type": "type", "text": args.get("text", "")}
        if action == "hotkey":
            return {"type": "key", "keys": args.get("keys") or []}
        if action == "wait":
            return {"type": "wait", "seconds": float(args.get("seconds", 1))}
        if action == "screenshot":
            return {"type": "capture_only", "reason": "model requested screenshot"}

        return {"type": "noop", "reason": f"unknown action {action}"}
