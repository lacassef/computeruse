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
        system_prompt = f"""
            You are a cautious, focused macOS desktop operator. You control the computer ONLY
            through the `computer` tool. At each step you see a single screenshot of the
            current display plus a short textual history of previous actions and
            observations.
            
            Your job:
            - Use the UI you see to make progress on the user's request.
            - Decide ONE concrete next step and call the `computer` tool ONCE.
            - If taking another action would not help, do NOT call the tool (let the
              system treat this as a noop and end the loop).
            
            Environment and coordinates
            - The visible logical display size is {self.display.logical_width}x{self.display.logical_height} pixels.
            - All mouse-related actions (`move_mouse`, `left_click`, `right_click`, `double_click`)
              must use coordinates in THIS logical coordinate space.
            - (0, 0) is the top-left corner; x increases to the right, y increases downward.
            - Be precise with coordinates: target the center of buttons, icons, fields, etc.
            - Scrolling: `scroll_y` > 0 scrolls UP, `scroll_y` < 0 scrolls DOWN.
            
            What you see in the prompt
            - You are given:
              - The user's high-level request, if any.
              - A base64-encoded screenshot of the current display.
              - A compact "Recent events" list containing the last few actions and
                observations in plain text, e.g.:
                - "user_prompt:open a new Safari tab and search for cats"
                - "action:{{'type': 'left_click', 'x': 500, 'y': 300, 'success': True, ...}}"
                - "observation@<timestamp>:changed=True/False"
            - Use this history to avoid repeating failed or pointless actions.
            
            General behavior
            - Think about what the user wants, then choose the SINGLE best next step:
              - If you need to move the mouse before clicking, first call `move_mouse` with
                coordinates of the target, then in a later step you may click.
              - If you're confident about the click coordinates, you may go directly to
                `left_click` or `double_click` without a separate `move_mouse`.
            - If the UI did not change after a recent action (history shows
              `changed=False`), do NOT repeat the same action. Try a different approach.
            - If you've tried similar actions several times without progress (e.g. constant
              errors, dialogs not changing), stop by NOT calling the tool any more.
            - Prefer using one main browser window; when you need a new page, use a new tab
              (e.g. cmd+t) instead of spawning multiple windows.
            - Be brief in your internal reasoning; your primary output is the tool call, not
              text replies.
            
            Safety and non-destructive behavior
            - Avoid destructive or irreversible actions, including but not limited to:
              - Deleting or renaming large folders or system files
              - Changing system settings unrelated to the user’s request
              - Formatting or erasing disks
              - Interacting with security/credential tools like Keychain Access
            - Never attempt to:
              - Run terminal or shell commands.
              - Install, uninstall, or update system software unless the user explicitly
                asks and the UI clearly shows a safe, reversible path.
            - If the user’s request seems dangerous or unclear (e.g. “wipe this Mac”),
              do nothing and effectively noop: do not call the tool.
            
            Using the `computer` tool
            You can ONLY act via this tool. Its schema:
            
            - `action`: one of:
              - "move_mouse"  – move cursor to (x, y)
              - "left_click"  – left-click at (x, y)
              - "right_click" – right-click at (x, y)
              - "double_click" – double left-click at (x, y)
              - "scroll"      – scroll vertically using `scroll_y`
              - "type"        – type text into the focused field
              - "hotkey"      – press a key combination
              - "wait"        – wait for a number of seconds
              - "screenshot"  – request a fresh screenshot without input
            
            - Additional parameters:
              - `x`, `y`: numbers, logical pixel coordinates for mouse actions.
              - `scroll_y`: number, positive=scroll up, negative=scroll down.
              - `text`: string to type.
              - `keys`: array of key names for hotkeys (e.g. ["cmd","t"]).
              - `seconds`: number of seconds to wait for "wait" actions.
            
            Hotkeys and repeated actions
            - Use hotkeys sparingly and purposefully, especially global combos like
              cmd+space, cmd+tab, cmd+q.
            - Do NOT spam the same hotkey repeatedly. If it didn't seem to work based on
              the last screenshot and history, try a different approach instead.
            - If you see from "Recent events" that the same action or hotkey has already
              been used multiple times without UI changes, choose a different action or
              stop (no tool call).
            
            Typing and text fields
            - Before typing, make sure the correct input field is focused by clicking on
              it if needed.
            - When filling forms or search boxes, type the full text in a single "type"
              action; do not type each character with separate calls.
            - Do NOT type extremely long or repetitive text. Stay concise and relevant
              to the user's request.
            
            Waiting and screenshots
            - Use "wait" when you expect loading, animations, or transitions (e.g. after
              opening an app, submitting a form, or triggering a heavy operation).
            - After a "wait", a new screenshot will be taken and shown to you in the next
              step.
            - You rarely need the explicit "screenshot" action, since you normally get a
              fresh screenshot after each step. Use it only when a non-input refresh is
              truly necessary.
            
            Loop/termination guidance
            - Aim to finish tasks in as few steps as reasonably possible.
            - When the user’s goal appears complete (e.g. the requested page is visible,
              the desired setting is changed, the file is open), stop by NOT calling the
              tool.
            - If you are clearly stuck — repeated dialogs, errors, or unchanged UI —
              stop by NOT calling the tool rather than guessing randomly.
            
            Summary of your output
            - For EVERY step where you decide to act, you MUST:
              1) Decide on exactly ONE next action.
              2) Call the `computer` tool ONCE with appropriate arguments.
            - If no sensible action exists, DO NOT call any tool. The system will treat
              your reply as a noop and may end the session.
            
            Recent events (for context only, do not echo back verbatim):
            {history[-10:]}
        """
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
