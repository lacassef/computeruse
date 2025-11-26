from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger
from macos_cua_agent.utils.macos_integration import get_display_info, get_system_info

if TYPE_CHECKING:  # pragma: no cover - typing only
    from macos_cua_agent.orchestrator.planning import Plan, Step

# OpenRouter exposes an OpenAI-compatible tool-calling API. We define our own
# computer tool schema so Claude Opus 4.5 can drive local actions.
COMPUTER_TOOL = {
    "type": "function",
    "function": {
        "name": "computer",
        "description": (
            "Control the macOS desktop: move and click the mouse, type, press hotkeys, "
            "scroll, wait, request a screenshot, or inspect the UI tree."
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
                        "open_app",
                        "inspect_ui",
                    ],
                },
                "x": {"type": "number", "description": "X coordinate in logical pixels."},
                "y": {"type": "number", "description": "Y coordinate in logical pixels."},
                "scroll_y": {
                    "type": "number",
                    "description": "Vertical scroll amount (positive up, negative down).",
                },
                "text": {"type": "string", "description": "Text to type."},
                "app_name": {"type": "string", "description": "Name of the application to open (for 'open_app' action)."},
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

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": (
            "Run safe, sandboxed shell commands in a constrained workspace. "
            "Use this for local file operations or running short scripts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Full command line, e.g. 'ls -la' or 'python script.py'.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional relative working directory under the agent workspace.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

NOTEBOOK_TOOL = {
    "type": "function",
    "function": {
        "name": "notebook",
        "description": "Manage a persistent notebook for storing research notes, facts, and data across steps.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add_note", "clear_notes"]},
                "content": {"type": "string", "description": "The note content to save."},
                "source": {"type": "string", "description": "Source of the info (e.g. url or 'user')."}
            },
            "required": ["action"]
        }
    }
}

BROWSER_TOOL = {
    "type": "function",
    "function": {
        "name": "browser",
        "description": "Interact with web browsers (Safari/Chrome) semantically to read content and get links without OCR.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": ["get_page_content", "get_links", "navigate"]},
                "app_name": {"type": "string", "description": "Safari or Google Chrome", "default": "Safari"},
                "url": {"type": "string", "description": "URL to navigate to"}
            },
            "required": ["command"]
        }
    }
}


class CognitiveCore:
    """Calls Claude Opus 4.5 via OpenRouter with a custom computer tool."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.display = get_display_info()
        self.system_info = get_system_info()
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
        plan: Optional["Plan"] = None,
        current_step: Optional["Step"] = None,
        loop_state: Optional[Dict[str, Any]] = None,
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
                observation_b64,
                history,
                user_prompt=user_prompt,
                repeat_info=repeat_info,
                plan=plan,
                current_step=current_step,
                loop_state=loop_state,
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
        plan: Optional["Plan"],
        current_step: Optional["Step"],
        loop_state: Optional[Dict[str, Any]],
    ) -> Any:
        """Send a vision + tool-calling request to OpenRouter."""
        plan_text = "No structured plan; infer progress from the user's request."
        if plan and current_step:
            upcoming = [
                f"- Step {s.id}: {s.description} (status={s.status})"
                for s in plan.steps
                if s.id != current_step.id
            ]
            plan_text = (
                "Current goal:\n"
                f"- Step {current_step.id}: {current_step.description}\n"
                f"- Success criteria: {current_step.success_criteria}\n"
            )
            if getattr(current_step, "expected_state", ""):
                plan_text += f"- Expected state: {current_step.expected_state}\n"
            if upcoming:
                plan_text += "Upcoming steps (context only):\n" + "\n".join(upcoming[:4])
        elif plan:
            plan_lines = [f"- Step {s.id}: {s.description} (status={s.status})" for s in plan.steps]
            plan_text = "Plan:\n" + "\n".join(plan_lines)

        loop_state_text = ""
        if loop_state:
            notebook = loop_state.get("notebook_summary", "")
            if notebook:
                loop_state_text += f"\n{notebook}\n"
            
            loop_bits = [f"{k}={v}" for k, v in loop_state.items() if v not in (None, "") and k != "notebook_summary"]
            if loop_bits:
                loop_state_text += "Loop state: " + ", ".join(loop_bits)

        system_prompt = f"""
            You are a cautious, focused macOS desktop operator. You have a robust toolbox including:
            - `computer`: for low-level mouse/keyboard interaction and UI inspection (`inspect_ui`).
            - `browser`: for high-speed reading and navigation of web pages (use this for research).
            - `notebook`: for saving facts and notes to persistent memory (use this to avoid forgetting things).
            - `shell`: for local workspace file operations.

            At each step you see a single screenshot of the current display plus a short textual history of previous actions and
            observations.
            {plan_text}
            {loop_state_text}

            Planning & Thinking
            - Always reason from what is currently visible: windows, icons, menus.
            - Use `inspect_ui` if visual elements are ambiguous or you need to find hidden controls.
            - For Research:
              1. Use `browser` tool to `get_links` or `get_page_content`.
              2. Read the content.
              3. SAVE key findings using `notebook` tool (`add_note`).
              4. This prevents data loss when context window fills up.

            Environment
            - System: {self.system_info}
            - Logical display: {self.display.logical_width}x{self.display.logical_height} pixels.
            - (0, 0) is top-left.
            
            Safety
            - No destructive actions.
            - No network access via shell (use browser tool).
            - `shell` is sandboxed.
            
            Action Selection
            - ONE action per step.
            - Prefer `browser` tools over `computer` OCR/Vision for text-heavy web tasks.
            - Prefer `inspect_ui` over random guessing of coordinates.
            
            Recent events:
            {history[-10:]}
        """
        if repeat_info and repeat_info.get("count", 0) >= 2:
            system_prompt += (
                f" Warning: last action repeated {repeat_info['count']} times "
                f"({repeat_info.get('action')}); choose a different next action."
            )
        if repeat_info and repeat_info.get("hint"):
            system_prompt += f" Hint from verifier: {repeat_info['hint']}."

        mime = "image/png" if self.settings.encode_format.lower() == "png" else "image/jpeg"

        task_hint = f"User request: {user_prompt}" if user_prompt else "No explicit user task provided."
        content = [
            {
                "type": "text",
                "text": f"{task_hint}\n\nPlan your next action, then call one tool once.",
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

        # Prepare reasoning parameters
        extra_body = {}
        if self.settings.reasoning_effort or self.settings.reasoning_max_tokens:
            reasoning_config = {}
            if self.settings.reasoning_effort:
                reasoning_config["effort"] = self.settings.reasoning_effort
            elif self.settings.reasoning_max_tokens:  # Use elif to ensure mutual exclusivity
                reasoning_config["max_tokens"] = self.settings.reasoning_max_tokens
            
            # Only add 'reasoning' to extra_body if at least one config is present
            if reasoning_config:
                extra_body["reasoning"] = reasoning_config

        return self.client.chat.completions.create(
            model=self.settings.openrouter_model,
            messages=messages,
            tools=[COMPUTER_TOOL, SHELL_TOOL, NOTEBOOK_TOOL, BROWSER_TOOL],
            tool_choice="auto",
            extra_body=extra_body if extra_body else None,
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
        tool_name = getattr(first.function, "name", None)
        args_raw = first.function.arguments if hasattr(first, "function") else "{}"
        try:
            args = json.loads(args_raw or "{}")
        except json.JSONDecodeError:
            return {"type": "noop", "reason": f"bad tool args: {args_raw}"}

        if tool_name == "computer":
            return self._map_tool_args(args)
        if tool_name == "shell":
            return self._map_shell_args(args)
        if tool_name == "notebook":
            return self._map_notebook_args(args)
        if tool_name == "browser":
            return self._map_browser_args(args)
        return {"type": "noop", "reason": f"unknown tool {tool_name}"}

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
        if action == "open_app":
            return {"type": "open_app", "app_name": args.get("app_name", "")}
        if action == "inspect_ui":
            return {"type": "inspect_ui"}

        return {"type": "noop", "reason": f"unknown action {action}"}

    def _map_shell_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        command = args.get("command") or ""
        cwd = args.get("cwd")
        if not command:
            return {"type": "noop", "reason": "shell command missing"}

        return {
            "type": "sandbox_shell",
            "cmd": command,
            "cwd": cwd,
            "execution": "shell",
        }

    def _map_notebook_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = args.get("action")
        return {
            "type": "notebook_op",
            "action": action,
            "content": args.get("content", ""),
            "source": args.get("source", "agent"),
            "execution": "notebook"
        }

    def _map_browser_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "browser_op",
            "command": args.get("command"),
            "app_name": args.get("app_name", "Safari"),
            "url": args.get("url"),
            "execution": "browser"
        }
