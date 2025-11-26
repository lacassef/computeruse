from __future__ import annotations

from typing import List, Optional

from macos_cua_agent.orchestrator.planning import Step
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger


class Reflector:
    """Uses a secondary model (ChatGPT/Gemini/etc.) to verify progress or offer hints."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.client = self._build_client()
        self.mime = "image/png" if settings.encode_format.lower() == "png" else "image/jpeg"

    def _build_client(self) -> Optional[object]:
        if not self.settings.enable_reflection:
            return None
        api_key = self.settings.reflector_api_key
        if not api_key:
            self.logger.info("Reflection disabled: REFLECTOR_API_KEY/OPENROUTER_API_KEY missing.")
            return None
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            self.logger.warning("openai package unavailable for reflection: %s", exc)
            return None
        return OpenAI(base_url=self.settings.reflector_base_url, api_key=api_key)

    @property
    def available(self) -> bool:
        return self.client is not None

    def is_step_complete(self, step: Step, history: List[str], screenshot_b64: str, changed: bool) -> bool:
        """Ask the verifier model if the step is complete; default to incomplete on failure."""
        if not self.client:
            return False
        expected_text = ""
        if getattr(step, "expected_state", ""):
            expected_text = f"Expected state before the step: {step.expected_state}\n"
        prompt = (
            "You are a visual verifier for a desktop agent. Given the current sub-step, "
            "recent history, and a screenshot, decide if the step is COMPLETE right now.\n"
            "Step: {desc}\nSuccess criteria: {criteria}\n{expected_text}\n"
            "Say 'yes' only if the screenshot clearly satisfies the success criteria. "
            "If you are uncertain, the view seems unrelated, or the UI looks mid-process "
            "(loading indicators, progress bars, partial results), answer 'no'."
        ).format(desc=step.description, criteria=step.success_criteria, expected_text=expected_text)
        content = [
            {"type": "text", "text": f"{prompt}\n\nRecent events:\n" + "\n".join(history[-20:])},
            {"type": "image_url", "image_url": {"url": f"data:{self.mime};base64,{screenshot_b64}"}},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.settings.reflector_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Reply with only 'yes' or 'no'. Be conservative: if there is any uncertainty, "
                            "the process looks in-progress (loading spinner, progress bar, 'downloading...', etc.), "
                            "or the success criteria might not be fully satisfied in the screenshot, answer 'no'."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                max_tokens=1000,
            )
            raw = response.choices[0].message.content if response and response.choices else ""
            text = "".join([frag.text for frag in raw]) if isinstance(raw, list) else str(raw or "")
            normalized = text.strip().lower()
            return normalized.startswith("y")
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Reflection check failed; treating step as incomplete: %s", exc)
            return False

    def suggest_hint(self, step: Optional[Step], history: List[str], screenshot_b64: str) -> str:
        if not self.client:
            return ""
        step_text = ""
        if step:
            step_text = f"Current step: {step.description}. Success criteria: {step.success_criteria}."
        prompt = (
            "The desktop agent appears stuck. Based on the screenshot and recent events, "
            "give one short hint (<=20 words) to unblock progress. Avoid telling the user, "
            "just give the agent a next idea."
        )
        content = [
            {"type": "text", "text": f"{prompt}\n{step_text}\nRecent events:\n" + "\n".join(history[-30:])},
            {"type": "image_url", "image_url": {"url": f"data:{self.mime};base64,{screenshot_b64}"}},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.settings.reflector_model,
                messages=[
                    {"role": "system", "content": "Provide one concise hint, no preamble."},
                    {"role": "user", "content": content},
                ],
                max_tokens=1000,
            )
            raw = response.choices[0].message.content if response and response.choices else ""
            return "".join([frag.text for frag in raw]) if isinstance(raw, list) else str(raw or "")
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Reflection hint failed: %s", exc)
            return ""
