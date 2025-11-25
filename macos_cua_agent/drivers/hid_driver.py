from __future__ import annotations

from typing import Iterable, Optional, Tuple

from macos_cua_agent.agent.state_manager import ActionResult
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.coordinates import point_to_px
from macos_cua_agent.utils.logger import get_logger
from macos_cua_agent.utils.macos_integration import get_display_info


class HIDDriver:
    """Executes low-level HID events. Defaults to dry-run unless ENABLE_HID is true."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.enabled = bool(settings.enable_hid)
        self.display = get_display_info()
        self.pg = self._import_pyautogui() if self.enabled else None

    def _import_pyautogui(self) -> Optional[object]:
        try:
            import pyautogui  # type: ignore

            pyautogui.FAILSAFE = True
            return pyautogui
        except Exception as exc:
            self.logger.warning("pyautogui unavailable; HID execution disabled: %s", exc)
            self.enabled = False
            return None

    def move(self, x: float, y: float) -> ActionResult:
        return self._perform(lambda: self.pg.moveTo(*self._to_px(x, y)), f"move({x},{y})")

    def left_click(self, x: float, y: float) -> ActionResult:
        return self._perform(lambda: self.pg.click(*self._to_px(x, y)), f"left_click({x},{y})")

    def right_click(self, x: float, y: float) -> ActionResult:
        return self._perform(lambda: self.pg.click(*self._to_px(x, y), button="right"), f"right_click({x},{y})")

    def type_text(self, text: str) -> ActionResult:
        return self._perform(lambda: self.pg.write(text, interval=0.02), f"type:{text!r}")

    def press_keys(self, keys: Iterable[str]) -> ActionResult:
        combo = list(keys)
        return self._perform(lambda: self.pg.hotkey(*combo), f"hotkey:{'+'.join(combo)}")

    def scroll(self, clicks: int) -> ActionResult:
        return self._perform(lambda: self.pg.scroll(clicks), f"scroll:{clicks}")

    def _perform(self, func, label: str) -> ActionResult:
        if not self.enabled or not self.pg:
            self.logger.info("Dry-run HID: %s", label)
            return ActionResult(success=True, reason="dry-run")
        try:
            func()
            return ActionResult(success=True)
        except Exception as exc:
            self.logger.error("HID action failed: %s", exc)
            return ActionResult(success=False, reason=str(exc))

    def _to_px(self, x: float, y: float) -> Tuple[int, int]:
        return point_to_px(x, y, self.display.scale_factor)

