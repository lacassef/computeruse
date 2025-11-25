from __future__ import annotations

from macos_cua_agent.agent.state_manager import ActionResult
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger


class SemanticDriver:
    """Stub for semantic execution via AX APIs or AppleScript."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)

    def execute(self, action: dict) -> ActionResult:
        self.logger.info("Semantic driver stub invoked with action: %s", action)
        return ActionResult(success=False, reason="semantic path not implemented")

