from __future__ import annotations

import subprocess
from typing import Optional

from macos_cua_agent.agent.state_manager import ActionResult
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger


class BrowserDriver:
    """
    Handles semantic interactions with web browsers (Safari, Chrome) via AppleScript.
    Treats the web as a data source (DOM) rather than pixels.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)

    def execute_browser_action(self, action: dict) -> ActionResult:
        """Dispatch browser actions."""
        cmd = action.get("command")
        app = action.get("app_name", "Safari")  # Default to Safari if not specified

        if cmd == "get_page_content":
            return self._get_page_content(app)
        if cmd == "get_links":
            return self._get_links(app)
        if cmd == "navigate":
            return self._navigate(app, action.get("url"))
        
        return ActionResult(success=False, reason=f"unknown browser command: {cmd}")

    def _get_page_content(self, app_name: str) -> ActionResult:
        js = "document.body.innerText"
        return self._run_js(app_name, js, "get_content")

    def _get_links(self, app_name: str) -> ActionResult:
        # valid JSON output from JS
        js = """
        JSON.stringify(Array.from(document.links).slice(0, 50).map(l => ({
            text: l.innerText.replace(/\\n/g, ' ').trim(),
            url: l.href
        })).filter(l => l.text.length > 0))
        """
        return self._run_js(app_name, js, "get_links")

    def _navigate(self, app_name: str, url: Optional[str]) -> ActionResult:
        if not url:
            return ActionResult(success=False, reason="url required")
        
        js = f"window.location.href = '{url}'"
        return self._run_js(app_name, js, "navigate")

    def _run_js(self, app_name: str, js_code: str, label: str) -> ActionResult:
        script = self._build_applescript(app_name, js_code)
        try:
            # self.logger.debug("Running AppleScript for %s: %s", label, script)
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return ActionResult(
                    success=True, 
                    reason=f"{label} success", 
                    metadata={"output": result.stdout.strip()}
                )
            return ActionResult(
                success=False, 
                reason=f"{label} failed: {result.stderr.strip()}"
            )
        except Exception as e:
            return ActionResult(success=False, reason=f"{label} exception: {str(e)}")

    def _build_applescript(self, app_name: str, js_code: str) -> str:
        # Escape double quotes and backslashes for AppleScript string
        safe_js = js_code.replace("\\", "\\\\").replace('"', '\\"')
        
        if "Chrome" in app_name:
            return f'''
            tell application "{app_name}"
                execute front window's active tab javascript "{safe_js}"
            end tell
            '''
        elif "Safari" in app_name:
            return f'''
            tell application "{app_name}"
                do JavaScript "{safe_js}" in front document
            end tell
            '''
        else:
            # Fallback/Assumption: behave like Chrome (Brave, Edge, etc often support same suite)
             return f'''
            tell application "{app_name}"
                execute front window's active tab javascript "{safe_js}"
            end tell
            '''
