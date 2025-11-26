from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from macos_cua_agent.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DisplayInfo:
    logical_width: int
    logical_height: int
    physical_width: int
    physical_height: int
    scale_factor: float


def get_display_info() -> DisplayInfo:
    """Return display info with graceful fallbacks when pyobjc is unavailable."""
    info = _from_pyobjc()
    if info:
        return info

    info = _from_pyautogui()
    if info:
        return info

    logger.warning("Falling back to default display info (1280x720, scale=1.0).")
    return DisplayInfo(
        logical_width=1280,
        logical_height=720,
        physical_width=1280,
        physical_height=720,
        scale_factor=1.0,
    )


def _from_pyobjc() -> Optional[DisplayInfo]:
    try:
        from AppKit import NSScreen  # type: ignore
    except Exception:
        return None

    screen = NSScreen.mainScreen()
    if not screen:
        return None

    frame = screen.frame()
    logical_width = int(frame.size.width)
    logical_height = int(frame.size.height)
    scale = float(screen.backingScaleFactor()) if hasattr(screen, "backingScaleFactor") else 1.0
    return DisplayInfo(
        logical_width=logical_width,
        logical_height=logical_height,
        physical_width=int(logical_width * scale),
        physical_height=int(logical_height * scale),
        scale_factor=scale,
    )


def _from_pyautogui() -> Optional[DisplayInfo]:
    try:
        import pyautogui  # type: ignore
    except Exception:
        return None

    width, height = pyautogui.size()
    return DisplayInfo(
        logical_width=int(width),
        logical_height=int(height),
        physical_width=int(width),
        physical_height=int(height),
        scale_factor=1.0,
    )


def has_screen_recording_permission() -> bool:
    try:
        import Quartz  # type: ignore

        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        return False


def has_accessibility_permission() -> bool:
    try:
        import Quartz  # type: ignore

        return bool(Quartz.AXIsProcessTrusted())
    except Exception:
        return False


def get_system_info() -> str:
    """Returns a string describing the macOS version and hardware model."""
    import platform
    import subprocess

    try:
        ver, _, arch = platform.mac_ver()
        model = "Unknown Mac"
        try:
            # sysctl -n hw.model returns something like "Mac14,2" or "MacBookPro18,3"
            res = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True)
            model = res.strip()
        except Exception:
            pass

        return f"macOS {ver} ({arch}) on {model}"
    except Exception:
        return "macOS (Unknown System)"

