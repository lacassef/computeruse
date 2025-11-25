from __future__ import annotations

import base64
import io
import time
from typing import Optional

from PIL import Image, ImageChops, ImageDraw

from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger
from macos_cua_agent.utils.macos_integration import DisplayInfo, get_display_info


class VisionPipeline:
    """Captures the framebuffer, aligns to logical resolution, and encodes to base64."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.display: DisplayInfo = get_display_info()
        self.mss = self._build_mss()

    def _build_mss(self) -> Optional[object]:
        try:
            import mss  # type: ignore

            return mss.mss()
        except Exception:
            self.logger.warning("mss not available; using placeholder images.")
            return None

    def capture_base64(self) -> str:
        image = self._grab_frame()
        logical_size = (self.display.logical_width, self.display.logical_height)
        if image.size != logical_size:
            image = image.resize(logical_size)
        return self._encode_image(image)

    def _grab_frame(self) -> Image.Image:
        if self.mss:
            try:
                monitor = self.mss.monitors[1]  # Primary monitor only for V1
                raw = self.mss.grab(monitor)
                img = Image.frombytes("RGB", raw.size, raw.rgb)
                return img
            except Exception as exc:
                self.logger.warning("mss capture failed; falling back to placeholder: %s", exc)
        return self._placeholder_frame()

    def _placeholder_frame(self) -> Image.Image:
        width, height = self.display.logical_width, self.display.logical_height
        img = Image.new("RGB", (width, height), color=(32, 32, 32))
        draw = ImageDraw.Draw(img)
        text = f"Placeholder frame @ {time.strftime('%H:%M:%S')} ({width}x{height})"
        draw.text((20, 20), text, fill=(200, 200, 200))
        return img

    def _encode_image(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        fmt = "PNG" if self.settings.encode_format.upper() == "PNG" else "JPEG"
        image.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def has_changed(self, previous_b64: str, current_b64: str, threshold: float = 0.01) -> bool:
        """Simple pixel delta check to decide if the UI changed."""
        try:
            prev_img = self._decode(previous_b64)
            curr_img = self._decode(current_b64)
        except Exception as exc:
            self.logger.warning("Failed to decode frames; assuming changed: %s", exc)
            return True

        diff = ImageChops.difference(prev_img, curr_img)
        histogram = diff.histogram()
        diff_score = sum(i * count for i, count in enumerate(histogram))
        max_score = 255 * sum(histogram)
        if max_score == 0:
            return False
        ratio = diff_score / max_score
        return ratio >= threshold

    def _decode(self, image_b64: str) -> Image.Image:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
