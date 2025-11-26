from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import platform

from macos_cua_agent.agent.state_manager import ActionResult
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger

# Attempt imports for Accessibility API
try:
    import ApplicationServices
    from ApplicationServices import (
        AXUIElementCreateSystemWide,
        AXUIElementCopyAttributeValue,
        kAXFocusedApplicationAttribute,
        kAXFocusedWindowAttribute,
        kAXChildrenAttribute,
        kAXRoleAttribute,
        kAXTitleAttribute,
        kAXValueAttribute,
        kAXPositionAttribute,
        kAXSizeAttribute,
        kAXSubroleAttribute,
        kAXRoleDescriptionAttribute,
    )
    from CoreFoundation import CFArrayGetCount, CFArrayGetValueAtIndex
    HAS_AX = True
except ImportError:
    HAS_AX = False


class AccessibilityDriver:
    """
    Provides semantic access to the macOS UI via the Accessibility API (AXUIElement).
    Allows the agent to 'see' the structure of windows and controls, not just pixels.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        if not HAS_AX:
            self.logger.warning("pyobjc ApplicationServices not found; AccessibilityDriver disabled.")

    def get_active_window_tree(self, max_depth: int = 5) -> ActionResult:
        """
        Captures the UI tree of the currently focused window.
        Returns a JSON-serializable dict structure.
        """
        if not HAS_AX:
            return ActionResult(success=False, reason="Accessibility API unavailable (missing pyobjc)")

        try:
            system_wide = AXUIElementCreateSystemWide()
            
            # Get focused app
            err, app = AXUIElementCopyAttributeValue(system_wide, kAXFocusedApplicationAttribute, None)
            if err != 0:
                return ActionResult(success=False, reason="No focused application found")

            # Get focused window
            err, window = AXUIElementCopyAttributeValue(app, kAXFocusedWindowAttribute, None)
            if err != 0:
                return ActionResult(success=False, reason="No focused window found")

            # Build tree
            tree = self._build_tree(window, depth=0, max_depth=max_depth)
            return ActionResult(success=True, reason="captured", metadata={"tree": tree})

        except Exception as e:
            self.logger.exception("Failed to capture accessibility tree")
            return ActionResult(success=False, reason=f"AX error: {str(e)}")

    def _build_tree(self, element: Any, depth: int, max_depth: int) -> Dict[str, Any]:
        if depth > max_depth:
            return {"role": "max_depth_reached"}

        node = {}
        
        # Basic attributes
        node["role"] = self._get_attr(element, kAXRoleAttribute)
        node["title"] = self._get_attr(element, kAXTitleAttribute)
        node["value"] = self._get_attr(element, kAXValueAttribute)
        
        # Construct frame from position and size
        pos = self._get_attr(element, kAXPositionAttribute)
        size = self._get_attr(element, kAXSizeAttribute)
        if pos is not None and size is not None:
            # Handle pyobjc struct wrappers if necessary, but usually they behave like objects/tuples
            # AXValue (CGPoint/CGSize) might need unwrapping if not auto-bridged
            try:
                x = getattr(pos, 'x', pos[0] if hasattr(pos, '__getitem__') else 0)
                y = getattr(pos, 'y', pos[1] if hasattr(pos, '__getitem__') else 0)
                w = getattr(size, 'width', size[0] if hasattr(size, '__getitem__') else 0)
                h = getattr(size, 'height', size[1] if hasattr(size, '__getitem__') else 0)
                node["frame"] = {'x': x, 'y': y, 'w': w, 'h': h}
            except Exception:
                 node["frame"] = None
        else:
            node["frame"] = None
        
        # Pruning: Skip uninteresting elements if they have no useful content
        # (This is a heuristic to save tokens)
        if not node["title"] and not node["value"] and node["role"] in ["AXGroup", "AXUnknown"]:
             pass # We still check children
        
        # Recursion
        children = []
        err, ax_children = AXUIElementCopyAttributeValue(element, kAXChildrenAttribute, None)
        
        if err == 0 and ax_children:
            # ax_children is a CFArray or list depending on bridge
            # Python bridge usually converts CFArray to list
            if isinstance(ax_children, (list, tuple)):
                for child in ax_children:
                    child_node = self._build_tree(child, depth + 1, max_depth)
                    if self._is_useful_node(child_node):
                        children.append(child_node)
            else:
                 # Handle raw CFArray if needed (rare with modern pyobjc)
                 pass
        
        if children:
            node["children"] = children
        elif not self._is_useful_node(node):
            # If it's a leaf node and useless, we might discard it, but 
            # let's be careful. For now, we return it if it has a frame.
            pass

        return node

    def _get_attr(self, element: Any, attr_name: str) -> Any:
        try:
            err, value = AXUIElementCopyAttributeValue(element, attr_name, None)
            if err == 0:
                # Convert complex types if necessary
                return value
        except Exception:
            pass
        return None

    def _is_useful_node(self, node: Dict[str, Any]) -> bool:
        """Heuristic to determine if a node is worth showing to the LLM."""
        # If it has children, it's useful (structure)
        if "children" in node and node["children"]:
            return True
        # If it has text or value
        if node.get("title") or node.get("value"):
            return True
        # If it is a control
        if node.get("role") in ["AXButton", "AXTextField", "AXTextArea", "AXLink", "AXCheckBox"]:
            return True
        return False
