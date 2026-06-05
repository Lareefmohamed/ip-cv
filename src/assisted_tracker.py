"""Assisted tracking using OpenCV's CSRT correlation-filter tracker.

CSRT (Channel and Spatial Reliability Tracker) is a *classical* computer-vision
tracker — discriminative correlation filters over HOG / colour-names features.
It is **not** a deep-learning model, so it stays inside the project's
"traditional CV" scope, yet it is far more robust than HSV colour masking on
hard footage: it follows a white ball, copes with motion blur, partial
occlusion and cluttered backgrounds because it locks onto the ball's *texture*,
not just its colour.

Two ways to start a track:
    * MANUAL  -- the user draws a box around the ball on a chosen frame
                 (CLI uses cv2.selectROI; the UI passes an explicit bbox).
    * AUTO    -- the HSV+motion detector finds the first confident ball and
                 hands the box to CSRT automatically (no clicking).

If CSRT reports failure for too many frames, the tracker tries to re-acquire
using the detector again (hybrid re-seed), which is what makes it survive a
ball briefly leaving / re-entering view.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .detector import BallDetector, Detection

BBox = Tuple[int, int, int, int]   # x, y, w, h


def _make_csrt():
    """Create a CSRT tracker across OpenCV API variants."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT"):
        return cv2.TrackerCSRT.create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError(
        "CSRT tracker unavailable. Install opencv-contrib-python:\n"
        "    pip install opencv-contrib-python")


def detection_to_bbox(d: Detection, pad: float = 1.6) -> BBox:
    """Convert a detector hit into a square CSRT init box around the ball."""
    half = max(3.0, d.radius * pad)
    x = int(d.cx - half)
    y = int(d.cy - half)
    s = int(half * 2)
    return (x, y, s, s)


class AssistedTracker:
    """CSRT tracker with optional HSV-detector auto-seed and re-acquisition."""

    def __init__(self, config: Optional[PipelineConfig] = None,
                 reacquire: bool = True):
        self.cfg = config or PipelineConfig()
        self.reacquire = reacquire
        self._csrt = None
        self._seed_detector = BallDetector(self.cfg.detector)
        self._lost = 0
        self.last_radius: float = 8.0
        self.trail: Deque[Tuple[int, int]] = deque(
            maxlen=self.cfg.tracker.trail_length)
        self.initialised = False

    # -- initialisation -----------------------------------------------------

    def init_with_bbox(self, frame: np.ndarray, bbox: BBox) -> bool:
        """Start tracking from an explicit (x, y, w, h) box (manual mode)."""
        bbox = self._clamp_bbox(bbox, frame.shape[1], frame.shape[0])
        if bbox is None:
            self.initialised = False
            return False
        self._csrt = _make_csrt()
        # NOTE: in OpenCV >= 4.5 Tracker.init() returns None (raises on error),
        # so we treat a clean call as success rather than checking a bool.
        try:
            self._csrt.init(frame, tuple(int(v) for v in bbox))
        except cv2.error:
            self._csrt = None
            self.initialised = False
            return False
        self.initialised = True
        self._lost = 0
        x, y, w, h = bbox
        self.last_radius = max(w, h) / 2.0
        self.trail.clear()
        self.trail.append((int(x + w / 2), int(y + h / 2)))
        return True

    @staticmethod
    def _clamp_bbox(bbox: BBox, width: int, height: int) -> Optional[BBox]:
        """Clamp a box inside the frame; reject if it has no area."""
        x, y, w, h = (int(v) for v in bbox)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(2, min(w, width - x))
        h = max(2, min(h, height - y))
        if w < 2 or h < 2:
            return None
        return (x, y, w, h)

    def auto_seed(self, frame: np.ndarray) -> bool:
        """Try to auto-initialise from the HSV+motion detector (auto mode).

        Returns True once a confident ball is found and CSRT is initialised.
        Call this each frame until it succeeds (the motion gate needs a few
        frames to warm up).
        """
        candidates, _ = self._seed_detector.detect(frame)
        if not candidates:
            return False
        return self.init_with_bbox(frame, detection_to_bbox(candidates[0]))

    # -- per-frame update ---------------------------------------------------

    def update(self, frame: np.ndarray
               ) -> Optional[Tuple[float, float, str]]:
        """Advance one frame. Returns (x, y, source) or None.

        source is "measured" (CSRT locked) or "reacquiring".
        """
        if not self.initialised or self._csrt is None:
            # Not started yet: in auto mode keep trying to seed.
            if self.auto_seed(frame):
                cx, cy = self.trail[-1]
                return (float(cx), float(cy), "measured")
            return None

        ok, box = self._csrt.update(frame)
        if ok:
            x, y, w, h = box
            cx, cy = x + w / 2.0, y + h / 2.0
            self.last_radius = max(w, h) / 2.0
            self._lost = 0
            self.trail.append((int(cx), int(cy)))
            return (cx, cy, "measured")

        # CSRT lost the ball.
        self._lost += 1
        if self.reacquire and self._lost <= self.cfg.tracker.max_coast_frames:
            # Try to re-seed from the detector around recent motion.
            if self.auto_seed(frame):
                cx, cy = self.trail[-1]
                return (float(cx), float(cy), "reacquiring")
        if self._lost > self.cfg.tracker.max_coast_frames:
            self.initialised = False
            self._csrt = None
        return None


# ---------------------------------------------------------------------------
# CLI helper: pick the ball with the mouse on a chosen frame.
# ---------------------------------------------------------------------------

def select_bbox_interactive(frame: np.ndarray,
                            window: str = "Select the ball - ENTER to confirm"
                            ) -> Optional[BBox]:
    """Open a window so the user can drag a box around the ball.

    Returns (x, y, w, h) or None if the selection was cancelled/empty.
    """
    box = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    x, y, w, h = box
    if w == 0 or h == 0:
        return None
    return (int(x), int(y), int(w), int(h))
