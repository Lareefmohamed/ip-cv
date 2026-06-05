"""Motion-consistency tracker using a constant-velocity Kalman filter.

The detector may return several ball-like candidates per frame (and the odd
false positive).  This tracker keeps a single best estimate of the ball's
state ``[x, y, vx, vy]`` and, each frame:

    1. predicts where the ball should be,
    2. picks the candidate closest to that prediction (within a gate),
    3. corrects the estimate with that measurement, OR
    4. "coasts" on the prediction for a few frames if nothing matches
       (handles brief occlusion, motion blur, missed detections).

This is what turns noisy per-frame detections into a clean, high-accuracy
track and prevents the marker from jumping onto crowd / clothing clutter.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from .config import TrackerConfig
from .detector import Detection


class _KalmanCV:
    """Thin wrapper over cv2.KalmanFilter for a 2-D constant-velocity model."""

    def __init__(self, x: float, y: float):
        # state: [x, y, vx, vy]  measurement: [x, y]
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kf = kf

    def predict(self) -> Tuple[float, float]:
        p = self.kf.predict()
        return float(p[0, 0]), float(p[1, 0])

    def correct(self, x: float, y: float) -> Tuple[float, float]:
        m = np.array([[np.float32(x)], [np.float32(y)]])
        s = self.kf.correct(m)
        return float(s[0, 0]), float(s[1, 0])


class BallTracker:
    """Single-object tracker with prediction-gated data association."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.cfg = config or TrackerConfig()
        self._kf: Optional[_KalmanCV] = None
        self._coast = 0          # consecutive coasted (un-measured) frames
        self._hits = 0           # consecutive successful measurements
        self.confirmed = False
        self.trail: Deque[Tuple[int, int]] = deque(maxlen=self.cfg.trail_length)
        self.last_radius: float = 0.0

    def reset(self) -> None:
        self._kf = None
        self._coast = 0
        self._hits = 0
        self.confirmed = False
        self.trail.clear()
        self.last_radius = 0.0

    def update(self, candidates: List[Detection]
               ) -> Optional[Tuple[float, float, str]]:
        """Advance one frame.

        Returns ``(x, y, source)`` where source is ``"measured"`` or
        ``"predicted"``, or ``None`` if there is currently no track.
        """
        # No active track yet -> start one from the best candidate.
        if self._kf is None:
            if not candidates:
                return None
            d = candidates[0]
            self._kf = _KalmanCV(d.cx, d.cy)
            self._hits = 1
            self.last_radius = d.radius
            self.trail.append((int(d.cx), int(d.cy)))
            return (d.cx, d.cy, "measured")

        # Predict current position.
        px, py = self._kf.predict()

        # Gate widens while coasting (uncertainty grows without measurements).
        gate = self.cfg.max_match_dist * (1.0 + 0.5 * self._coast)
        match = self._closest_within(candidates, px, py, gate)

        if match is not None:
            cx, cy = self._kf.correct(match.cx, match.cy)
            self._coast = 0
            self._hits += 1
            self.last_radius = match.radius
            if self._hits >= self.cfg.min_hits_to_confirm:
                self.confirmed = True
            self.trail.append((int(cx), int(cy)))
            return (cx, cy, "measured")

        # Nothing matched -> coast on the prediction.
        self._coast += 1
        self._hits = 0
        if self._coast > self.cfg.max_coast_frames:
            self.reset()
            return None
        self.trail.append((int(px), int(py)))
        return (px, py, "predicted")

    @staticmethod
    def _closest_within(candidates: List[Detection], px: float, py: float,
                        gate: float) -> Optional[Detection]:
        best: Optional[Detection] = None
        best_d = gate
        for c in candidates:
            dist = float(np.hypot(c.cx - px, c.cy - py))
            if dist <= best_d:
                best_d = dist
                best = c
        return best
