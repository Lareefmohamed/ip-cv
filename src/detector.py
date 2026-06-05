"""Cricket ball detector built from traditional computer-vision primitives.

Pipeline (per frame):
    1. Gaussian blur               -> reduce sensor / compression noise
    2. BGR -> HSV conversion       -> separate colour (H) from brightness (V)
    3. HSV colour thresholding     -> binary colour mask of the ball's colour
    4. MOG2 background subtraction -> binary motion mask (the moving ball)
    5. mask = colour AND motion    -> suppress static same-coloured clutter
    6. morphological open + close  -> remove speckle, fill the blob
    7. contour extraction          -> candidate blobs
    8. geometric filtering         -> keep only round, ball-sized blobs
    9. centroid via image moments  -> sub-pixel (x, y) for the best candidate

Returning *all* surviving candidates (not just one) lets the Kalman tracker
pick the one that is consistent with the ball's motion, which is what gives
the system its high real-world accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .config import DetectorConfig


@dataclass
class Detection:
    """A single ball candidate found in one frame."""

    cx: float            # centroid x (sub-pixel, from image moments)
    cy: float            # centroid y
    radius: float        # min-enclosing-circle radius (px)
    area: float          # contour area (px^2)
    circularity: float   # 4*pi*A / P^2  in [0, 1]
    fill_ratio: float    # contour area / enclosing-circle area in [0, 1]

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    def score(self) -> float:
        """Higher == more ball-like.  Used to rank candidates when the tracker
        has no prior (e.g. the very first frame)."""
        return self.circularity * self.fill_ratio


class BallDetector:
    """Stateful detector.  Holds the MOG2 background model across frames."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()
        self._bg = None
        if self.cfg.use_motion:
            self._init_bg()

    # -- public API ---------------------------------------------------------

    def reset(self) -> None:
        """Forget the learned background (call when starting a new video)."""
        if self.cfg.use_motion:
            self._init_bg()

    def detect(self, frame_bgr: np.ndarray) -> tuple[List[Detection], np.ndarray]:
        """Return (candidates, debug_mask) for one BGR frame.

        ``debug_mask`` is the final binary mask, handy for the UI / tuning.
        """
        cfg = self.cfg

        # 1. Noise reduction.
        if cfg.blur_ksize and cfg.blur_ksize >= 3:
            k = cfg.blur_ksize | 1  # force odd
            blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
        else:
            blurred = frame_bgr

        # 2 + 3. HSV colour mask.
        color_mask = self._color_mask(blurred)

        # 4 + 5. Motion gate (optional but on by default).
        if cfg.use_motion and self._bg is not None:
            motion = self._bg.apply(blurred)
            # Drop MOG2 shadow pixels (value 127) -> keep only hard foreground.
            motion = cv2.threshold(motion, 200, 255, cv2.THRESH_BINARY)[1]
            if cfg.motion_dilate > 0:
                mk = np.ones((cfg.motion_dilate, cfg.motion_dilate), np.uint8)
                motion = cv2.dilate(motion, mk, iterations=1)
            mask = cv2.bitwise_and(color_mask, motion)
        else:
            mask = color_mask

        # 6. Morphological cleaning.
        mask = self._clean(mask)

        # 7 + 8 + 9. Contours -> filter -> centroids.
        candidates = self._contours_to_detections(mask)
        return candidates, mask

    # -- internals ----------------------------------------------------------

    def _init_bg(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.mog2_history,
            varThreshold=self.cfg.mog2_var_threshold,
            detectShadows=True,
        )

    def _color_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = None
        for lower, upper in self.cfg.ranges():
            part = cv2.inRange(hsv, np.array(lower, np.uint8),
                               np.array(upper, np.uint8))
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        return mask

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        k = cfg.morph_ksize | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # Opening (erode then dilate) removes isolated speckle.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                iterations=cfg.erode_iter)
        # Closing (dilate then erode) fills internal gaps -> solid blob.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                iterations=cfg.dilate_iter)
        return mask

    def _contours_to_detections(self, mask: np.ndarray) -> List[Detection]:
        cfg = self.cfg
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg.min_area or area > cfg.max_area:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < cfg.min_circularity:
                continue

            (ex, ey), radius = cv2.minEnclosingCircle(c)
            if radius < cfg.min_radius or radius > cfg.max_radius:
                continue

            circle_area = math.pi * radius * radius
            fill_ratio = area / circle_area if circle_area > 0 else 0.0
            if fill_ratio < cfg.min_fill_ratio:
                continue

            # Centroid via image moments (sub-pixel, more stable than the
            # enclosing-circle centre for slightly irregular blobs).
            m = cv2.moments(c)
            if m["m00"] == 0:
                cx, cy = ex, ey
            else:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]

            out.append(Detection(cx, cy, radius, area, circularity, fill_ratio))

        # Best (most ball-like) first.
        out.sort(key=lambda d: d.score(), reverse=True)
        return out
