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
    motion_overlap: float = 0.0   # fraction of blob pixels flagged as motion
    is_blur: bool = False         # accepted as an elongated motion-blur streak

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    def score(self) -> float:
        """Higher == more ball-like.  Used to rank candidates when the tracker
        has no prior (e.g. the very first frame).  Motion overlap is a gentle
        boost so a moving blob outranks a static same-coloured one."""
        base = self.circularity * self.fill_ratio
        return base * (1.0 + 0.5 * self.motion_overlap)


class BallDetector:
    """Stateful detector.  Holds the MOG2 background model across frames."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()
        self._bg = None
        self._frame_count = 0
        # Per-frame tally of why contours were rejected (for --diagnose).
        self.last_stats: dict[str, int] = {}
        if self.cfg.use_motion:
            self._init_bg()

    # -- public API ---------------------------------------------------------

    def reset(self) -> None:
        """Forget the learned background (call when starting a new video)."""
        self._frame_count = 0
        if self.cfg.use_motion:
            self._init_bg()

    def detect(self, frame_bgr: np.ndarray,
               roi: Optional[tuple[float, float, float]] = None
               ) -> tuple[List[Detection], np.ndarray]:
        """Return (candidates, debug_mask) for one BGR frame.

        ``roi`` is an optional ``(cx, cy, radius)`` search window (usually the
        tracker's prediction); blobs inside it are judged with relaxed
        thresholds for higher recall.  ``debug_mask`` is the final binary mask.
        """
        cfg = self.cfg
        self._frame_count += 1

        # 1. Noise reduction.
        if cfg.blur_ksize and cfg.blur_ksize >= 3:
            k = cfg.blur_ksize | 1  # force odd
            blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
        else:
            blurred = frame_bgr

        # 2 + 3. HSV colour mask.
        color_mask = self._color_mask(blurred)

        # 4 + 5. Motion gate.  MOG2 needs a few frames to learn the background,
        # so during warm-up we fall back to colour-only.
        motion = None
        warm = self._frame_count > cfg.warmup_frames
        if cfg.use_motion and self._bg is not None:
            motion = self._bg.apply(blurred)
            # Drop MOG2 shadow pixels (value 127) -> keep only hard foreground.
            motion = cv2.threshold(motion, 200, 255, cv2.THRESH_BINARY)[1]
            if cfg.motion_dilate > 0:
                mk = np.ones((cfg.motion_dilate, cfg.motion_dilate), np.uint8)
                motion = cv2.dilate(motion, mk, iterations=1)

        if cfg.use_motion and motion is not None and not cfg.soft_motion:
            # Hard gate (default): blob must be the right colour AND moving.
            mask = cv2.bitwise_and(color_mask, motion)
            motion_for_score = motion
        elif cfg.use_motion and cfg.soft_motion:
            # Soft gate: keep all colour blobs; motion only *ranks* them (and is
            # ignored until MOG2 is warm).  Far higher recall on fast balls.
            mask = color_mask
            motion_for_score = motion if warm else None
        else:
            # Motion gating disabled entirely.
            mask = color_mask
            motion_for_score = None

        # 6. Morphological cleaning.
        mask = self._clean(mask)

        # 7 + 8 + 9. Contours -> filter -> centroids.
        candidates = self._contours_to_detections(mask, motion_for_score, roi)
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

    @staticmethod
    def _in_roi(cx: float, cy: float,
                roi: Optional[tuple[float, float, float]]) -> bool:
        if roi is None:
            return False
        rx, ry, rr = roi
        return (cx - rx) ** 2 + (cy - ry) ** 2 <= rr * rr

    def _motion_overlap(self, c, motion: Optional[np.ndarray],
                        area: float) -> float:
        """Fraction of the contour's pixels that are flagged as motion."""
        if motion is None or area <= 0:
            return 0.0
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            return 0.0
        blob = np.zeros((h, w), np.uint8)
        cv2.drawContours(blob, [c], -1, 255, -1, offset=(-x, -y))
        sub = motion[y:y + h, x:x + w]
        inter = cv2.bitwise_and(blob, sub)
        blob_px = int(cv2.countNonZero(blob))
        return (cv2.countNonZero(inter) / blob_px) if blob_px else 0.0

    def _contours_to_detections(self, mask: np.ndarray,
                                motion: Optional[np.ndarray] = None,
                                roi: Optional[tuple[float, float, float]] = None
                                ) -> List[Detection]:
        cfg = self.cfg
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        stats = {"total": len(contours), "too_small": 0, "shape_rejected": 0,
                 "bad_radius": 0, "accepted": 0, "accepted_blur": 0}
        out: List[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            m = cv2.moments(c)
            if m["m00"] != 0:
                cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            else:
                cx, cy = float(c[:, 0, 0].mean()), float(c[:, 0, 1].mean())
            in_roi = self._in_roi(cx, cy, roi)

            # ROI blobs get relaxed thresholds (recall where the ball is due).
            min_area = cfg.roi_min_area if in_roi else cfg.min_area
            min_circ = cfg.roi_min_circularity if in_roi else cfg.min_circularity
            min_fill = cfg.roi_min_fill_ratio if in_roi else cfg.min_fill_ratio

            if area < min_area or area > cfg.max_area:
                stats["too_small"] += 1
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                stats["shape_rejected"] += 1
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)

            (ex, ey), radius = cv2.minEnclosingCircle(c)
            if radius < cfg.min_radius or radius > cfg.max_radius:
                stats["bad_radius"] += 1
                continue
            circle_area = math.pi * radius * radius
            fill_ratio = area / circle_area if circle_area > 0 else 0.0

            is_blur = False
            if circularity < min_circ or fill_ratio < min_fill:
                # Maybe it's a motion-blur streak: convex/solid + sized right.
                hull_area = cv2.contourArea(cv2.convexHull(c))
                solidity = area / hull_area if hull_area > 0 else 0.0
                accept_blur = (cfg.accept_blur or in_roi) and \
                    solidity >= cfg.min_solidity
                if not accept_blur:
                    stats["shape_rejected"] += 1
                    continue
                is_blur = True

            overlap = self._motion_overlap(c, motion, area)
            out.append(Detection(cx, cy, radius, area, circularity,
                                  fill_ratio, overlap, is_blur))
            stats["accepted"] += 1
            if is_blur:
                stats["accepted_blur"] += 1

        self.last_stats = stats
        # Best (most ball-like) first.
        out.sort(key=lambda d: d.score(), reverse=True)
        return out
