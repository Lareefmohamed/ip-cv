"""Hybrid cricket-ball tracker: detector + CSRT + Kalman, with gap-filling.

This is the most robust engine and the recommended default for general real
footage.  Each frame it fuses three classical techniques so that a miss in any
one is covered by the others:

    * HSV+motion DETECTOR  - accurate, ROI-guided around the prediction.
    * CSRT TRACKER         - texture lock; carries frames the detector misses
                             (white ball, motion blur, clutter).
    * KALMAN FILTER        - smooths the path and *fills* short gaps with a
                             prediction so the trajectory stays continuous.

Per-frame decision order:
    1. Kalman predicts (px, py).
    2. Detector runs ROI-guided around (px, py); pick the candidate nearest the
       prediction within the gate  -> "measured".
    3. Else CSRT update; if it locks near the prediction               -> "csrt".
    4. Else coast on the Kalman prediction (gap-fill)                   -> "filled".
       After ``max_coast_frames`` with nothing, the track resets.
    5. A confident detection re-seeds CSRT so it never drifts for long.

The return shape ``(x, y, source)`` and the ``.trail`` / ``.last_radius``
attributes match BallTracker, so it drops straight into ``draw_overlay``.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from .assisted_tracker import _make_csrt, detection_to_bbox
from .config import PipelineConfig
from .detector import BallDetector, Detection
from .tracker import _KalmanCV

# Per-frame outcome labels (used for --diagnose).
OUTCOMES = ("measured", "csrt", "filled", "lost")


class HybridTracker:
    """Detector + CSRT + Kalman fusion with smooth gap-filling."""

    def __init__(self, config: Optional[PipelineConfig] = None,
                 fill_gaps: bool = True):
        self.cfg = config or PipelineConfig()
        self.fill_gaps = fill_gaps
        self.detector = BallDetector(self.cfg.detector)

        self._kf: Optional[_KalmanCV] = None
        self._csrt = None
        self._coast = 0
        self._hits = 0
        self.confirmed = False
        self.last_radius: float = 8.0
        self.trail: Deque[Tuple[int, int]] = deque(
            maxlen=self.cfg.tracker.trail_length)
        # Diagnostics: counts of each per-frame outcome + detector reject sums.
        self.outcomes: dict[str, int] = {k: 0 for k in OUTCOMES}
        self.reject_totals: dict[str, int] = {}

    def reset(self) -> None:
        self.detector.reset()
        self._kf = None
        self._csrt = None
        self._coast = 0
        self._hits = 0
        self.confirmed = False
        self.trail.clear()

    # -- per frame ----------------------------------------------------------

    def update(self, frame: np.ndarray
               ) -> Optional[Tuple[float, float, str]]:
        cfg_t = self.cfg.tracker

        # --- 1. Kalman prediction (None until a track exists) -------------
        pred: Optional[Tuple[float, float]] = None
        roi = None
        if self._kf is not None:
            pred = self._kf.predict()
            gate = cfg_t.max_match_dist * (1.0 + 0.5 * self._coast)
            roi = (pred[0], pred[1], gate)

        # --- 2. Detector (ROI-guided once we have a prediction) -----------
        candidates, _ = self.detector.detect(frame, roi=roi)
        self._accumulate_rejects()

        match = self._pick(candidates, pred)
        if match is not None:
            return self._accept_measurement(frame, match, pred)

        # --- 3. No detection: try CSRT lock -------------------------------
        if self._csrt is not None:
            ok, box = self._csrt.update(frame)
            if ok:
                x, y, w, h = box
                cx, cy = x + w / 2.0, y + h / 2.0
                # Trust CSRT only if it stays near the Kalman prediction.
                if pred is None or np.hypot(cx - pred[0], cy - pred[1]) <= \
                        cfg_t.max_match_dist * (1.5 + self._coast):
                    self.last_radius = max(w, h) / 2.0
                    self._after_hit(cx, cy, correct=True)
                    self.outcomes["csrt"] += 1
                    return (cx, cy, "csrt")

        # --- 4. Nothing: gap-fill on the prediction (or give up) ----------
        if self._kf is not None and pred is not None:
            self._coast += 1
            self._hits = 0
            if self._coast <= cfg_t.max_coast_frames and self.fill_gaps:
                self.trail.append((int(pred[0]), int(pred[1])))
                self.outcomes["filled"] += 1
                return (pred[0], pred[1], "filled")
            if self._coast > cfg_t.max_coast_frames:
                self.reset()
        self.outcomes["lost"] += 1
        return None

    # -- helpers ------------------------------------------------------------

    def _pick(self, candidates: List[Detection],
              pred: Optional[Tuple[float, float]]) -> Optional[Detection]:
        if not candidates:
            return None
        if pred is None:
            # Seeding a NEW track: start on a *moving* blob so we never lock
            # onto static same-coloured clutter (boards, caps).  Only require
            # motion once MOG2 is warm and motion gating is enabled; otherwise
            # fall back to the best-scoring blob.
            cfg_d = self.cfg.detector
            if cfg_d.use_motion:
                warm = self.detector._frame_count > cfg_d.warmup_frames
                if not warm:
                    return None     # wait for MOG2 before seeding (avoid clutter)
                movers = [c for c in candidates if c.motion_overlap > 0.05]
                if not movers:
                    return None     # nothing moving yet -> don't seed on clutter
                return max(movers, key=lambda c: c.score())
            return candidates[0]
        # Continuing a track: take the nearest candidate within the gate, but
        # prefer *moving* blobs so the lock isn't stolen by static red clutter
        # that happens to sit near the predicted path.
        gate = self.cfg.tracker.max_match_dist * (1.0 + 0.5 * self._coast)
        in_gate = [(float(np.hypot(c.cx - pred[0], c.cy - pred[1])), c)
                   for c in candidates]
        in_gate = [(d, c) for d, c in in_gate if d <= gate]
        if not in_gate:
            return None
        movers = [(d, c) for d, c in in_gate if c.motion_overlap > 0.05]
        pool = movers if movers else in_gate
        return min(pool, key=lambda dc: dc[0])[1]

    def _accept_measurement(self, frame, d: Detection,
                            pred) -> Tuple[float, float, str]:
        if self._kf is None:
            self._kf = _KalmanCV(d.cx, d.cy)
            cx, cy = d.cx, d.cy
        else:
            cx, cy = self._kf.correct(d.cx, d.cy)
        self.last_radius = d.radius
        self._after_hit(cx, cy, correct=False)
        # (Re)seed CSRT from the fresh, confident detection box.
        self._seed_csrt(frame, d)
        self.outcomes["measured"] += 1
        return (cx, cy, "measured")

    def _after_hit(self, cx: float, cy: float, correct: bool) -> None:
        if correct and self._kf is not None:
            self._kf.correct(cx, cy)
        self._coast = 0
        self._hits += 1
        if self._hits >= self.cfg.tracker.min_hits_to_confirm:
            self.confirmed = True
        self.trail.append((int(cx), int(cy)))

    def _seed_csrt(self, frame, d: Detection) -> None:
        bbox = detection_to_bbox(d)
        h, w = frame.shape[:2]
        x = max(0, min(int(bbox[0]), w - 2))
        y = max(0, min(int(bbox[1]), h - 2))
        bw = max(2, min(int(bbox[2]), w - x))
        bh = max(2, min(int(bbox[3]), h - y))
        try:
            self._csrt = _make_csrt()
            self._csrt.init(frame, (x, y, bw, bh))
        except Exception:
            self._csrt = None

    def _accumulate_rejects(self) -> None:
        for k, v in self.detector.last_stats.items():
            self.reject_totals[k] = self.reject_totals.get(k, 0) + v
