"""YOLO hybrid cricket-ball tracker: YOLOv8 + classical detector + CSRT + Kalman.

The most accurate engine.  YOLO provides ball-only detections (COCO class 32),
so helmets / gloves / pads can never be picked up; the classical detector and
CSRT primarily ASSIST an existing YOLO-seeded track.  Only if YOLO finds
nothing for ``classical_seed_after`` frames may the classical detector seed a
track (moving-blob rules) -- and a strong YOLO detection elsewhere re-seeds
the track immediately if that pick was wrong.

Per-frame decision order:
    1. Kalman predicts (px, py).
    2. YOLO detects.  With a track: nearest detection within the gate is
       accepted at any confidence (a weak detection sitting exactly on the
       prediction is trustworthy -- this is what keeps the lock through
       motion blur at normal speed)                              -> "yolo".
       Without a track: only a detection with conf >= seed_conf may START
       a track (prevents low-confidence false positives seeding).
    3. Else the classical HSV+motion detector runs ROI-guided around the
       prediction (existing tracks only)                         -> "measured".
    4. Else CSRT update; trusted only if it stays near the prediction
                                                                 -> "csrt".
    5. Else coast on the Kalman prediction (gap-fill)            -> "filled".
       After ``max_coast_frames`` with nothing, the track resets.

The return shape ``(x, y, source)`` and the ``.trail`` / ``.trail_t`` /
``.last_radius`` attributes match HybridTracker, so it drops straight into
``draw_overlay`` and the pipeline helpers.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from .assisted_tracker import _make_csrt, detection_to_bbox
from .config import PipelineConfig
from .detector import BallDetector, Detection
from .tracker import _KalmanCV
from .yolo_detector import YoloBallDetector

# Per-frame outcome labels (used for --diagnose).
OUTCOMES = ("yolo", "measured", "csrt", "filled", "lost")


class YoloHybridTracker:
    """YOLO + classical detector + CSRT + Kalman fusion with gap-filling."""

    def __init__(self, config: Optional[PipelineConfig] = None,
                 fill_gaps: bool = True):
        self.cfg = config or PipelineConfig()
        self.fill_gaps = fill_gaps
        self.yolo = YoloBallDetector(self.cfg.yolo, self.cfg.detector)
        self.detector = BallDetector(self.cfg.detector)   # ROI fallback only

        self._kf: Optional[_KalmanCV] = None
        self._csrt = None
        self._coast = 0
        self._hits = 0
        self.confirmed = False
        self.last_radius: float = 8.0
        self.trail: Deque[Tuple[int, int]] = deque(
            maxlen=self.cfg.tracker.trail_length)
        self.trail_t: Deque[int] = deque(maxlen=self.cfg.tracker.trail_length)
        self._frame_idx = -1
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
        self.trail_t.clear()

    # -- per frame ----------------------------------------------------------

    def update(self, frame: np.ndarray
               ) -> Optional[Tuple[float, float, str]]:
        cfg_t = self.cfg.tracker
        self._frame_idx += 1

        # --- 1. Kalman prediction (None until a track exists) -------------
        pred: Optional[Tuple[float, float]] = None
        roi = None
        if self._kf is not None:
            pred = self._kf.predict()
            gate = cfg_t.max_match_dist * (1.0 + 0.5 * self._coast)
            roi = (pred[0], pred[1], gate)

        # --- 2. Classical detector first: keeps MOG2 warm and produces the
        #        motion mask used to vet YOLO seeds on THIS frame. -----------
        candidates, _ = self.detector.detect(frame, roi=roi)
        self._accumulate_rejects()

        # --- 3. YOLO: ball-class-only detections ---------------------------
        yolo_dets = self.yolo.detect(frame)
        match = self._pick_yolo(yolo_dets, pred)
        if match is not None:
            return self._accept_measurement(frame, match, "yolo")

        # A strong, MOVING YOLO hit away from a coasting / unconfirmed track
        # means the current track is probably wrong (e.g. a classical seed
        # locked onto the wrong object) -- drop it and follow YOLO's ball.
        if pred is not None and (self._coast >= 1 or not self.confirmed):
            strong = [d for d in yolo_dets if self._may_seed(d)]
            if strong:
                best = max(strong, key=lambda d: d.conf)
                self.reset()
                return self._accept_measurement(frame, best, "yolo")

        # --- 4. Classical detector: ROI assist, or last-resort seeding -----
        if pred is not None:
            cand = self._pick_classical(candidates, pred)
            if cand is not None:
                return self._accept_measurement(frame, cand, "measured")
        elif self._frame_idx >= self.cfg.yolo.classical_seed_after:
            # Safety net: if YOLO has found nothing for a while, let the
            # classical detector seed (moving-blob rules).  A strong YOLO
            # detection later re-seeds if this picks the wrong object.
            cand = self._seed_classical(candidates)
            if cand is not None:
                return self._accept_measurement(frame, cand, "measured")

        # --- 4. No detection: try CSRT lock --------------------------------
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

        # --- 5. Nothing: gap-fill on the prediction (or give up) -----------
        if self._kf is not None and pred is not None:
            self._coast += 1
            self._hits = 0
            if self._coast <= cfg_t.max_coast_frames and self.fill_gaps:
                self.trail.append((int(pred[0]), int(pred[1])))
                self.trail_t.append(self._frame_idx)
                self.outcomes["filled"] += 1
                return (pred[0], pred[1], "filled")
            if self._coast > cfg_t.max_coast_frames:
                self.reset()
        self.outcomes["lost"] += 1
        return None

    # -- helpers ------------------------------------------------------------

    def _is_moving(self, d: Detection, min_overlap: float = 0.05) -> bool:
        """True when the detection sits on MOG2-flagged motion (or when no
        motion information is available -- gate disabled / still warming).

        This is what stops YOLO from seeding a track on a STATIC ball-like
        object (spare ball by the pitch, ball logo on a board): the real
        delivery is always moving."""
        mask = self.detector.last_motion_mask
        if mask is None:
            return not self.cfg.detector.use_motion or \
                self.detector._frame_count > self.cfg.detector.warmup_frames
        h, w = mask.shape[:2]
        r = max(int(d.radius), 3)
        x0, x1 = max(0, int(d.cx) - r), min(w, int(d.cx) + r + 1)
        y0, y1 = max(0, int(d.cy) - r), min(h, int(d.cy) + r + 1)
        if x0 >= x1 or y0 >= y1:
            return False
        sub = mask[y0:y1, x0:x1]
        return float(np.count_nonzero(sub)) / sub.size >= min_overlap

    def _is_ball_coloured(self, d: Detection,
                          min_overlap: float = 0.10) -> bool:
        """True when the detection's disc overlaps the configured ball-colour
        HSV mask.  YOLO knows "sports ball" but not which ball -- a moving
        white shoe or a second white object can fool it; the colour profile
        (red / white / pink, or a calibrated custom range) settles it."""
        mask = self.detector.last_color_mask
        if mask is None:
            return True
        h, w = mask.shape[:2]
        r = max(int(d.radius), 3)
        x0, x1 = max(0, int(d.cx) - r), min(w, int(d.cx) + r + 1)
        y0, y1 = max(0, int(d.cy) - r), min(h, int(d.cy) + r + 1)
        if x0 >= x1 or y0 >= y1:
            return False
        sub = mask[y0:y1, x0:x1]
        return float(np.count_nonzero(sub)) / sub.size >= min_overlap

    def _may_seed(self, d: Detection) -> bool:
        return (d.conf >= self.cfg.yolo.seed_conf and self._is_moving(d)
                and self._is_ball_coloured(d))

    def _pick_yolo(self, dets: List[Detection],
                   pred: Optional[Tuple[float, float]]
                   ) -> Optional[Detection]:
        if not dets:
            return None
        if pred is None:
            # Seeding a NEW track: demand real confidence AND motion AND the
            # right colour, so a stray weak detection, a static ball-like
            # object (spare ball, logo) or a moving white shoe can't start a
            # track on the wrong thing.
            seeds = [d for d in dets if self._may_seed(d)]
            return max(seeds, key=lambda d: d.conf) if seeds else None
        # Continuing a track: nearest in-gate detection wins.  Any confidence
        # is fine -- proximity to the prediction is the evidence.
        gate = self.cfg.tracker.max_match_dist * (1.0 + 0.5 * self._coast)
        in_gate = [(float(np.hypot(d.cx - pred[0], d.cy - pred[1])), d)
                   for d in dets]
        in_gate = [(dist, d) for dist, d in in_gate if dist <= gate]
        if not in_gate:
            return None
        return min(in_gate, key=lambda dd: dd[0])[1]

    def _seed_classical(self, candidates: List[Detection]
                        ) -> Optional[Detection]:
        """Hybrid-style seeding: only a MOVING blob may start a track, and
        only once the MOG2 background model is warm."""
        if not candidates:
            return None
        cfg_d = self.cfg.detector
        if cfg_d.use_motion:
            if self.detector._frame_count <= cfg_d.warmup_frames:
                return None
            movers = [c for c in candidates if c.motion_overlap > 0.05]
            if not movers:
                return None
            return max(movers, key=lambda c: c.score())
        return candidates[0]

    def _pick_classical(self, candidates: List[Detection],
                        pred: Tuple[float, float]) -> Optional[Detection]:
        """Nearest in-gate classical blob; moving blobs preferred.

        Never called without a prediction -- the classical detector is not
        allowed to seed tracks in this engine (that is YOLO's job, which is
        what keeps helmets / gloves out).
        """
        if not candidates:
            return None
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
                            source: str) -> Tuple[float, float, str]:
        if self._kf is None:
            self._kf = _KalmanCV(d.cx, d.cy)
            cx, cy = d.cx, d.cy
        else:
            cx, cy = self._kf.correct(d.cx, d.cy)
        self.last_radius = d.radius
        self._after_hit(cx, cy, correct=False)
        # (Re)seed CSRT from the fresh detection box.
        self._seed_csrt(frame, d)
        self.outcomes[source] += 1
        return (cx, cy, source)

    def _after_hit(self, cx: float, cy: float, correct: bool) -> None:
        if correct and self._kf is not None:
            self._kf.correct(cx, cy)
        self._coast = 0
        self._hits += 1
        if self._hits >= self.cfg.tracker.min_hits_to_confirm:
            self.confirmed = True
        self.trail.append((int(cx), int(cy)))
        self.trail_t.append(self._frame_idx)

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
