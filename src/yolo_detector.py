"""Deep-learning ball detector built on YOLOv8 (ultralytics).

Why this exists: the classical HSV+shape detector cannot tell a ball from any
other round, ball-coloured blob (helmets, gloves, logos).  YOLO is restricted
to COCO class 32 ("sports ball"), so by construction it never reports a
helmet or glove as the ball.  It also recognises motion-blurred balls that
fail the classical circularity test, which is what makes normal-speed
footage trackable.

ultralytics (and its torch dependency) is imported lazily so the classical
engines keep working without it installed.
"""

from __future__ import annotations

import importlib.util
from typing import List, Optional

import numpy as np

from .config import DetectorConfig, YoloConfig
from .detector import Detection


class YoloBallDetector:
    """Stateless per-frame YOLO detector returning ``Detection`` objects."""

    def __init__(self, config: Optional[YoloConfig] = None,
                 shape_cfg: Optional[DetectorConfig] = None):
        self.cfg = config or YoloConfig()
        self.shape_cfg = shape_cfg
        self._model = None  # lazy-loaded on first detect()

    @staticmethod
    def available() -> bool:
        """True when the ultralytics package is importable."""
        return importlib.util.find_spec("ultralytics") is not None

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "The YOLO engine needs the 'ultralytics' package. "
                    "Install it with:  pip install ultralytics") from exc
            self._model = YOLO(self.cfg.weights)
        return self._model

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run YOLO on one BGR frame; return ball candidates, best first.

        Candidates are mapped onto the classical ``Detection`` dataclass
        (circularity/fill_ratio = 1.0) so they flow through the existing
        Kalman gating and drawing code unchanged.
        """
        model = self._load()
        results = model.predict(
            frame_bgr,
            classes=list(self.cfg.classes),
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            max_det=self.cfg.max_det,
            device=self.cfg.device or None,
            verbose=False,
        )
        out: List[Detection] = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out
        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (cx, cy, w, h), conf in zip(xywh, confs):
            radius = float(max(w, h)) / 2.0
            # Radius bounds from the classical config are a cheap second
            # defence: a 200 px "sports ball" is a mislabel, not the ball.
            if self.shape_cfg is not None:
                if radius < self.shape_cfg.min_radius or \
                        radius > self.shape_cfg.max_radius:
                    continue
            out.append(Detection(
                cx=float(cx), cy=float(cy), radius=radius,
                area=float(w) * float(h),
                circularity=1.0, fill_ratio=1.0,
                motion_overlap=0.0, is_blur=False, conf=float(conf)))
        out.sort(key=lambda d: d.conf, reverse=True)
        return out
