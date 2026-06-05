"""End-to-end video processing pipeline.

Wires the detector, tracker and trajectory overlay together and exposes a
single ``process_frame`` call plus convenience helpers for whole-video
processing.  Both the CLI (``run.py``) and the Streamlit app build on this.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .detector import BallDetector, Detection
from .tracker import BallTracker
from .trajectory import draw_overlay


@dataclass
class FrameResult:
    """Everything produced for a single frame."""

    index: int
    frame: np.ndarray                                   # annotated BGR frame
    mask: np.ndarray                                    # final binary mask
    ball: Optional[Tuple[float, float, str]]            # (x, y, source)
    candidates: List[Detection]
    fps: float


class CricketBallPipeline:
    """Stateful, frame-by-frame cricket-ball tracking pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.detector = BallDetector(self.cfg.detector)
        self.tracker = BallTracker(self.cfg.tracker)
        self._last_t: Optional[float] = None
        self._fps_ema: float = 0.0
        self.frame_index = -1
        # Recorded measured centroids (for accuracy reporting / export).
        self.track_log: List[Tuple[int, float, float, str]] = []

    def reset(self) -> None:
        self.detector.reset()
        self.tracker.reset()
        self._last_t = None
        self._fps_ema = 0.0
        self.frame_index = -1
        self.track_log.clear()

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        self.frame_index += 1

        candidates, mask = self.detector.detect(frame_bgr)
        ball = self.tracker.update(candidates)

        # FPS (exponential moving average of processing time only).
        now = time.perf_counter()
        if self._last_t is not None:
            inst = 1.0 / max(now - self._last_t, 1e-6)
            self._fps_ema = inst if self._fps_ema == 0 else (
                0.9 * self._fps_ema + 0.1 * inst)
        self._last_t = now

        if ball is not None:
            self.track_log.append((self.frame_index, ball[0], ball[1], ball[2]))

        annotated = draw_overlay(
            frame_bgr, self.cfg, ball, self.tracker.last_radius,
            self.tracker.trail, fps=self._fps_ema or None,
            frame_idx=self.frame_index)

        return FrameResult(self.frame_index, annotated, mask, ball,
                           candidates, self._fps_ema)


# ---------------------------------------------------------------------------
# Whole-video helpers
# ---------------------------------------------------------------------------

def iter_video(path: str) -> Iterator[np.ndarray]:
    """Yield BGR frames from a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def video_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def process_video(in_path: str,
                  out_path: Optional[str] = None,
                  config: Optional[PipelineConfig] = None,
                  progress: Optional[Callable[[int, int], None]] = None
                  ) -> dict:
    """Process a whole video, optionally writing an annotated copy.

    Returns a stats dict: detection rate, mean processing FPS, frame count.
    """
    meta = video_meta(in_path)
    pipe = CricketBallPipeline(config)

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, meta["fps"] or 30.0,
                                 (meta["width"], meta["height"]))

    measured = 0
    total = 0
    fps_sum = 0.0
    try:
        for frame in iter_video(in_path):
            res = pipe.process_frame(frame)
            total += 1
            fps_sum += res.fps
            if res.ball is not None and res.ball[2] == "measured":
                measured += 1
            if writer is not None:
                writer.write(res.frame)
            if progress is not None:
                progress(total, meta["frames"])
    finally:
        if writer is not None:
            writer.release()

    return {
        "frames": total,
        "measured_detections": measured,
        "detection_rate": (measured / total) if total else 0.0,
        "mean_proc_fps": (fps_sum / total) if total else 0.0,
        "video_fps": meta["fps"],
        "out_path": out_path,
        "track_log": pipe.track_log,
    }


def process_video_assisted(in_path: str,
                           bbox: Optional[Tuple[int, int, int, int]] = None,
                           start_frame: int = 0,
                           out_path: Optional[str] = None,
                           config: Optional[PipelineConfig] = None,
                           reacquire: bool = True,
                           progress: Optional[Callable[[int, int], None]] = None
                           ) -> dict:
    """Process a video with the CSRT assisted tracker.

    If ``bbox`` is given it is used to initialise CSRT on ``start_frame``
    (manual mode).  If ``bbox`` is None the HSV detector auto-seeds the track
    (auto mode).  Returns the same stats dict shape as ``process_video``.
    """
    # Local import avoids a hard CSRT dependency for the automatic pipeline.
    from .assisted_tracker import AssistedTracker

    meta = video_meta(in_path)
    cfg = config or PipelineConfig()
    tracker = AssistedTracker(cfg, reacquire=reacquire)

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, meta["fps"] or 30.0,
                                 (meta["width"], meta["height"]))

    measured = total = 0
    fps_sum = 0.0
    track_log: List[Tuple[int, float, float, str]] = []
    seeded = False
    try:
        for idx, frame in enumerate(iter_video(in_path)):
            if idx < start_frame:
                continue
            t0 = time.perf_counter()

            if not seeded:
                if bbox is not None:
                    seeded = tracker.init_with_bbox(frame, bbox)
                    ball = (bbox[0] + bbox[2] / 2.0,
                            bbox[1] + bbox[3] / 2.0, "measured") if seeded else None
                else:
                    ball = tracker.update(frame)   # auto_seed attempts inside
                    seeded = tracker.initialised
            else:
                ball = tracker.update(frame)

            dt = time.perf_counter() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_sum += fps
            total += 1
            if ball is not None and ball[2] == "measured":
                measured += 1
            if ball is not None:
                track_log.append((idx, ball[0], ball[1], ball[2]))

            annotated = draw_overlay(frame, cfg, ball, tracker.last_radius,
                                     tracker.trail, fps=fps, frame_idx=idx)
            if writer is not None:
                writer.write(annotated)
            if progress is not None:
                progress(total, meta["frames"])
    finally:
        if writer is not None:
            writer.release()

    return {
        "frames": total,
        "measured_detections": measured,
        "detection_rate": (measured / total) if total else 0.0,
        "mean_proc_fps": (fps_sum / total) if total else 0.0,
        "video_fps": meta["fps"],
        "out_path": out_path,
        "track_log": track_log,
        "mode": "assisted-manual" if bbox is not None else "assisted-auto",
    }
