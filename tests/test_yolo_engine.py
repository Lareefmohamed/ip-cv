"""YOLO engine integration test (skipped when ultralytics is not installed).

YOLO may score zero on the synthetic clip (drawn circles are not photo-real
sports balls), so this test asserts the FALLBACK chain: once a track exists,
classical-in-ROI + CSRT + Kalman must keep accuracy on the hard clip at or
above the floor the hybrid engine already guarantees.

Run directly:        python tests/test_yolo_engine.py
Or under pytest:     pytest tests/test_yolo_engine.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
    pytest.importorskip("ultralytics")
except ImportError:                     # running directly without pytest
    import importlib.util
    if importlib.util.find_spec("ultralytics") is None:
        print("ultralytics not installed - skipping YOLO engine test.")
        sys.exit(0)

from src.config import make_preset                           # noqa: E402
from src.pipeline import process_video_yolo                  # noqa: E402
from test_accuracy import (HARD, HARD_TOL, _ensure_hard,     # noqa: E402
                           _load_hard_gt, _score_log)


def test_yolo_engine_on_hard_clip():
    _ensure_hard()
    gt = _load_hard_gt()

    cfg = make_preset("yolo")
    cfg.detector.color = "red"
    # Synthetic circles are tiny; keep inference cheap for CI-ish runs.
    cfg.yolo.imgsz = 640
    stats = process_video_yolo(HARD, out_path=None, config=cfg, fill_gaps=True)

    score = _score_log(stats["track_log"], gt, HARD_TOL)
    # Even with zero genuine YOLO hits the classical/CSRT fallback must hold
    # the same floor the hybrid engine is tested against.
    assert score["accuracy_on_visible"] >= 0.80, (score, stats["outcomes"])
    assert score["mean_err_px"] <= HARD_TOL, score


if __name__ == "__main__":
    test_yolo_engine_on_hard_clip()
    print("YOLO engine test passed.")
