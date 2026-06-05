"""Objective accuracy evaluation on the synthetic clip with ground truth.

Generates a synthetic clip (if not present), runs the pipeline, and compares
the tracked centroid against the known ground-truth centroid each frame.

A detection counts as CORRECT when it lands within ``tol`` pixels of the true
centre.  Reports detection rate, localisation accuracy (correct / detected),
overall accuracy, mean pixel error and processing FPS.

Run directly:        python tests/test_accuracy.py
Or under pytest:     pytest tests/test_accuracy.py
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PipelineConfig                      # noqa: E402
from src.pipeline import CricketBallPipeline, iter_video   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(HERE, "_sample.mp4")
GT = os.path.join(HERE, "_sample.gt.csv")
TOL = 12.0          # px - within ~1 ball radius counts as a correct hit


def _ensure_sample():
    if os.path.exists(SAMPLE) and os.path.exists(GT):
        return
    script = os.path.join(ROOT, "scripts", "make_sample_video.py")
    subprocess.run([sys.executable, script, "--out", SAMPLE, "--frames", "120"],
                   check=True, cwd=ROOT)


def _load_gt() -> dict[int, tuple[float, float]]:
    gt = {}
    with open(GT) as fh:
        for row in csv.DictReader(fh):
            gt[int(row["frame"])] = (float(row["x"]), float(row["y"]))
    return gt


def evaluate() -> dict:
    _ensure_sample()
    gt = _load_gt()

    cfg = PipelineConfig()
    cfg.detector.color = "red"
    pipe = CricketBallPipeline(cfg)

    total = detected = correct = 0
    err_sum = 0.0
    fps_sum = 0.0
    for frame in iter_video(SAMPLE):
        res = pipe.process_frame(frame)
        idx = res.index
        total += 1
        fps_sum += res.fps
        if res.ball is None or idx not in gt:
            continue
        detected += 1
        gx, gy = gt[idx]
        err = float(np.hypot(res.ball[0] - gx, res.ball[1] - gy))
        err_sum += err
        if err <= TOL:
            correct += 1

    return {
        "frames": total,
        "detected": detected,
        "correct": correct,
        "detection_rate": detected / total if total else 0.0,
        "localisation_acc": correct / detected if detected else 0.0,
        "overall_acc": correct / total if total else 0.0,
        "mean_err_px": err_sum / detected if detected else float("nan"),
        "mean_proc_fps": fps_sum / total if total else 0.0,
    }


def test_high_accuracy():
    """Pytest entry point: assert the project hits its accuracy targets."""
    m = evaluate()
    # Proposal targets: 85-90% detection, >=30 FPS. Synthetic clip is clean,
    # so we expect to comfortably exceed both.
    assert m["overall_acc"] >= 0.85, m
    assert m["mean_err_px"] <= TOL, m
    assert m["mean_proc_fps"] >= 30.0, m


def _print(m: dict):
    print("=" * 50)
    print("  Cricket Ball Tracker - accuracy evaluation")
    print("=" * 50)
    print(f"  Frames                : {m['frames']}")
    print(f"  Frames with detection : {m['detected']}")
    print(f"  Correct (<= {TOL:.0f}px)     : {m['correct']}")
    print(f"  Detection rate        : {m['detection_rate']:.1%}")
    print(f"  Localisation accuracy : {m['localisation_acc']:.1%}")
    print(f"  Overall accuracy      : {m['overall_acc']:.1%}")
    print(f"  Mean pixel error      : {m['mean_err_px']:.2f} px")
    print(f"  Mean processing speed : {m['mean_proc_fps']:.1f} FPS")
    print("=" * 50)


if __name__ == "__main__":
    _print(evaluate())
