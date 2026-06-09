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


# ---------------------------------------------------------------------------
# Hard clip: fast, motion-blurred ball with camera jitter + brief occlusion.
# Compares the Automatic engine against the Hybrid engine on the same footage.
# ---------------------------------------------------------------------------
HARD = os.path.join(HERE, "_hard.mp4")
HARD_GT = os.path.join(HERE, "_hard.gt.csv")
HARD_TOL = 18.0     # px - a fast/blurred ball is larger & harder to pin down


def _ensure_hard():
    if os.path.exists(HARD) and os.path.exists(HARD_GT):
        return
    script = os.path.join(ROOT, "scripts", "make_sample_video.py")
    subprocess.run([sys.executable, script, "--out", HARD, "--frames", "120",
                    "--hard"], check=True, cwd=ROOT)


def _load_hard_gt():
    """Return {frame: (x, y, occluded)}."""
    gt = {}
    with open(HARD_GT) as fh:
        for row in csv.DictReader(fh):
            gt[int(row["frame"])] = (float(row["x"]), float(row["y"]),
                                     bool(int(row.get("occluded", 0))))
    return gt


def _score_log(track_log, gt, tol):
    """Score a [(frame,x,y,source)] log against ground truth (visible frames)."""
    visible = {f for f, (_, _, occ) in gt.items() if not occ}
    by_frame = {f: (x, y) for f, x, y, _ in track_log}
    correct = err_sum = n_err = 0
    for f in visible:
        if f not in by_frame:
            continue
        gx, gy, _ = gt[f]
        err = float(np.hypot(by_frame[f][0] - gx, by_frame[f][1] - gy))
        err_sum += err
        n_err += 1
        if err <= tol:
            correct += 1
    return {
        "visible": len(visible),
        "matched": n_err,
        "correct": correct,
        "accuracy_on_visible": correct / len(visible) if visible else 0.0,
        "mean_err_px": err_sum / n_err if n_err else float("nan"),
    }


def evaluate_hard() -> dict:
    """Run Automatic vs Hybrid on the hard clip; return both scorecards."""
    from src.config import make_preset
    from src.pipeline import process_video_hybrid

    _ensure_hard()
    gt = _load_hard_gt()

    # Automatic engine (per-frame, measured only).
    auto_cfg = PipelineConfig()
    auto_cfg.detector.color = "red"
    pipe = CricketBallPipeline(auto_cfg)
    auto_log = []
    for frame in iter_video(HARD):
        res = pipe.process_frame(frame)
        if res.ball is not None and res.ball[2] == "measured":
            auto_log.append((res.index, res.ball[0], res.ball[1], "measured"))

    # Hybrid engine.
    hy_cfg = make_preset("hybrid")
    hy_cfg.detector.color = "red"
    hy = process_video_hybrid(HARD, out_path=None, config=hy_cfg, fill_gaps=True)

    return {
        "automatic": _score_log(auto_log, gt, HARD_TOL),
        "hybrid": _score_log(hy["track_log"], gt, HARD_TOL),
        "hybrid_stats": hy,
    }


def test_hybrid_improves_coverage():
    """The hybrid engine must track the hard clip far better than Automatic."""
    r = evaluate_hard()
    auto = r["automatic"]["accuracy_on_visible"]
    hyb = r["hybrid"]["accuracy_on_visible"]
    assert hyb > auto, (auto, hyb)
    assert hyb >= 0.80, r["hybrid"]
    assert r["hybrid"]["mean_err_px"] <= HARD_TOL, r["hybrid"]
    assert r["hybrid_stats"]["mean_proc_fps"] >= 30.0, r["hybrid_stats"]


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


def _print_hard(r: dict):
    a, h, s = r["automatic"], r["hybrid"], r["hybrid_stats"]
    print("=" * 50)
    print("  HARD clip (fast + blur + jitter + occlusion)")
    print("=" * 50)
    print(f"  Visible frames        : {h['visible']}")
    print(f"  Automatic accuracy    : {a['accuracy_on_visible']:.1%}  "
          f"(err {a['mean_err_px']:.1f}px)")
    print(f"  Hybrid    accuracy    : {h['accuracy_on_visible']:.1%}  "
          f"(err {h['mean_err_px']:.1f}px)")
    print(f"  Hybrid detection rate : {s['detection_rate']:.1%}")
    print(f"  Hybrid coverage       : {s['coverage']:.1%}")
    print(f"  Hybrid speed          : {s['mean_proc_fps']:.1f} FPS")
    print(f"  Hybrid outcomes       : {s['outcomes']}")
    print("=" * 50)


if __name__ == "__main__":
    _print(evaluate())
    _print_hard(evaluate_hard())
