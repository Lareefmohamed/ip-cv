"""Parametric trajectory fit must work for ALL camera views.

The legacy y = f(x) fit fails whenever x barely changes (front-on or
behind-the-bowler views, ball travelling vertically in the image).  The
parametric x(t), y(t) fit has no such restriction.

Run directly:        python tests/test_trajectory_parametric.py
Or under pytest:     pytest tests/test_trajectory_parametric.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trajectory import (fit_trajectory, fit_trajectory_parametric,  # noqa: E402
                            smooth_curve_points_parametric)


def test_vertical_path_now_fits():
    """Ball moving straight down the image (front-on view): the old fit
    returns None, the parametric fit must produce a usable curve."""
    points = [(320, 50 + 10 * i) for i in range(20)]
    assert fit_trajectory(points) is None          # the original bug
    curve = smooth_curve_points_parametric(points)
    assert len(curve) >= 2
    # The curve must actually follow the points (x stays ~320, y increases).
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    assert all(abs(x - 320) <= 2 for x in xs)
    assert ys[-1] > ys[0]


def test_diagonal_path():
    points = [(50 + 8 * i, 400 - 12 * i + i * i) for i in range(15)]
    curve = smooth_curve_points_parametric(points)
    assert len(curve) >= 2
    assert curve[0][0] < curve[-1][0]   # moves right over time


def test_side_view_still_works():
    """Classic side-view parabola keeps fitting fine parametrically."""
    points = [(20 * i, 300 - 40 * i + 4 * i * i) for i in range(12)]
    curve = smooth_curve_points_parametric(points, extrapolate=0.12)
    assert len(curve) >= 2


def test_short_trail_no_crash():
    """3 points: degree must degrade to linear, not crash."""
    points = [(100, 100), (110, 120), (120, 145)]
    curve = smooth_curve_points_parametric(points)
    assert len(curve) >= 2
    assert fit_trajectory_parametric(points[:1]) is None
    assert smooth_curve_points_parametric(points[:1]) == []


def test_skipped_frames_use_timestamps():
    """Explicit frame indices keep the fit correct across gaps."""
    ts = [0, 1, 2, 6, 7, 8]
    points = [(10 * t, 5 * t) for t in ts]
    fit = fit_trajectory_parametric(points, ts)
    assert fit is not None
    poly_x, poly_y, t_min, t_max = fit
    assert abs(poly_x(4) - 40) < 1.0    # interpolates the gap linearly
    assert (t_min, t_max) == (0.0, 8.0)


def test_extrapolation_is_clamped():
    """Wild quadratic extrapolation must not overflow int conversion."""
    points = [(i, i * i * 50) for i in range(10)]
    curve = smooth_curve_points_parametric(points, extrapolate=5.0, clamp=1e4)
    assert all(abs(x) <= 1e4 and abs(y) <= 1e4 for x, y in curve)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("All parametric trajectory tests passed.")
