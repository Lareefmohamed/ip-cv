"""Trajectory modelling and Hawk-Eye style overlay drawing.

Two responsibilities:
    * fit a smooth curve to the recorded centroids (polynomial regression)
      so jitter is removed and the path can be extrapolated a little beyond
      the last detection, and
    * draw the trail, the fitted curve, the current ball marker and an HUD.

The curve is fitted PARAMETRICALLY -- x(t) and y(t) over frame time -- so it
works for any camera view (side-on, behind the bowler, front-on, diagonal).
The legacy y = f(x) fit is kept for backward compatibility but only works
when the ball travels horizontally (side view).
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import PipelineConfig


def fit_trajectory(points: Sequence[Tuple[float, float]], degree: int = 2
                   ) -> Optional[np.poly1d]:
    """Fit y as a polynomial of x.  Returns ``None`` if under-determined."""
    if len(points) < degree + 1:
        return None
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    # Need a reasonable spread in x for the fit to be meaningful.
    if float(xs.max() - xs.min()) < 1e-3:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore poorly-conditioned fits
            coeffs = np.polyfit(xs, ys, degree)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return np.poly1d(coeffs)


def smooth_curve_points(points: Sequence[Tuple[float, float]], degree: int,
                        extrapolate: float = 0.0, n: int = 100
                        ) -> List[Tuple[int, int]]:
    """Return densely-sampled (x, y) points along the fitted curve.

    ``extrapolate`` extends the x-range by this fraction past the last point
    so the curve previews where the ball is heading.
    """
    poly = fit_trajectory(points, degree)
    if poly is None:
        return []
    xs = np.array([p[0] for p in points], dtype=np.float64)
    x_min, x_max = float(xs.min()), float(xs.max())
    span = x_max - x_min
    x_max += span * extrapolate
    sample = np.linspace(x_min, x_max, n)
    return [(int(x), int(poly(x))) for x in sample]


def fit_trajectory_parametric(points: Sequence[Tuple[float, float]],
                              ts: Optional[Sequence[float]] = None,
                              deg_x: int = 2, deg_y: int = 2
                              ) -> Optional[Tuple[np.poly1d, np.poly1d,
                                                  float, float]]:
    """Fit x(t) and y(t) polynomials over frame time.

    Unlike ``fit_trajectory`` (y as a function of x) this works for ANY ball
    direction -- vertical, front-on, diagonal -- because time always advances
    even when x does not.  ``ts`` are the frame indices of the points; when
    omitted the points are assumed equally spaced.

    Returns ``(poly_x, poly_y, t_min, t_max)`` or ``None`` if under-determined.
    """
    n = len(points)
    if n < 2:
        return None
    if ts is None or len(ts) != n:
        t = np.arange(n, dtype=np.float64)
    else:
        t = np.array(ts, dtype=np.float64)
    if float(t.max() - t.min()) < 1e-9:
        return None
    # Short trails can't support a quadratic -- degrade gracefully.
    if n < 5:
        deg_x = min(deg_x, 1)
        deg_y = min(deg_y, 1)
    deg_x = min(deg_x, n - 1)
    deg_y = min(deg_y, n - 1)
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore poorly-conditioned fits
            cx = np.polyfit(t, xs, deg_x)
            cy = np.polyfit(t, ys, deg_y)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return np.poly1d(cx), np.poly1d(cy), float(t.min()), float(t.max())


def smooth_curve_points_parametric(points: Sequence[Tuple[float, float]],
                                   ts: Optional[Sequence[float]] = None,
                                   extrapolate: float = 0.0, n: int = 100,
                                   deg_x: int = 2, deg_y: int = 2,
                                   clamp: float = 1e6
                                   ) -> List[Tuple[int, int]]:
    """Densely sample the parametric fit; works for all camera views.

    ``extrapolate`` extends the time range by this fraction past the last
    point to preview where the ball is heading.  Coordinates are clamped to
    ``±clamp`` so a wild extrapolation can't overflow int32 in cv2.polylines.
    """
    fit = fit_trajectory_parametric(points, ts, deg_x, deg_y)
    if fit is None:
        return []
    poly_x, poly_y, t_min, t_max = fit
    span = t_max - t_min
    sample = np.linspace(t_min, t_max + span * extrapolate, n)
    xs = np.clip(poly_x(sample), -clamp, clamp)
    ys = np.clip(poly_y(sample), -clamp, clamp)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def draw_overlay(frame: np.ndarray,
                 cfg: PipelineConfig,
                 ball: Optional[Tuple[float, float, str]],
                 radius: float,
                 trail: Sequence[Tuple[int, int]],
                 fps: Optional[float] = None,
                 frame_idx: Optional[int] = None,
                 trail_t: Optional[Sequence[float]] = None) -> np.ndarray:
    """Render trail + smoothed curve + ball marker + HUD onto a copy.

    ``trail_t`` (optional) gives the frame index of each trail point so the
    parametric fit stays correct when frames were skipped.
    """
    out = frame.copy()

    # 1. Smoothed regression curve (drawn first, underneath).  Parametric
    #    x(t), y(t) fit -> renders for every camera view, not just side-on.
    if cfg.draw_smoothed_curve and len(trail) >= 3:
        extra = 0.12 if cfg.draw_prediction else 0.0
        h, w = frame.shape[:2]
        clamp = 4.0 * float(np.hypot(w, h))
        ts = list(trail_t) if trail_t is not None else None
        curve = smooth_curve_points_parametric(
            list(trail), ts, extra,
            deg_x=min(cfg.poly_degree, 2), deg_y=cfg.poly_degree,
            clamp=clamp)
        if len(curve) >= 2:
            cv2.polylines(out, [np.array(curve, np.int32)], False,
                          cfg.curve_color, 2, cv2.LINE_AA)

    # 2. Raw fading trail of recorded centroids.
    if cfg.draw_raw_points and len(trail) >= 2:
        pts = list(trail)
        n = len(pts)
        for i in range(1, n):
            thickness = max(1, int(np.sqrt(64.0 / (n - i + 1)) * 1.2))
            cv2.line(out, pts[i - 1], pts[i], cfg.trail_color,
                     thickness, cv2.LINE_AA)

    # 3. Current ball marker.
    if ball is not None:
        x, y, source = ball
        r = max(4, int(radius))
        color = cfg.ball_color if source == "measured" else (0, 165, 255)
        cv2.circle(out, (int(x), int(y)), r, color, 2, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
        label = f"({int(x)},{int(y)})"
        if source == "predicted":
            label += " [pred]"
        cv2.putText(out, label, (int(x) + r + 4, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 4. HUD.
    hud = []
    if frame_idx is not None:
        hud.append(f"frame {frame_idx}")
    if fps is not None:
        hud.append(f"{fps:5.1f} FPS")
    if hud:
        cv2.putText(out, "  ".join(hud), (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return out
