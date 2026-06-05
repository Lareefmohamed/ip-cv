"""Trajectory modelling and Hawk-Eye style overlay drawing.

Two responsibilities:
    * fit a smooth quadratic curve  y = a*x^2 + b*x + c  to the recorded
      centroids (polynomial regression) so jitter is removed and the path
      can be extrapolated a little beyond the last detection, and
    * draw the trail, the fitted curve, the current ball marker and an HUD.
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


def draw_overlay(frame: np.ndarray,
                 cfg: PipelineConfig,
                 ball: Optional[Tuple[float, float, str]],
                 radius: float,
                 trail: Sequence[Tuple[int, int]],
                 fps: Optional[float] = None,
                 frame_idx: Optional[int] = None) -> np.ndarray:
    """Render trail + smoothed curve + ball marker + HUD onto a copy."""
    out = frame.copy()

    # 1. Smoothed regression curve (drawn first, underneath).
    if cfg.draw_smoothed_curve and len(trail) >= cfg.poly_degree + 1:
        extra = 0.12 if cfg.draw_prediction else 0.0
        curve = smooth_curve_points(list(trail), cfg.poly_degree, extra)
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
