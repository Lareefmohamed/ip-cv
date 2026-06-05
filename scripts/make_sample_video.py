"""Generate a synthetic cricket-ball clip WITH ground-truth coordinates.

Produces a video of a red ball travelling along a realistic projectile arc
over a cluttered, noisy background (including *static* red distractors and
moving non-ball objects).  The ground-truth centroid of every frame is saved
alongside, so detection accuracy can be measured objectively.

Usage:
    python scripts/make_sample_video.py --out sample.mp4 --frames 120
Outputs:
    sample.mp4         the clip
    sample.gt.csv      ground truth: frame,x,y,radius
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np

W, H = 854, 480


def _background(rng: np.random.Generator) -> np.ndarray:
    """A green pitch + textured stands with a few STATIC red distractors."""
    bg = np.zeros((H, W, 3), np.uint8)
    bg[:, :] = (60, 110, 60)                       # green field (BGR)
    cv2.rectangle(bg, (0, 0), (W, 150), (90, 90, 90), -1)   # grey stands
    # Speckled crowd texture in the stands.
    for _ in range(1500):
        x, y = rng.integers(0, W), rng.integers(0, 150)
        col = tuple(int(c) for c in rng.integers(40, 200, 3))
        cv2.circle(bg, (int(x), int(y)), 1, col, -1)
    # Static red distractors (advertising boards / a red cap) — these MUST NOT
    # be tracked; the motion gate should reject them.
    cv2.rectangle(bg, (40, 90), (110, 120), (0, 0, 180), -1)
    cv2.circle(bg, (760, 70), 9, (0, 0, 200), -1)
    cv2.line(bg, (0, 300), (W, 320), (50, 90, 50), 3)        # crease line
    return bg


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="sample.mp4")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    base = _background(rng)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))

    # Projectile arc: ball thrown from the left, bounces once.
    x0, y0 = 80.0, 180.0
    vx, vy = 6.4, 1.5
    g = 0.55
    radius = 9
    x, y = x0, y0

    gt_rows = []
    for f in range(args.frames):
        frame = base.copy()

        # A moving NON-red distractor (white player) — wrong colour, must be
        # ignored by the colour mask.
        px = int(120 + (f * 3) % (W - 240))
        cv2.circle(frame, (px, 360), 16, (230, 230, 230), -1)

        # Physics update for the ball.
        x += vx
        y += vy
        vy += g
        if y > H - 40 and vy > 0:        # bounce off the pitch
            vy = -vy * 0.7
        x = min(x, W - 5)

        # Draw the red ball with a subtle seam + mild motion blur smear.
        cx, cy = int(x), int(y)
        cv2.circle(frame, (cx, cy), radius, (0, 0, 210), -1)
        cv2.circle(frame, (cx, cy), radius, (0, 0, 150), 1)
        cv2.line(frame, (cx - radius, cy), (cx + radius, cy), (20, 20, 90), 1)

        # Sensor noise.
        noise = rng.integers(-12, 12, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        writer.write(frame)
        gt_rows.append((f, round(x, 1), round(y, 1), radius))

    writer.release()

    gt_path = os.path.splitext(args.out)[0] + ".gt.csv"
    with open(gt_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x", "y", "radius"])
        w.writerows(gt_rows)

    print(f"Wrote {args.out} ({args.frames} frames) and {gt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
