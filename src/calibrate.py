"""One-click HSV auto-calibration.

Given a box the user drew around the ball, sample the HSV pixels inside it and
build a robust colour range (5th-95th percentile per channel).  This adapts the
detector to *any* ball colour and lighting from a single click, which is the
main reason the system generalises beyond the red-ball preset.

Hue wraps around at 180 in OpenCV, so a red ball straddling 0/179 is detected
and split into the two ranges that ``DetectorConfig.ranges()`` already OR-s.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .config import HSVRange

BBox = Tuple[int, int, int, int]


def calibrate_hsv(frame_bgr: np.ndarray, bbox: BBox,
                  s_pad: int = 40, v_pad: int = 50,
                  h_pad: int = 8) -> List[HSVRange]:
    """Return HSV range(s) covering the ball pixels inside ``bbox``.

    The centre of the box is sampled (a disc) so background corners of the box
    don't pollute the colour estimate.
    """
    x, y, w, h = (int(v) for v in bbox)
    H, W = frame_bgr.shape[:2]
    x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
    w = max(2, min(w, W - x)); h = max(2, min(h, H - y))
    patch = frame_bgr[y:y + h, x:x + w]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # Sample a centred disc (radius = 40% of the smaller side).
    ph, pw = hsv.shape[:2]
    cy, cx = ph / 2.0, pw / 2.0
    r = 0.4 * min(ph, pw)
    yy, xx = np.ogrid[:ph, :pw]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    pix = hsv[disc] if disc.any() else hsv.reshape(-1, 3)

    hh = pix[:, 0].astype(np.int32)
    ss = pix[:, 1].astype(np.int32)
    vv = pix[:, 2].astype(np.int32)

    s_lo = int(max(0, np.percentile(ss, 5) - s_pad))
    s_hi = int(min(255, np.percentile(ss, 95) + s_pad))
    v_lo = int(max(0, np.percentile(vv, 5) - v_pad))
    v_hi = int(min(255, np.percentile(vv, 95) + v_pad))

    h_lo = float(np.percentile(hh, 5))
    h_hi = float(np.percentile(hh, 95))

    # Detect hue wraparound: if the hue spread is wide, the colour likely
    # straddles 0/179 (classic red).  Split into two ranges.
    if h_hi - h_lo > 90:
        low_hues = hh[hh < 90]
        high_hues = hh[hh >= 90]
        r1 = (int(max(0, low_hues.min() - h_pad)),
              int(min(89, low_hues.max() + h_pad)))
        r2 = (int(max(90, high_hues.min() - h_pad)),
              int(min(179, high_hues.max() + h_pad)))
        return [
            ((r1[0], s_lo, v_lo), (r1[1], s_hi, v_hi)),
            ((r2[0], s_lo, v_lo), (r2[1], s_hi, v_hi)),
        ]

    h_lo = int(max(0, h_lo - h_pad))
    h_hi = int(min(179, h_hi + h_pad))
    return [((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))]
