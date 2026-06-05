"""Interactive HSV range tuner.

Scrub trackbars until only the ball is white in the mask view, then copy the
printed range into src/config.py (COLOR_PROFILES) or pass via the UI.

Usage:
    python scripts/tune_hsv.py --input clip.mp4
    python scripts/tune_hsv.py --image frame.jpg
Keys:  q = quit and print the range,  space = pause/resume video.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

WIN = "HSV tuner (q to quit)"


def _nothing(_):
    pass


def _make_trackbars():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    for name, val in [("H min", 0), ("H max", 179), ("S min", 120),
                      ("S max", 255), ("V min", 70), ("V max", 255)]:
        hi = 179 if name.startswith("H") else 255
        cv2.createTrackbar(name, WIN, val, hi, _nothing)


def _read() -> tuple[np.ndarray, np.ndarray]:
    g = lambda n: cv2.getTrackbarPos(n, WIN)
    lower = np.array([g("H min"), g("S min"), g("V min")], np.uint8)
    upper = np.array([g("H max"), g("S max"), g("V max")], np.uint8)
    return lower, upper


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="video path")
    p.add_argument("--image", help="single image path")
    args = p.parse_args(argv)
    if not args.input and not args.image:
        p.error("pass --input <video> or --image <image>")

    _make_trackbars()
    cap = cv2.VideoCapture(args.input) if args.input else None
    still = cv2.imread(args.image) if args.image else None
    paused = False
    frame = still

    while True:
        if cap is not None and not paused:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop
                continue
            frame = f
        if frame is None:
            break

        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower, upper = _read()
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        combo = np.hstack([frame,
                           cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                           result])
        cv2.imshow(WIN, combo)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    lower, upper = _read()
    print("Selected HSV range:")
    print(f"  (({lower[0]}, {lower[1]}, {lower[2]}), "
          f"({upper[0]}, {upper[1]}, {upper[2]})),")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
