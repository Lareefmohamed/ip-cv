"""Command-line runner for the Cricket Ball Trajectory Tracker.

Examples
--------
Process a clip of a red ball and save the annotated result:
    python run.py --input clip.mp4 --output tracked.mp4 --color red

Live preview window (press 'q' to quit, 'm' to toggle the mask view):
    python run.py --input clip.mp4 --show

Tune sensitivity for white balls and disable the motion gate:
    python run.py --input clip.mp4 --color white --no-motion --show
"""

from __future__ import annotations

import argparse
import sys

import cv2

from src.config import PRESETS, PipelineConfig, make_preset
from src.pipeline import (CricketBallPipeline, iter_video, process_video,
                          process_video_assisted, video_meta)


def build_config(args) -> PipelineConfig:
    cfg = make_preset(args.preset)
    # Explicit flags override the preset.
    if args.color is not None:
        cfg.detector.color = args.color
    if args.no_motion:
        cfg.detector.use_motion = False
    if args.min_circularity is not None:
        cfg.detector.min_circularity = args.min_circularity
    cfg.poly_degree = args.poly_degree
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cricket ball trajectory tracker")
    p.add_argument("--input", "-i", required=True, help="input video path")
    p.add_argument("--output", "-o", help="annotated output video path (.mp4)")
    p.add_argument("--preset", default="default", choices=PRESETS,
                   help="tuning preset: default (steady red ball), fast "
                        "(fast bowling), white-ball, broadcast (camera cuts/pans)")
    p.add_argument("--color", default=None,
                   choices=["red", "white", "pink"],
                   help="ball colour profile (overrides preset)")
    p.add_argument("--no-motion", action="store_true",
                   help="disable MOG2 motion gating (use for static camera w/ "
                        "little background clutter, or very short clips)")
    p.add_argument("--min-circularity", type=float, default=None,
                   help="reject blobs less circular than this (0-1)")
    p.add_argument("--poly-degree", type=int, default=2,
                   help="trajectory polynomial degree (2 = quadratic)")
    p.add_argument("--show", action="store_true",
                   help="display a live preview window")
    p.add_argument("--assisted", action="store_true",
                   help="use the CSRT assisted tracker (robust on hard footage "
                        "like white balls / clutter). Auto-seeds from the HSV "
                        "detector unless --select is given.")
    p.add_argument("--select", action="store_true",
                   help="with --assisted: open a window to draw a box around "
                        "the ball on the start frame")
    p.add_argument("--select-frame", type=int, default=0,
                   help="frame index to seed the assisted tracker on")
    args = p.parse_args(argv)

    cfg = build_config(args)

    if args.assisted:
        return _run_assisted(args, cfg)

    if not args.show:
        if not args.output:
            print("Nothing to do: pass --output to save or --show to preview.",
                  file=sys.stderr)
            return 2
        print(f"Processing {args.input} -> {args.output} ...")
        stats = process_video(
            args.input, args.output, cfg,
            progress=lambda i, n: print(f"\r  frame {i}/{n or '?'}",
                                        end="", flush=True))
        print()
        _report(stats)
        return 0

    # Live preview path.
    pipe = CricketBallPipeline(cfg)
    show_mask = False
    measured = total = 0
    for frame in iter_video(args.input):
        res = pipe.process_frame(frame)
        total += 1
        if res.ball is not None and res.ball[2] == "measured":
            measured += 1
        view = cv2.cvtColor(res.mask, cv2.COLOR_GRAY2BGR) if show_mask else res.frame
        cv2.imshow("Cricket Ball Tracker", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_mask = not show_mask
    cv2.destroyAllWindows()
    rate = (measured / total) if total else 0.0
    print(f"Detection rate: {rate:.1%} ({measured}/{total} frames)")
    return 0


def _run_assisted(args, cfg) -> int:
    """CSRT assisted-tracking path (manual box or auto-seed)."""
    if not args.output and not args.show:
        print("Pass --output to save (assisted mode needs an output or --show).",
              file=sys.stderr)
        return 2

    bbox = None
    if args.select:
        # Grab the chosen seed frame and let the user draw the ball box.
        from src.assisted_tracker import select_bbox_interactive
        cap = cv2.VideoCapture(args.input)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.select_frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("Could not read the selected frame.", file=sys.stderr)
            return 1
        bbox = select_bbox_interactive(frame)
        if bbox is None:
            print("No box selected - aborting.", file=sys.stderr)
            return 1
        print(f"Seed box: {bbox}")

    mode = "manual box" if bbox else "auto-seed (HSV detector)"
    print(f"Assisted CSRT tracking ({mode}) {args.input} -> {args.output} ...")
    stats = process_video_assisted(
        args.input, bbox=bbox, start_frame=args.select_frame,
        out_path=args.output, config=cfg,
        progress=lambda i, n: print(f"\r  frame {i}/{n or '?'}",
                                    end="", flush=True))
    print()
    _report(stats)
    return 0


def _report(stats: dict) -> None:
    print("=" * 44)
    print(f"  Frames processed : {stats['frames']}")
    print(f"  Measured frames  : {stats['measured_detections']}")
    print(f"  Detection rate   : {stats['detection_rate']:.1%}")
    print(f"  Mean processing  : {stats['mean_proc_fps']:.1f} FPS")
    print(f"  Output           : {stats['out_path']}")
    print("=" * 44)


if __name__ == "__main__":
    raise SystemExit(main())
