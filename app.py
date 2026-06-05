"""Streamlit web app for the Cricket Ball Trajectory Tracker.

Run with:
    streamlit run app.py

Two tracking engines, both traditional CV (no deep learning):
  * Automatic   - HSV colour masking + MOG2 motion gating + Kalman tracking.
                  Best for steady cameras with a clearly visible (ideally red)
                  ball.
  * Assisted    - CSRT correlation-filter tracker. Far more robust on hard
                  footage (white balls, clutter, motion blur). Either auto-seeds
                  from the detector, or you place a box on the ball yourself.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from src.config import PipelineConfig
from src.pipeline import (CricketBallPipeline, process_video_assisted,
                          video_meta)

st.set_page_config(page_title="Cricket Ball Trajectory Tracker",
                   page_icon="🏏", layout="wide")

st.title("🏏 Cricket Ball Trajectory Tracker")
st.caption("Traditional computer vision — HSV masking + MOG2 + Kalman, or the "
           "CSRT assisted tracker. No deep learning. (Group 39, EC7205)")


# --- Sidebar controls ------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Tracking mode",
        ["Automatic (HSV + motion)",
         "Assisted (CSRT, auto-seed)",
         "Assisted (CSRT, click the ball)"],
        index=0,
        help="Automatic = colour-based, best for a clear red ball on a steady "
             "camera. Assisted = CSRT, robust on white balls / clutter / blur.")
    is_assisted = mode.startswith("Assisted")
    is_manual = mode.endswith("click the ball)")

    color = st.selectbox("Ball colour", ["red", "white", "pink"], index=0,
                         help="Used by Automatic mode and by Assisted auto-seed.")
    use_motion = st.checkbox(
        "Motion gating (MOG2)", value=True,
        help="Suppresses static same-coloured clutter. Turn off for very "
             "short clips or broadcast footage with camera cuts/pans.")
    min_circ = st.slider("Min circularity", 0.0, 1.0, 0.55, 0.05,
                         help="Higher = stricter round shape. Lower (~0.3) for "
                              "fast, motion-blurred balls.")
    blur = st.slider("Gaussian blur kernel", 0, 15, 5, 1)
    poly_deg = st.selectbox("Trajectory polynomial degree", [1, 2, 3], index=1)
    show_mask = st.checkbox("Show binary mask preview (Automatic only)",
                            value=False)
    st.markdown("---")
    st.markdown("**Tip:** tune HSV for your footage with "
                "`python scripts/tune_hsv.py --input clip.mp4`.")


def make_config() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.detector.color = color
    cfg.detector.use_motion = use_motion
    cfg.detector.min_circularity = min_circ
    cfg.detector.blur_ksize = blur
    cfg.poly_degree = poly_deg
    return cfg


uploaded = st.file_uploader("Upload a cricket video",
                            type=["mp4", "mov", "avi", "mkv"])

if uploaded is None:
    st.info("👆 Upload a short clip of a delivery to begin.")
    st.stop()

# Persist the upload to a temp file OpenCV can read.
suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
tmp_in.write(uploaded.read())
tmp_in.close()

meta = video_meta(tmp_in.name)
c1, c2, c3 = st.columns(3)
c1.metric("Resolution", f"{meta['width']}x{meta['height']}")
c2.metric("Frames", meta["frames"])
c3.metric("Source FPS", f"{meta['fps']:.0f}")


def read_frame(path: str, index: int):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# --- Manual seed-box placement (only for the "click the ball" mode) ---------
seed_frame = 0
bbox = None
if is_manual:
    st.subheader("1) Place a box on the ball")
    seed_frame = st.slider("Seed frame", 0, max(meta["frames"] - 1, 0),
                           min(meta["frames"] // 3, max(meta["frames"] - 1, 0)))
    frame = read_frame(tmp_in.name, seed_frame)
    if frame is not None:
        h, w = frame.shape[:2]
        cc = st.columns(4)
        bx = cc[0].number_input("Box X", 0, w - 1, w // 2 - 10)
        by = cc[1].number_input("Box Y", 0, h - 1, h // 2 - 10)
        bw = cc[2].number_input("Box W", 2, w, 24)
        bh = cc[3].number_input("Box H", 2, h, 24)
        bbox = (int(bx), int(by), int(bw), int(bh))
        prev = frame.copy()
        cv2.rectangle(prev, (bbox[0], bbox[1]),
                      (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
        st.image(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB),
                 caption="Adjust X/Y/W/H until the red box sits on the ball",
                 use_container_width=True)
    st.subheader("2) Process")
elif mode == "Assisted (CSRT, auto-seed)":
    seed_frame = st.slider(
        "Start tracking from frame", 0, max(meta["frames"] - 1, 0), 0,
        help="Skip a long run-up by starting near the delivery.")

if not st.button("▶ Process video", type="primary"):
    st.stop()

cfg = make_config()
out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name


# --- Processing ------------------------------------------------------------
if is_assisted:
    progress = st.progress(0.0, text="Tracking (CSRT)…")
    n_frames = max(meta["frames"], 1)

    def _cb(i, n):
        progress.progress(min(i / n_frames, 1.0),
                          text=f"Tracking… {i}/{meta['frames']}")

    stats = process_video_assisted(
        tmp_in.name, bbox=bbox if is_manual else None,
        start_frame=seed_frame, out_path=out_path, config=cfg, progress=_cb)
    progress.empty()
    measured = stats["measured_detections"]
    total = stats["frames"]
    rate = stats["detection_rate"]
    proc_fps = stats["mean_proc_fps"]
    track_log = stats["track_log"]
else:
    pipe = CricketBallPipeline(cfg)
    cap = cv2.VideoCapture(tmp_in.name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, meta["fps"] or 30.0,
                             (meta["width"], meta["height"]))
    progress = st.progress(0.0, text="Processing…")
    preview = st.empty()
    measured = total = 0
    fps_sum = 0.0
    n_frames = max(meta["frames"], 1)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pipe.process_frame(frame)
        total += 1
        fps_sum += res.fps
        if res.ball is not None and res.ball[2] == "measured":
            measured += 1
        writer.write(res.frame)
        if total % 5 == 0 or total == 1:
            view = (cv2.cvtColor(res.mask, cv2.COLOR_GRAY2BGR)
                    if show_mask else res.frame)
            preview.image(cv2.cvtColor(view, cv2.COLOR_BGR2RGB),
                          caption=f"Frame {total}", use_container_width=True)
        progress.progress(min(total / n_frames, 1.0),
                          text=f"Processing… {total}/{meta['frames']}")
    cap.release()
    writer.release()
    progress.empty()
    rate = (measured / total) if total else 0.0
    proc_fps = (fps_sum / total) if total else 0.0
    track_log = pipe.track_log


# --- Results ---------------------------------------------------------------
st.success(f"Done! Mode: {mode}")
m1, m2, m3 = st.columns(3)
m1.metric("Tracked rate", f"{rate:.1%}")
m2.metric("Frames tracked", f"{measured}/{total}")
m3.metric("Processing speed", f"{proc_fps:.1f} FPS")

with open(out_path, "rb") as f:
    data = f.read()
st.video(data)
st.download_button("⬇ Download annotated video", data,
                   file_name="tracked_" + uploaded.name, mime="video/mp4")

if track_log:
    csv = "frame,x,y,source\n" + "\n".join(
        f"{i},{x:.1f},{y:.1f},{s}" for i, x, y, s in track_log)
    st.download_button("⬇ Download trajectory (CSV)", csv,
                       file_name="trajectory.csv", mime="text/csv")
