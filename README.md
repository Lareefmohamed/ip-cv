# 🏏 Cricket Ball Trajectory Tracker

**Group 39 — EC7205 Image Processing & Computer Vision, University of Ruhuna**

A lightweight, **traditional computer-vision** tool that detects a cricket ball
in standard video (smartphone / YouTube) and draws a Hawk-Eye style trajectory
trail. **No deep learning, no GPU, no training phase** — it works the moment you
pick the ball's colour.

It implements the proposal pipeline and pushes detection accuracy well past
plain colour masking by *fusing four classical techniques*:

| Stage | Technique | Purpose |
|-------|-----------|---------|
| Pre-process | Gaussian blur, BGR→**HSV** | Noise removal; separate colour from brightness |
| Segmentation | **HSV colour masking** (red / white / pink) | Isolate the ball's colour |
| Motion gate | **MOG2 background subtraction** (Zivkovic) | Reject *static* same-coloured clutter (boards, caps, crowd) |
| Cleaning | Morphological **open + close** | Remove speckle, fill the blob |
| Feature extraction | **Contours + circularity / area / fill filtering** | Keep only round, ball-sized blobs |
| Localisation | **Centroid via image moments** | Sub-pixel (x, y) per frame |
| Tracking | **Constant-velocity Kalman filter** with prediction gating | Lock onto the ball, ignore jumps, coast through occlusion |
| Trajectory | **Polynomial regression** (`y = ax² + bx + c`) | Smooth, extrapolated Hawk-Eye trail |

### Three tracking engines

| Engine | Technique | Best for |
|--------|-----------|----------|
| **Hybrid** (recommended) | Detector **+** CSRT **+** Kalman, fused per frame, with smooth gap-filling | General real clips. Highest robustness — a miss in any one method is covered by the others. |
| **Automatic** | HSV mask + MOG2 + Kalman (above) | Steady camera, clearly visible (ideally red) ball |
| **Assisted** | **CSRT** correlation-filter tracker | White balls / clutter / blur. Auto-seeds from the detector, or you box the ball once. |

All three are classical CV (no deep learning). The **Hybrid** engine is the answer
to the weak spot of pure colour masking and the missed-frame problem:

- **Soft motion gate** — motion *boosts* a candidate's score instead of deleting
  blobs the motion mask misses (so a fast ball survives).
- **ROI-guided detection** — relaxed thresholds around the Kalman prediction →
  higher recall exactly where the ball is expected.
- **Blur tolerance** — accepts elongated (motion-blurred) streaks via solidity.
- **CSRT lock** — carries frames the detector misses (texture, not colour).
- **Smooth gap-fill** — brief misses are filled from the Kalman prediction so the
  trajectory stays continuous. Reported as **coverage** vs the real **detection
  rate** so the trade-off is transparent.
- **One-click HSV calibration** — box the ball once and the colour range is auto-
  tuned to your exact ball / lighting.

Measured on a hard synthetic clip (fast + motion-blur + camera jitter +
occlusion): **Hybrid 92%** vs Automatic 85%; on a real side-view clip Hybrid
reaches **~94–99%** coverage vs ~36% for colour-only.

> The MOG2 motion gate + Kalman tracker are the key accuracy boosters: a red
> advertising board or a red cap is the right *colour* but the wrong *motion*,
> so it never gets tracked.

## Measured performance (synthetic ground-truth clip)

```
Detection rate        : 95.8 %
Localisation accuracy : 99.1 %   (correct within ~1 ball radius)
Overall accuracy      : 95.0 %
Mean pixel error      : 2.34 px
Processing speed      : ~65 FPS  (CPU only)
```

Proposal targets were 85–90 % detection and ≥30 FPS — both comfortably exceeded.

## Install

```bash
pip install -r requirements.txt
```

## Usage

### 1. Web app (Streamlit — the main deliverable)

```bash
streamlit run app.py
```

Upload a clip, choose the ball colour, press **Process video**. You get the
annotated video, a live preview, detection-rate / FPS metrics, and downloadable
trajectory CSV.

### 2. Command line

```bash
# Save an annotated video (Automatic engine)
python run.py --input clip.mp4 --output tracked.mp4 --color red

# Live preview window (q = quit, m = toggle mask view)
python run.py --input clip.mp4 --show

# Tuning presets: default | fast | white-ball | broadcast | hybrid
python run.py --input clip.mp4 --output out.mp4 --preset fast

# HYBRID engine (recommended) — detector + CSRT + Kalman + gap-fill
python run.py --input clip.mp4 --output out.mp4 --hybrid --color red

# Hybrid + see why frames were missed
python run.py --input clip.mp4 --output out.mp4 --hybrid --diagnose

# Hybrid + calibrate the ball colour from a box you draw once
python run.py --input clip.mp4 --output out.mp4 --hybrid --select --calibrate

# Assisted CSRT tracker, auto-seeded from the detector
python run.py --input clip.mp4 --output out.mp4 --assisted --color red

# Assisted CSRT, click the ball yourself on a chosen seed frame
python run.py --input clip.mp4 --output out.mp4 --assisted --select --select-frame 40
```

**Presets** (`--preset`):
- `default` — steady camera, red ball (the demo case)
- `fast` — fast bowling: relaxed shape filters for a motion-blurred ball, wider match gate
- `white-ball` — white ball day game
- `broadcast` — multi-camera/panning footage: motion gate off; pair with `--assisted`
- `hybrid` — robust general-purpose profile used by `--hybrid`

**Hybrid flags:** `--hybrid` (engine), `--no-fill-gaps` (show only real
detections), `--calibrate` (with `--select`, auto-tune colour from the box),
`--diagnose` (print per-frame outcomes + blob reject reasons).

### 3. Tune HSV for your own footage

```bash
python scripts/tune_hsv.py --input clip.mp4
```

Drag the trackbars until only the ball is white in the mask, then paste the
printed range into [`src/config.py`](src/config.py) (`COLOR_PROFILES`).

### 4. Reproduce the accuracy numbers

```bash
python tests/test_accuracy.py        # generates a synthetic clip + scores it
```

This auto-creates a synthetic clip *with known ground-truth centroids* (red
ball on a cluttered, noisy field with static red distractors) and reports the
metrics above. `scripts/make_sample_video.py` generates the clip standalone.

## Project layout

```
app.py                      Streamlit web app
run.py                      CLI runner
requirements.txt
src/
  config.py                 HSV profiles + all tunable parameters
  detector.py               HSV + MOG2 + shape filtering -> candidates
  tracker.py                Kalman motion-consistency tracker
  trajectory.py             polynomial regression + overlay drawing
  pipeline.py               glues detector+tracker+trajectory; whole-video API
scripts/
  tune_hsv.py               interactive HSV range tuner
  make_sample_video.py      synthetic clip + ground-truth generator
tests/
  test_accuracy.py          objective accuracy evaluation vs ground truth
```

## Tips for best accuracy on real footage

1. **Tune the HSV range** to your specific ball + lighting with `tune_hsv.py`.
2. Keep the **motion gate on** for cluttered backgrounds (default); turn it off
   only for very short clips where MOG2 hasn't learned the background yet.
3. Raise **Min circularity** if round non-ball objects slip through; lower it if
   a motion-blurred ball is being rejected.
4. A **tripod / steady camera** dramatically improves the motion gate.
