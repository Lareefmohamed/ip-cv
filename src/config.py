"""Central configuration for the cricket ball detection pipeline.

All tunable parameters live here so the detector, tracker and Streamlit UI
share a single source of truth.  Values are expressed in OpenCV HSV space
where H in [0, 179], S in [0, 255], V in [0, 255].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

# A HSV range is (lower, upper) with each bound being an (H, S, V) triple.
HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


# ---------------------------------------------------------------------------
# Pre-defined colour profiles for common cricket balls.
#
# Red wraps around the hue circle (0 and 179), so the red profile needs TWO
# ranges that are OR-ed together.  White is low-saturation / high-value.
# ---------------------------------------------------------------------------
COLOR_PROFILES: dict[str, List[HSVRange]] = {
    "red": [
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (179, 255, 255)),
    ],
    "white": [
        ((0, 0, 200), (179, 40, 255)),
    ],
    # Pink (day-night ball) sits between red and magenta.
    "pink": [
        ((150, 80, 120), (179, 255, 255)),
        ((0, 80, 120), (8, 255, 255)),
    ],
}


@dataclass
class DetectorConfig:
    """Parameters controlling the detection pipeline.

    Defaults are tuned for 720p-ish smartphone footage of a red ball.  The
    Streamlit UI and HSV tuner overwrite these at runtime.
    """

    # Which colour profile to use ("red" / "white" / "pink") or a custom list.
    color: str = "red"
    custom_ranges: List[HSVRange] | None = None

    # Pre-processing.
    blur_ksize: int = 5            # Gaussian blur kernel (odd number, 0 disables)

    # Morphology (mask cleaning).
    morph_ksize: int = 5           # structuring-element size
    erode_iter: int = 1
    dilate_iter: int = 2

    # Background subtraction (motion gate).  This is the key accuracy booster:
    # it suppresses static red/white objects in the crowd / stands.
    use_motion: bool = True
    mog2_history: int = 200
    mog2_var_threshold: float = 32.0
    motion_dilate: int = 9         # dilate the motion mask so it covers a fast,
    #                                motion-blurred ball comfortably.

    # Contour / shape filtering.
    min_area: float = 12.0         # px^2 - reject tiny speckle
    max_area: float = 8000.0       # px^2 - reject huge blobs (clothing, pads)
    min_circularity: float = 0.55  # 4*pi*A / P^2  (1.0 == perfect circle)
    min_radius: float = 2.0        # px - minimum enclosing-circle radius
    max_radius: float = 60.0       # px
    min_fill_ratio: float = 0.55   # contour area / enclosing-circle area

    def ranges(self) -> List[HSVRange]:
        """Resolve the active list of HSV ranges."""
        if self.custom_ranges:
            return self.custom_ranges
        return COLOR_PROFILES.get(self.color, COLOR_PROFILES["red"])


@dataclass
class TrackerConfig:
    """Parameters for the Kalman motion-consistency tracker."""

    # Maximum distance (px) a detection may sit from the Kalman prediction and
    # still be accepted as the same ball.  Scaled up while coasting.
    max_match_dist: float = 90.0
    # How many consecutive frames the tracker may "coast" on prediction alone
    # before the track is dropped (handles brief occlusion / missed detection).
    max_coast_frames: int = 8
    # Frames of consistent detection required before a track is "confirmed".
    min_hits_to_confirm: int = 3
    # Length of the visual trajectory trail (number of points kept).
    trail_length: int = 64


@dataclass
class PipelineConfig:
    """Top-level config bundling detector + tracker + drawing options."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    # Trajectory drawing.
    draw_raw_points: bool = True
    draw_smoothed_curve: bool = True
    draw_prediction: bool = True
    poly_degree: int = 2           # quadratic (y = a x^2 + b x + c)
    trail_color: Tuple[int, int, int] = (0, 255, 255)   # BGR - yellow
    ball_color: Tuple[int, int, int] = (0, 0, 255)      # BGR - red marker
    curve_color: Tuple[int, int, int] = (0, 255, 0)     # BGR - green fit


# ---------------------------------------------------------------------------
# Ready-made presets so footage doesn't have to be hand-tuned every time.
# ---------------------------------------------------------------------------
def make_preset(name: str) -> PipelineConfig:
    """Return a PipelineConfig tuned for a common scenario.

    Presets
    -------
    default     : red ball, steady camera (the demo / tripod case).
    fast        : fast bowling - relax shape filters so a motion-blurred
                  (elliptical) ball still passes, and widen the match gate.
    white-ball  : white ball day game - white profile, relaxed shape.
    broadcast   : multi-camera / panning footage - motion gate OFF (it breaks
                  on camera motion); pair with the CSRT assisted tracker.
    """
    cfg = PipelineConfig()
    name = (name or "default").lower()
    if name == "default":
        return cfg
    if name == "fast":
        cfg.detector.color = "red"
        cfg.detector.min_circularity = 0.30
        cfg.detector.min_fill_ratio = 0.40
        cfg.detector.min_area = 6.0
        cfg.detector.motion_dilate = 13
        cfg.tracker.max_match_dist = 160.0
        cfg.tracker.max_coast_frames = 12
        return cfg
    if name in ("white", "white-ball", "white_ball"):
        cfg.detector.color = "white"
        cfg.detector.min_circularity = 0.35
        cfg.detector.min_fill_ratio = 0.45
        return cfg
    if name == "broadcast":
        cfg.detector.use_motion = False
        cfg.detector.min_circularity = 0.30
        cfg.detector.min_fill_ratio = 0.40
        cfg.tracker.max_match_dist = 200.0
        return cfg
    raise ValueError(f"Unknown preset: {name!r}")


PRESETS = ["default", "fast", "white-ball", "broadcast"]
