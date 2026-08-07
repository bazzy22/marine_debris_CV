from dataclasses import dataclass
from typing import Tuple
import os

# ── Default File Paths & Settings ──────────────────────────────────────────
DEFAULT_MODEL = os.path.join("trained model", "best.pt")
DEFAULT_MODE = "yolo"
DEFAULT_LEFT_SRC = os.path.join("video", "plastic_video.mp4")
DEFAULT_RIGHT_SRC = os.path.join("video", "plastic_video.mp4")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class StereoConfig:
    """
    All tunable hardware and algorithm parameters in one place.

    Units
    -----
    baseline      : metres  (not cm — depth formula gives metres)
    focal_length  : pixels  (from calibration)
    """

    # ── Camera geometry ────────────────────────────────────────────────────
    baseline: float = 0.12         # Inter-camera distance in metres (12 cm rig)
    focal_length: float = 1114.0   # Calibrated focal length in pixels

    # ── Pi Camera Module 3 capture settings ─────────────────────────────────
    pi_resolution: Tuple[int, int] = (640, 480)   # Lowered from 720p for Pi 5 headroom
    pi_fps: int = 30
    pi_ae_settle_s: float = 2.0    # Time to let auto-exposure/AWB converge

    # ── Sparse block matching (DEFAULT distance method — cheap) ─────────────
    match_patch_size: int = 32       # Template patch side length, pixels
    match_search_range: int = 300     # Max disparity searched, pixels
    match_row_tolerance: int = 6     # Vertical search slack (rows)
    match_min_confidence: float = 0.5  # Discard matches below this NCC score

    # ── Dense StereoSGBM (OPT-IN via --dense-depth — expensive) ─────────────
    min_disparity: int = 0
    num_disparities: int = 48      # Must be divisible by 16 (reduced for Pi 5)
    block_size: int = 9            # Matcher window (odd, 3–11; smaller = faster)

    # ── WLS filter ─────────────────────────────────────────────────────────
    wls_lambda: float = 8000.0
    wls_sigma: float = 1.5

    # ── Detection ──────────────────────────────────────────────────────────
    min_object_area: int = 400
    patch_size: int = 7            # NxN neighbourhood for depth sampling

    # ── Display ────────────────────────────────────────────────────────────
    display: bool = True           # Set False for headless / production use

    # ── YOLO half-resolution inference ─────────────────────────────────────
    yolo_half_res: bool = True     # Halve frame before YOLO → ~4× faster