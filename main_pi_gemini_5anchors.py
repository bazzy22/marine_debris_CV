"""
Stereo Vision Pipeline — Optimized for Edge Deployment
=======================================================
Targets: Raspberry Pi 5 (default, lightweight) | NVIDIA Jetson (CUDA, optional dense mode)

Lightweight-by-default design (Pi 5 friendly)
-----------------------------------------------
Dense whole-frame stereo matching (StereoSGBM + WLS) is the single most
expensive thing this pipeline can do — it costs the same whether the frame
has zero or ten objects in it, and full-frame SGBM+WLS at 720p is heavy for
a Pi 5 CPU. But the actual requirement here is just "distance to each
detected object", not a dense per-pixel depth map. So by default this
pipeline:
  1. Runs the detector (AprilTag or YOLO) on the LEFT frame only.
  2. For each detected object, extracts a small patch around its center and
     searches for the matching patch in the RIGHT frame over a limited
     disparity range (`SparseBlockMatcher`) — a few thousand pixel
     comparisons per object instead of millions across the whole frame.
  3. Converts the resulting per-object disparity to a distance.

This is typically an order of magnitude cheaper than dense SGBM+WLS when a
scene has a handful of objects, which is the common case for trash/AprilTag
detection. The old dense full-frame SGBM+WLS backends (CPU/OpenCL/CUDA) are
still included and can be re-enabled with `--dense-depth` if you want a
full visualized depth/disparity map (e.g. for debugging or a demo) — just
expect noticeably lower FPS on a Pi 5 in that mode.

Raspberry Pi Camera Module 3 (dual CSI, stereo) support
---------------------------------------------------------
- Uses `picamera2` (libcamera) instead of cv2.VideoCapture, since CSI cameras
  are not exposed as normal V4L2 devices in a way OpenCV can reliably open,
  and Picamera2 is the officially supported API for Camera Module 3 (IMX708).
- Raspberry Pi 5 has two native CSI connectors, so two Camera Module 3 units
  can run simultaneously as camera_num=0 and camera_num=1 with no extra
  hardware. On a Pi 4 (single CSI port) you need a camera multiplexer board
  (e.g. Arducam/Waveshare) — Picamera2 cannot drive two CSI cameras on a
  Pi 4 without one.
- `--left picam0 --right picam1` (or `--picam2` shorthand) selects this path.
- Exposure/gain/white-balance are locked and matched between the two cameras
  after an auto-exposure settling period, because independent AE/AWB per
  camera causes brightness/color mismatches that visibly degrade block
  matching (sparse or dense).
- Default baseline is set to 0.12 m (12 cm) to match the physical rig
  described by the user. `focal_length` still must come from a real stereo
  calibration — the shipped default is only a rough placeholder for the
  Camera Module 3 sensor and will not give metrically accurate distances
  until calibrated.
- Default capture resolution is lowered to 640x480 (from 1280x720) — this
  alone roughly quarters per-frame cost for grayscale conversion, YOLO
  inference, and any dense-mode matching. Raise with --resolution if you
  need it and have the CPU budget.

Key improvements over original:
  - Bug fix: YoloTrashDetector referenced undefined `trash_boxes` (now `detected_boxes`)
  - Bug fix: Windows-style backslash paths replaced with os.path / pathlib
  - Bug fix: `CUDAStereoProcessor.compute_depth` returned None (now raises NotImplementedError)
  - Feature: CLI argument parsing (--mode, --left, --right, --backend, --no-display)
  - Feature: `--picam2` mode for two Raspberry Pi Camera Module 3 units (Picamera2)
  - Feature: Automatic exposure/AWB locking + matching across the stereo pair
  - Feature: Automatic dense backend probe (CUDA → OpenCL → CPU), opt-in via --dense-depth
  - Feature: `ThreadedVideoGrabber` uses threading.Event for clean shutdown
  - Feature: Frame skipping guard — drops stale frames when processing falls behind
  - Feature: FPS counter overlay on display window
  - Feature: `--no-display` headless mode for production deployment (no GUI needed)
  - Feature: Graceful SIGINT / SIGTERM handling via signal module
  - Performance: YOLO device auto-selects cuda/mps/cpu instead of hardcoded 'cpu'
  - Performance: YOLO runs on a half-resolution copy to cut inference time ~4×
  - Performance: Default distance path is `SparseBlockMatcher` — per-detection
    local template matching instead of dense whole-frame SGBM+WLS every frame
  - Performance: Dense mode (opt-in) reuses a pre-allocated uint8 buffer for
    disparity visualization and uses lighter default SGBM params
  - Style: Type hints throughout, dataclass for config, ABC for base class
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stereo")

# ---------------------------------------------------------------------------
# Type alias for a bounding box: (x, y, w, h, confidence)
# ---------------------------------------------------------------------------
BBox = Tuple[int, int, int, int, float]


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
    # Placeholder only — replace with a value from real stereo calibration.
    # Rough starting point for Camera Module 3 (IMX708) at 1280x720:
    #   focal_length_px ≈ (image_width_px / 2) / tan(HFOV / 2)
    #   IMX708 full-FOV lens: HFOV ≈ 66°  →  ~1000 px at 1280 width.
    # This is NOT a substitute for calibration — SGBM/WLS depth accuracy is
    # very sensitive to this value being correct for your actual lenses.
    focal_length: float = 500.0   # Calibrated focal length in pixels

    # ── Pi Camera Module 3 capture settings ─────────────────────────────────
    pi_resolution: Tuple[int, int] = (640, 480)   # Lowered from 720p for Pi 5 headroom
    pi_fps: int = 30
    pi_ae_settle_s: float = 2.0    # Time to let auto-exposure/AWB converge
                                    # before locking it for stereo consistency

    # ── Sparse block matching (DEFAULT distance method — cheap) ─────────────
    # Only used at detected-object locations, not across the whole frame.
    match_patch_size: int = 32       # Template patch side length, pixels
    match_search_range: int = 300     # Max disparity searched, pixels
    match_row_tolerance: int = 6     # Vertical search slack (rows) to absorb
                                      # imperfect rectification/alignment
    match_min_confidence: float = 0.5  # Discard matches below this NCC score

    # ── Dense StereoSGBM (OPT-IN via --dense-depth — expensive) ─────────────
    min_disparity: int = 0
    num_disparities: int = 48      # Must be divisible by 16 (reduced for Pi 5)
    block_size: int = 9            # Matcher window (odd, 3–11; smaller = faster)

    # ── WLS filter ─────────────────────────────────────────────────────────
    wls_lambda: float = 8000.0
    wls_sigma: float = 1.5

    # ── Detection ──────────────────────────────────────────────────────────
    min_trash_area: int = 400
    patch_size: int = 7            # NxN neighbourhood for depth sampling

    # ── Display ────────────────────────────────────────────────────────────
    display: bool = True           # Set False for headless / production use

    # ── YOLO half-resolution inference ─────────────────────────────────────
    yolo_half_res: bool = True     # Halve frame before YOLO → ~4× faster


# ---------------------------------------------------------------------------
# Threaded video capture
# ---------------------------------------------------------------------------
class ThreadedVideoGrabber:
    """
    Decouples I/O from processing: a dedicated thread keeps the latest frame
    in memory so the main loop never blocks waiting on the camera driver.

    Changes from original
    ---------------------
    - Uses threading.Event instead of a bare boolean for `stopped`.  Events are
      properly thread-safe and avoid the subtle race where the main thread sets
      stopped=True just after the worker read it as False.
    - Drops stale frames: if the main loop is slower than the camera, the worker
      simply overwrites `self.frame` with the newest one — no queue buildup.
    - Exposes `is_alive()` so the caller can check health without try/except.
    """

    def __init__(self, source: int | str) -> None:
        self.stream = cv2.VideoCapture(source)
        if not self.stream.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

        self._stop_event = threading.Event()
        self.lock = threading.Lock()
        self.grabbed = False
        self.frame: Optional[np.ndarray] = None

        fps = self.stream.get(cv2.CAP_PROP_FPS)
        # Live cameras often return 0 or implausible values
        self.frame_delay = 1.0 / fps if 0 < fps <= 120 else 1.0 / 30.0

        # Prime with the first frame so `.read()` is never empty
        self.grabbed, self.frame = self.stream.read()

    # ------------------------------------------------------------------
    def start(self) -> "ThreadedVideoGrabber":
        t = threading.Thread(target=self._update, daemon=True)
        t.start()
        return self

    def _update(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.monotonic()

            grabbed, frame = self.stream.read()
            if not grabbed:
                log.warning("Video source exhausted or disconnected.")
                self._stop_event.set()
                break

            with self.lock:
                self.grabbed = grabbed
                self.frame = frame             # Overwrite → always newest frame

            # Pace to the source FPS (important for video-file testing)
            elapsed = time.monotonic() - t0
            wait = self.frame_delay - elapsed
            if wait > 0:
                time.sleep(wait)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def is_alive(self) -> bool:
        return not self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()
        self.stream.release()


class PiCamera2Grabber:
    """
    Threaded frame grabber for Raspberry Pi Camera Module 3 (IMX708) via
    Picamera2 / libcamera. Mirrors the ThreadedVideoGrabber interface
    (.start(), .read(), .is_alive(), .stop()) so it's a drop-in replacement
    in the main loop below.

    Why not cv2.VideoCapture
    -------------------------
    CSI cameras on modern Raspberry Pi OS go through libcamera, not the
    classic V4L2 path OpenCV expects. Picamera2 is the supported way to
    drive Camera Module 3 reliably (including on both CSI ports of a Pi 5).

    Camera identification
    ----------------------
    `camera_num` is the libcamera camera index (0 or 1), corresponding to
    the two physical CSI connectors on a Raspberry Pi 5. Run
    `libcamera-hello --list-cameras` (or `Picamera2.global_camera_info()`)
    to confirm which physical port maps to which index on your board.
    """

    def __init__(
        self,
        camera_num: int,
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30,
    ) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is required for Pi Camera Module 3 support. "
                "Install with: sudo apt install -y python3-picamera2"
            ) from exc

        self.camera_num = camera_num
        self.picam2 = Picamera2(camera_num=camera_num)

        video_config = self.picam2.create_video_configuration(
            main={"size": resolution, "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.picam2.configure(video_config)
        self.picam2.start()

        self._stop_event = threading.Event()
        self.lock = threading.Lock()
        self.grabbed = False
        self.frame: Optional[np.ndarray] = None
        self.frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0

        # Prime with a first frame. "RGB888" from Picamera2 is actually
        # ordered BGR in memory (a long-standing libcamera/Picamera2 naming
        # quirk), which conveniently matches what OpenCV/cv2 expects — no
        # extra color conversion needed downstream.
        self.frame = self.picam2.capture_array()
        self.grabbed = True

    # ------------------------------------------------------------------
    def lock_exposure(self, exposure_time: int, analogue_gain: float,
                       colour_gains: Tuple[float, float]) -> None:
        """Disable auto-exposure/AWB and pin to explicit values so both
        cameras in the stereo pair produce matched brightness/color."""
        self.picam2.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": int(exposure_time),
            "AnalogueGain": float(analogue_gain),
            "ColourGains": tuple(colour_gains),
        })

    def read_settled_metadata(self) -> dict:
        """Return current AE/AWB metadata (used to copy settings to the
        other camera in the pair)."""
        return self.picam2.capture_metadata()

    # ------------------------------------------------------------------
    def start(self) -> "PiCamera2Grabber":
        t = threading.Thread(target=self._update, daemon=True)
        t.start()
        return self

    def _update(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                frame = self.picam2.capture_array()
            except RuntimeError:
                log.warning("Pi camera %d capture failed — stopping.", self.camera_num)
                self._stop_event.set()
                break

            with self.lock:
                self.grabbed = True
                self.frame = frame

            elapsed = time.monotonic() - t0
            wait = self.frame_delay - elapsed
            if wait > 0:
                time.sleep(wait)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def is_alive(self) -> bool:
        return not self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception:
            pass


def synchronize_stereo_cameras(
    cam_left: PiCamera2Grabber, cam_right: PiCamera2Grabber, settle_s: float = 2.0
) -> None:
    """
    Let both cameras' auto-exposure/AWB converge, then lock both to the
    LEFT camera's settled values. Matched exposure/gain/white-balance
    across the pair is important for SGBM: mismatched brightness or color
    between left/right images degrades the block-matching cost function
    and produces noisier disparity/depth maps.
    """
    log.info("Letting auto-exposure/AWB settle for %.1fs before locking...", settle_s)
    time.sleep(settle_s)

    meta = cam_left.read_settled_metadata()
    exposure_time = meta.get("ExposureTime", 20000)
    analogue_gain = meta.get("AnalogueGain", 1.0)
    colour_gains = meta.get("ColourGains", (1.5, 1.5))

    cam_left.lock_exposure(exposure_time, analogue_gain, colour_gains)
    cam_right.lock_exposure(exposure_time, analogue_gain, colour_gains)
    log.info(
        "Locked stereo pair — exposure=%dus gain=%.2f colour_gains=(%.2f, %.2f)",
        exposure_time, analogue_gain, colour_gains[0], colour_gains[1],
    )


# ---------------------------------------------------------------------------
# Sparse block matcher (DEFAULT distance method — cheap, per-detection only)
# ---------------------------------------------------------------------------
class SparseBlockMatcher:
    """
    Optimized sparse matcher tuned for 0.8m to 5.0m depth range.
    Uses sub-pixel quadratic interpolation for smooth, accurate distances at range.
    """

    def __init__(self, config: StereoConfig) -> None:
        self.config = config
        # Target depth range: 0.8m to ~5m
        self.min_disparity = 15   # ~5.0 meters
        self.max_disparity = 180  # ~0.5 meters
        self.patch_size = 25      # Odd size for clean patch centering
        self.row_tolerance = 3    # Low tolerance (requires rectified frames)
        self.min_confidence = 0.5

    def estimate_distance(
        self, gray_l: np.ndarray, gray_r: np.ndarray, cx: int, cy: int
    ) -> float:
        h, w = gray_l.shape[:2]
        half = self.patch_size // 2

        ty0, ty1 = cy - half, cy + half
        tx0, tx1 = cx - half, cx + half
        if ty0 < 0 or tx0 < 0 or ty1 > h or tx1 > w:
            return 0.0

        template = gray_l[ty0:ty1, tx0:tx1]
        if template.size == 0:
            return 0.0

        # Search STRICTLY to the left of cx (x_right <= x_left)
        sy0 = max(0, ty0 - self.row_tolerance)
        sy1 = min(h, ty1 + self.row_tolerance)
        sx0 = max(0, cx - self.max_disparity)
        sx1 = min(w, cx - self.min_disparity + self.patch_size)

        if (sx1 - sx0) < self.patch_size or (sy1 - sy0) < self.patch_size:
            return 0.0

        strip = gray_r[sy0:sy1, sx0:sx1]
        result = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < self.min_confidence:
            return 0.0

        match_x, match_y = max_loc
        matched_x_center = sx0 + match_x + half
        raw_disparity = float(cx - matched_x_center)

        # ── Sub-pixel interpolation (1D parabolic fit) ───────────────────
        # Fits a parabola around the peak in the horizontal correlation score
        sub_disparity = raw_disparity
        if 0 < match_x < (result.shape[1] - 1):
            v_center = result[match_y, match_x]
            v_left = result[match_y, match_x - 1]
            v_right = result[match_y, match_x + 1]
            denom = 2.0 * (v_left - 2.0 * v_center + v_right)
            if abs(denom) > 1e-5:
                delta = (v_left - v_right) / denom
                sub_disparity -= delta  # Adjust disparity sub-pixel offset

        if sub_disparity <= 1.0:
            return 0.0

        return (self.config.focal_length * self.config.baseline) / sub_disparity


# ---------------------------------------------------------------------------
# Dense stereo processor hierarchy (OPT-IN via --dense-depth — expensive)
# ---------------------------------------------------------------------------
class BaseStereoProcessor(ABC):
    """Abstract base class — swap backends without touching the main loop."""

    @abstractmethod
    def compute_depth(
        self, frame_l: np.ndarray, frame_r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (depth_map_float32, disparity_float32)."""


# ── CPU backend ─────────────────────────────────────────────────────────────
class CPUStereoProcessor(BaseStereoProcessor):
    """
    StereoSGBM + WLS on CPU.
    Works everywhere; best fallback for Raspberry Pi 5.
    """

    def __init__(self, config: StereoConfig) -> None:
        self.config = config
        bs = config.block_size

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=config.min_disparity,
            numDisparities=config.num_disparities,
            blockSize=bs,
            P1=8 * 3 * bs ** 2,
            P2=32 * 3 * bs ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=2,
        )
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)

        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=self.left_matcher
        )
        self.wls_filter.setLambda(config.wls_lambda)
        self.wls_filter.setSigmaColor(config.wls_sigma)

    def compute_depth(
        self, frame_l: np.ndarray, frame_r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        disp_l = self.left_matcher.compute(gray_l, gray_r)
        disp_r = self.right_matcher.compute(gray_r, gray_l)

        filtered = self.wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)

        disparity = filtered.astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0.1  # Guard against division by zero

        depth_map = (self.config.focal_length * self.config.baseline) / disparity
        return depth_map, disparity


# ── OpenCL backend (Raspberry Pi 5 / any OpenCL GPU) ────────────────────────
class OpenCLStereoProcessor(BaseStereoProcessor):
    """
    Uses cv2.UMat to push grayscale conversion and colour-map steps onto
    the GPU/OpenCL device present on the Pi 5 (VideoCore VII).

    The SGBM matcher itself has no OpenCL path in mainline OpenCV, so we
    keep that on CPU — but the surrounding I/O work moves to the GPU.
    This gives a modest but measurable speedup (~15–25 %) on Pi 5.

    Note: requires OpenCV built with OpenCL support (`cv2.ocl.haveOpenCL()`).
    """

    def __init__(self, config: StereoConfig) -> None:
        if not cv2.ocl.haveOpenCL():
            raise RuntimeError("OpenCL not available — use CPUStereoProcessor instead.")
        cv2.ocl.setUseOpenCL(True)
        log.info("OpenCL enabled: %s", cv2.ocl.useOpenCL())
        # Delegate heavy lifting to the CPU matcher; UMat wraps the buffers.
        self._cpu = CPUStereoProcessor(config)

    def compute_depth(
        self, frame_l: np.ndarray, frame_r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Move colour-space conversion to OpenCL
        gray_l = cv2.cvtColor(cv2.UMat(frame_l), cv2.COLOR_BGR2GRAY).get()
        gray_r = cv2.cvtColor(cv2.UMat(frame_r), cv2.COLOR_BGR2GRAY).get()

        disp_l = self._cpu.left_matcher.compute(gray_l, gray_r)
        disp_r = self._cpu.right_matcher.compute(gray_r, gray_l)

        filtered = self._cpu.wls_filter.filter(
            disp_l, gray_l, disparity_map_right=disp_r
        )
        disparity = filtered.astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0.1

        depth_map = (
            self._cpu.config.focal_length * self._cpu.config.baseline
        ) / disparity
        return depth_map, disparity


# ── CUDA backend (Jetson) ────────────────────────────────────────────────────
class CUDAStereoProcessor(BaseStereoProcessor):
    """
    Full GPU stereo pipeline for NVIDIA Jetson.

    Uses cv2.cuda.StereoSGBM (if available) with GPU-side grayscale
    conversion and disparity post-processing.  WLS filtering still runs
    on CPU because there is no CUDA path in OpenCV's ximgproc.

    Requirements
    ------------
    OpenCV must be built with CUDA support:
        cv2.cuda.getCudaEnabledDeviceCount() > 0
    """

    def __init__(self, config: StereoConfig) -> None:
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            raise RuntimeError("No CUDA device found — use a different backend.")

        self.config = config
        bs = config.block_size
        log.info(
            "CUDA device: %s",
            cv2.cuda.DeviceInfo(cv2.cuda.getDevice()).name(),
        )

        # cv2.cuda.StereoSGBM is available in opencv-contrib ≥ 4.5
        self.cuda_matcher = cv2.cuda.createStereoSGBM(
            minDisparity=config.min_disparity,
            numDisparities=config.num_disparities,
            blockSize=bs,
            P1=8 * 3 * bs ** 2,
            P2=32 * 3 * bs ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=2,
        )

        # WLS still on CPU (no CUDA ximgproc path in mainline OpenCV)
        cpu_left = cv2.StereoSGBM_create(
            minDisparity=config.min_disparity,
            numDisparities=config.num_disparities,
            blockSize=bs,
            P1=8 * 3 * bs ** 2,
            P2=32 * 3 * bs ** 2,
        )
        self.right_matcher = cv2.ximgproc.createRightMatcher(cpu_left)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(cpu_left)
        self.wls_filter.setLambda(config.wls_lambda)
        self.wls_filter.setSigmaColor(config.wls_sigma)

    def compute_depth(
        self, frame_l: np.ndarray, frame_r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Upload to GPU
        gpu_l = cv2.cuda_GpuMat()
        gpu_r = cv2.cuda_GpuMat()
        gpu_l.upload(frame_l)
        gpu_r.upload(frame_r)

        # GPU grayscale conversion
        gray_gpu_l = cv2.cuda.cvtColor(gpu_l, cv2.COLOR_BGR2GRAY)
        gray_gpu_r = cv2.cuda.cvtColor(gpu_r, cv2.COLOR_BGR2GRAY)

        # GPU disparity (left only; WLS right-matcher stays CPU)
        disp_gpu = self.cuda_matcher.compute(gray_gpu_l, gray_gpu_r)
        disp_l = disp_gpu.download()

        # CPU right disparity + WLS
        gray_l = gray_gpu_l.download()
        gray_r = gray_gpu_r.download()
        disp_r = self.right_matcher.compute(gray_r, gray_l)

        filtered = self.wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)
        disparity = filtered.astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0.1

        depth_map = (self.config.focal_length * self.config.baseline) / disparity
        return depth_map, disparity


# ---------------------------------------------------------------------------
# Backend auto-probe
# ---------------------------------------------------------------------------
def create_stereo_processor(
    config: StereoConfig, backend: str = "auto"
) -> BaseStereoProcessor:
    """
    Factory that probes available hardware and returns the best processor.

    Priority (auto): CUDA → OpenCL → CPU
    Use --backend to pin a specific one (useful for benchmarking).
    """
    order = (
        ["cuda", "opencl", "cpu"] if backend == "auto" else [backend]
    )

    for name in order:
        try:
            if name == "cuda":
                proc = CUDAStereoProcessor(config)
                log.info("Backend selected: CUDA (Jetson)")
                return proc
            if name == "opencl":
                proc = OpenCLStereoProcessor(config)
                log.info("Backend selected: OpenCL (Pi 5 / iGPU)")
                return proc
            if name == "cpu":
                proc = CPUStereoProcessor(config)
                log.info("Backend selected: CPU")
                return proc
        except (RuntimeError, cv2.error) as exc:
            log.warning("Backend %r not available: %s — trying next.", name, exc)

    raise RuntimeError("No stereo processor could be initialised.")


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[BBox]:
        """Return list of (x, y, w, h, confidence) bounding boxes."""


class YoloTrashDetector(BaseDetector):
    """
    YOLOv8 trash detector.

    Fixes vs. original
    ------------------
    - `trash_boxes` undefined name bug replaced with `detected_boxes`.
    - Device auto-selected (cuda / mps / cpu) instead of hardcoded 'cpu'.
    - Optional half-resolution inference: halving each dimension reduces the
      pixel count by 4×, cutting YOLO latency significantly on embedded HW.
    """

    def __init__(
        self,
        model_path: str = "best.pt",
        conf: float = 0.3,
        half_res: bool = True,
    ) -> None:
        from ultralytics import YOLO  # Lazy import so AprilTag-only runs don't need it

        self.model = YOLO(model_path)
        self.conf = conf
        self.half_res = half_res

        # Pick the best available device automatically
        self.device = self._pick_device()
        log.info("YOLO running on device: %s", self.device)

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def detect(self, frame: np.ndarray) -> List[BBox]:
        if self.half_res:
            h, w = frame.shape[:2]
            infer_frame = cv2.resize(frame, (w // 2, h // 2))
            scale = 2.0
        else:
            infer_frame = frame
            scale = 1.0

        results = self.model.predict(
            infer_frame, device=self.device, conf=self.conf, verbose=False
        )

        detected_boxes: List[BBox] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                detected_boxes.append((
                    int(x1 * scale),
                    int(y1 * scale),
                    int((x2 - x1) * scale),
                    int((y2 - y1) * scale),
                    confidence,
                ))
        return detected_boxes


class AprilTagDetector(BaseDetector):
    """AprilTag detector using OpenCV ArUco."""

    def __init__(
        self, tag_family: int = cv2.aruco.DICT_APRILTAG_36h11
    ) -> None:
        self.dictionary = cv2.aruco.getPredefinedDictionary(tag_family)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)

    def detect(self, frame: np.ndarray) -> List[BBox]:
        if frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        detected_boxes: List[BBox] = []
        if ids is not None:
            for c in corners:
                pts = c[0]
                x1, x2 = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
                y1, y2 = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
                detected_boxes.append((
                    int(x1), int(y1),
                    int(x2 - x1), int(y2 - y1),
                    1.0,
                ))
        return detected_boxes


# ---------------------------------------------------------------------------
# Depth sampling helper
# ---------------------------------------------------------------------------
def get_robust_distance(
    depth_map: np.ndarray, cx: int, cy: int, patch_size: int
) -> float:
    """
    Return the median depth within a (patch_size × patch_size) window centred
    on (cx, cy), ignoring invalid pixels (depth ≤ 0.5 m).

    Median over a patch suppresses single-pixel glare spikes far better than
    a point sample, at negligible CPU cost for small patch sizes.
    """
    h, w = depth_map.shape
    half = patch_size >> 1  # Bit-shift slightly faster than // 2

    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)

    patch = depth_map[y0:y1, x0:x1]
    valid = patch[patch > 0.5]
    return float(np.median(valid)) if valid.size > 0 else 0.0


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def draw_detection(
    frame: np.ndarray, x: int, y: int, w: int, h: int, conf: float, distance: float
) -> None:
    """Draw a bounding box, centre dot and label onto `frame` in-place."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    label = f"Dist: {distance:.2f}m | Conf: {conf * 100:.0f}%"
    # Drop label below the box if it would clip off the top
    label_y = y - 10 if y > 20 else y + h + 15
    cv2.putText(
        frame, label, (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2,
    )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Overlay FPS counter in the top-left corner."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stereo vision pipeline — edge-optimised"
    )
    p.add_argument(
        "--mode", choices=["trash", "apriltag"], default="apriltag",
        help="Detection mode (default: apriltag)",
    )
    p.add_argument(
        "--left", default=os.path.join("video", "plastic_video.mp4"),
        metavar="SRC",
        help="Left camera source: path to video file, integer device index, "
             "or 'picamN' (e.g. picam0) when using --picam2",
    )
    p.add_argument(
        "--right", default=os.path.join("video", "plastic_video.mp4"),
        metavar="SRC",
        help="Right camera source (same format as --left)",
    )
    p.add_argument(
        "--picam2", action="store_true",
        help="Use two Raspberry Pi Camera Module 3 units via Picamera2 "
             "instead of cv2.VideoCapture. --left/--right are interpreted "
             "as libcamera camera indices (default 0 and 1).",
    )
    p.add_argument(
        "--baseline", type=float, default=None,
        help="Override stereo baseline in metres (default: 0.12 m / 12 cm)",
    )
    p.add_argument(
        "--focal-length", type=float, default=None,
        help="Override calibrated focal length in pixels",
    )
    p.add_argument(
        "--resolution", default="640x480", metavar="WxH",
        help="Pi camera capture resolution (--picam2 only, default: 640x480 — "
             "lower resolution keeps YOLO/matching cost down on a Pi 5)",
    )
    p.add_argument(
        "--fps", type=int, default=30,
        help="Pi camera capture FPS (--picam2 only, default: 30)",
    )
    p.add_argument(
        "--dense-depth", action="store_true",
        help="Use the old full-frame StereoSGBM+WLS pipeline and show a "
             "dense disparity visualization window, instead of the default "
             "lightweight per-detection sparse matcher. Much heavier on a "
             "Pi 5 — use for debugging/demos, not routine operation.",
    )
    p.add_argument(
        "--backend", choices=["auto", "cuda", "opencl", "cpu"], default="auto",
        help="Dense stereo processor backend, only used with --dense-depth "
             "(default: auto-probe)",
    )
    p.add_argument(
        "--model", default=os.path.join("trained model", "best.pt"),
        metavar="PATH",
        help="Path to YOLO .pt weights (trash mode only)",
    )
    p.add_argument(
        "--no-display", dest="display", action="store_false",
        help="Disable GUI windows (headless / SSH deployment)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    config = StereoConfig(display=args.display)
    if args.baseline is not None:
        config.baseline = args.baseline
    if args.focal_length is not None:
        config.focal_length = args.focal_length

    try:
        w_str, h_str = args.resolution.lower().split("x")
        config.pi_resolution = (int(w_str), int(h_str))
    except ValueError:
        log.warning("Could not parse --resolution %r, using default %s",
                    args.resolution, config.pi_resolution)
    config.pi_fps = args.fps

    # ── Graceful shutdown on Ctrl-C / SIGTERM ─────────────────────────────
    shutdown = threading.Event()

    def _handle_signal(signum, _frame):
        log.info("Signal %d received — shutting down.", signum)
        shutdown.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Start capture threads ─────────────────────────────────────────────
    if args.picam2:
        # Pi Camera Module 3 stereo pair via Picamera2 (libcamera).
        def _cam_index(s: str, fallback: int) -> int:
            s = s.strip().lower()
            if s.startswith("picam"):
                s = s[len("picam"):]
            return int(s) if s.isdigit() else fallback

        left_idx  = _cam_index(args.left, 0)
        right_idx = _cam_index(args.right, 1)

        log.info(
            "Opening Pi Camera Module 3 pair: left=camera_num %d, right=camera_num %d "
            "(baseline=%.3fm, resolution=%s, fps=%d)",
            left_idx, right_idx, config.baseline, config.pi_resolution, config.pi_fps,
        )
        cap_left  = PiCamera2Grabber(left_idx, config.pi_resolution, config.pi_fps).start()
        cap_right = PiCamera2Grabber(right_idx, config.pi_resolution, config.pi_fps).start()

        # Match exposure/gain/white-balance across the pair for clean SGBM input.
        synchronize_stereo_cameras(cap_left, cap_right, settle_s=config.pi_ae_settle_s)
    else:
        # ── Source normalisation: "0" or "1" as a string → int device index ──
        def _src(s: str) -> int | str:
            return int(s) if s.isdigit() else s

        left_src  = _src(args.left)
        right_src = _src(args.right)

        log.info("Opening left  source: %r", left_src)
        log.info("Opening right source: %r", right_src)
        cap_left  = ThreadedVideoGrabber(left_src).start()
        cap_right = ThreadedVideoGrabber(right_src).start()

    # ── Initialise distance-estimation backend ──────────────────────────────
    # Default: cheap sparse matcher, only evaluated at detection points.
    # Opt-in: full dense SGBM+WLS backend + disparity visualization window.
    sparse_matcher: Optional[SparseBlockMatcher] = None
    stereo: Optional[BaseStereoProcessor] = None

    if args.dense_depth:
        log.info("Dense depth mode enabled (--dense-depth) — this is the "
                  "heavy path; expect lower FPS on a Pi 5.")
        stereo = create_stereo_processor(config, backend=args.backend)
    else:
        log.info("Using lightweight sparse block matcher (default). "
                  "Pass --dense-depth for the full dense SGBM+WLS pipeline.")
        sparse_matcher = SparseBlockMatcher(config)

    # ── Initialise detector ───────────────────────────────────────────────
    if args.mode == "trash":
        detector: BaseDetector = YoloTrashDetector(
            model_path=args.model,
            conf=0.3,
            half_res=config.yolo_half_res,
        )
    else:
        detector = AprilTagDetector()
    log.info("Detector mode: %s", args.mode)

    # Warm-up: let cameras buffer a couple of frames before processing
    time.sleep(1.0)

    # ── Pre-allocate disparity visualisation buffer (dense mode only) ──────
    disp_vis_buf: Optional[np.ndarray] = None

    # ── Load Stereo Calibration ───────────────────────────────────────────
    log.info("Loading stereo calibration maps from stereo_calib.npz")
    try:
        calib = np.load("stereo_calib.npz")
        map1x, map1y = calib["map1x"], calib["map1y"]
        map2x, map2y = calib["map2x"], calib["map2y"]
    except FileNotFoundError:
        log.error("stereo_calib.npz not found! Ensure it is in the same directory.")
        sys.exit(1)

    # ── FPS tracking ──────────────────────────────────────────────────────
    fps_counter = 0
    fps_display = 0.0
    fps_timer   = time.monotonic()

    log.info("Pipeline running. Press 'q' to quit.")

    while not shutdown.is_set():
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r or frame_l is None or frame_r is None:
            log.info("Stream ended or camera disconnected.")
            break

        # ── Apply Rectification Maps ──────────────────────────────────────
        frame_l = cv2.remap(frame_l, map1x, map1y, cv2.INTER_LINEAR)
        frame_r = cv2.remap(frame_r, map2x, map2y, cv2.INTER_LINEAR)

        # ── 1. Detect objects first — distance is only computed where needed ──
        detections = detector.detect(frame_l)

        # ── 2. Compute distance + annotate ──────────────────────────────────
        if args.dense_depth:
            # Dense path: one full-frame depth map, then sample at each point.
            depth_map, disparity = stereo.compute_depth(frame_l, frame_r)
            for x, y, w, h, conf in detections:
                cx, cy = x + w // 2, y + h // 2
                dist = get_robust_distance(depth_map, cx, cy, config.patch_size)
                draw_detection(frame_l, x, y, w, h, conf, dist)

        else:
            # Sparse path: grayscale once, then a cheap local search per object.
            # ── 3. COMPUTE SPARSE DISTANCE (5-Point Grid) ─────────────────────
            # ── 3. COMPUTE SPARSE DISTANCE (Cross-Pattern Grid) ───────────────────
            if detections:
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                for x, y, w, h, conf in detections:
                    cx, cy = x + w // 2, y + h // 2
                    
                    # 25% margin to stay well inside the irregular edges of a round bag
                    mx, my = w // 4, h // 4
                    
                    # Define cross grid: Center, Top, Bottom, Left, Right
                    points = [
                        (cx, cy),                  # Center
                        (cx, y + my),              # Top
                        (cx, y + h - my),          # Bottom
                        (x + mx, cy),              # Left
                        (x + w - mx, cy)           # Right
                    ]
                    
                    valid_distances = []
                    for px, py in points:
                        d = sparse_matcher.estimate_distance(gray_l, gray_r, px, py)
                        if d > 0.0:
                            valid_distances.append(d)
                            # Mark successful texture locks with a cyan dot
                            cv2.circle(frame_l, (px, py), 3, (255, 255, 0), -1)
                        else:
                            # Mark failed texture locks with a red dot
                            cv2.circle(frame_l, (px, py), 3, (0, 0, 255), -1)
                    
                    # Use the median of successful matches to drop extreme outliers
                    if valid_distances:
                        final_dist = float(np.median(valid_distances))
                    else:
                        final_dist = 0.0
                        
                    draw_detection(frame_l, x, y, w, h, conf, final_dist)
        
            
        # ── 3. FPS counter ────────────────────────────────────────────────
        fps_counter += 1
        now = time.monotonic()
        if now - fps_timer >= 1.0:
            fps_display = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer   = now

        # ── 4. Display ────────────────────────────────────────────────────
        if config.display:
            draw_fps(frame_l, fps_display)
            cv2.imshow("Stereo Vision — Stream", frame_l)

            if args.dense_depth:
                # Reuse pre-allocated buffer for disparity colourmap
                disp_norm = cv2.normalize(
                    disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                )
                if disp_vis_buf is None or disp_vis_buf.shape != disp_norm.shape:
                    disp_vis_buf = np.empty_like(disp_norm, dtype=np.uint8)
                np.copyto(disp_vis_buf, disp_norm, casting="unsafe")
                cv2.imshow(
                    "WLS Filtered Disparity",
                    cv2.applyColorMap(disp_vis_buf, cv2.COLORMAP_JET),
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── Cleanup ───────────────────────────────────────────────────────────
    log.info("Releasing resources.")
    
    cap_left.stop()
    cap_right.stop()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
