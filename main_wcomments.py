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

Key improvements over original:
  - Refactor: Abstracted configuration to external `config.py`
  - Refactor: Generalized `YoloTrashDetector` to a universal `YoloDetector`
  - Performance: YOLO runs on a half-resolution copy to cut inference time ~4×
  - Performance: Default distance path is `SparseBlockMatcher` 
"""

from __future__ import annotations

import argparse
import logging
from math import dist
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Abstracted configuration state payload
import config
from config import StereoConfig

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
# Establishes a thread-safe stdout logger for asynchronous execution tracing.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stereo")


# Type alias for bounding box schema: (x, y, width, height, confidence)
BBox = Tuple[int, int, int, int, float]


# ---------------------------------------------------------------------------
# I/O Acquisition Layer
# ---------------------------------------------------------------------------
class ThreadedVideoGrabber:
    """
    Asynchronous V4L2 video stream handler.
    Decouples I/O blocking operations from the main inference loop. 
    Implements a write-preferring lock to guarantee the inference engine 
    always receives the most recent frame, dropping stale frames organically.
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
        # Fallback heuristic for generic webcams returning invalid metadata
        self.frame_delay = 1.0 / fps if 0 < fps <= 120 else 1.0 / 30.0

        # Prime the buffer to prevent NoneType exceptions on immediate read
        self.grabbed, self.frame = self.stream.read()

    def start(self) -> "ThreadedVideoGrabber":
        t = threading.Thread(target=self._update, daemon=True)
        t.start()
        return self

    def _update(self) -> None:
        """Background daemon polling the hardware buffer."""
        while not self._stop_event.is_set():
            t0 = time.monotonic()

            grabbed, frame = self.stream.read()
            if not grabbed:
                log.warning("Video source exhausted or disconnected.")
                self._stop_event.set()
                break

            # Mutex lock strictly limits scope to reference assignment
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame             

            # Throttle loop to match target FPS, preventing CPU spin-locking
            elapsed = time.monotonic() - t0
            wait = self.frame_delay - elapsed
            if wait > 0:
                time.sleep(wait)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Deep copies the frame payload to prevent race conditions during inference."""
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
    libcamera (Picamera2) integration for native Raspberry Pi CSI pipelines.
    Provides direct memory access (DMA) to ISP frames, bypassing inefficient V4L2 bridging.
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

        # Configure libcamera pipeline for RGB888 (which maps natively to OpenCV's BGR layout)
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

        self.frame = self.picam2.capture_array()
        self.grabbed = True

    def lock_exposure(self, exposure_time: int, analogue_gain: float,
                       colour_gains: Tuple[float, float]) -> None:
        """Overrides internal ISP algorithms to force static photometric properties."""
        self.picam2.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": int(exposure_time),
            "AnalogueGain": float(analogue_gain),
            "ColourGains": tuple(colour_gains),
        })

    def read_settled_metadata(self) -> dict:
        return self.picam2.capture_metadata()

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
    Forces photometric consistency across the stereo pair.
    SGBM and NCC algorithms fail drastically if left/right sensors have differing 
    exposure or white balance. This function allows the master ISP to converge, 
    extracts the settled state, and statically locks both sensors to identical values.
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
# Sparse Distance Estimation Layer
# ---------------------------------------------------------------------------
class SparseBlockMatcher:
    """
    Highly optimized, region-of-interest normalized cross-correlation (NCC) matcher.
    O(N) complexity where N is the number of detected objects, avoiding the O(W*H) 
    complexity of dense disparity mapping.
    """
    def __init__(self, config: StereoConfig) -> None:
        self.config = config
        self.min_disparity = 15   
        self.max_disparity = 180  
        self.patch_size = 25      
        self.row_tolerance = 3    
        self.min_confidence = 0.5

    def estimate_distance(
        self, gray_l: np.ndarray, gray_r: np.ndarray, cx: int, cy: int
    ) -> float:
        """
        Executes localized epipolar search. Extracts a template from the left frame
        and sweeps across the corresponding epipolar line segment in the right frame.
        Applies parabolic interpolation for sub-pixel disparity resolution.
        """
        h, w = gray_l.shape[:2]
        half = self.patch_size // 2

        # Define bounds for the reference patch (left frame)
        ty0, ty1 = cy - half, cy + half
        tx0, tx1 = cx - half, cx + half
        if ty0 < 0 or tx0 < 0 or ty1 > h or tx1 > w:
            return 0.0

        template = gray_l[ty0:ty1, tx0:tx1]
        if template.size == 0:
            return 0.0

        # Define bounded horizontal search space (right frame)
        # Search constrained strictly leftward to respect epipolar geometry
        sy0 = max(0, ty0 - self.row_tolerance)
        sy1 = min(h, ty1 + self.row_tolerance)
        sx0 = max(0, cx - self.max_disparity)
        sx1 = min(w, cx - self.min_disparity + self.patch_size)

        if (sx1 - sx0) < self.patch_size or (sy1 - sy0) < self.patch_size:
            return 0.0

        # Execute normalized cross-correlation
        strip = gray_r[sy0:sy1, sx0:sx1]
        result = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < self.min_confidence:
            return 0.0

        match_x, match_y = max_loc
        matched_x_center = sx0 + match_x + half
        raw_disparity = float(cx - matched_x_center)

        # 1D Parabolic sub-pixel interpolation around the response peak
        sub_disparity = raw_disparity
        if 0 < match_x < (result.shape[1] - 1):
            v_center = result[match_y, match_x]
            v_left = result[match_y, match_x - 1]
            v_right = result[match_y, match_x + 1]
            denom = 2.0 * (v_left - 2.0 * v_center + v_right)
            if abs(denom) > 1e-5:
                delta = (v_left - v_right) / denom
                sub_disparity -= delta  

        if sub_disparity <= 1.0:
            return 0.0

        # Depth mapping via standard triangulation formula: Z = (f * B) / d
        return (self.config.focal_length * self.config.baseline) / sub_disparity


# ---------------------------------------------------------------------------
# Dense Distance Estimation Layer (Opt-In Hardware Accelerated)
# ---------------------------------------------------------------------------
class BaseStereoProcessor(ABC):
    @abstractmethod
    def compute_depth(
        self, frame_l: np.ndarray, frame_r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class CPUStereoProcessor(BaseStereoProcessor):
    """Fallback processor utilizing standard AVX/SSE CPU vectorization for SGBM."""
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

        # Normalize WLS integer output to standard disparity units
        disparity = filtered.astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0.1  

        depth_map = (self.config.focal_length * self.config.baseline) / disparity
        return depth_map, disparity


class OpenCLStereoProcessor(BaseStereoProcessor):
    """
    iGPU acceleration using OpenCL Transparent API (T-API).
    Offloads memory-intensive color conversions to the VideoCore VII GPU (Pi 5),
    leaving SGBM logic on the CPU.
    """
    def __init__(self, config: StereoConfig) -> None:
        if not cv2.ocl.haveOpenCL():
            raise RuntimeError("OpenCL not available — use CPUStereoProcessor instead.")
        cv2.ocl.setUseOpenCL(True)
        log.info("OpenCL enabled: %s", cv2.ocl.useOpenCL())
        self._cpu = CPUStereoProcessor(config)

    def compute_depth(
        self, frame_l: np.ndarray, frame_r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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


class CUDAStereoProcessor(BaseStereoProcessor):
    """NVIDIA Jetson optimized pipeline shifting core SGBM execution to CUDA cores."""
    def __init__(self, config: StereoConfig) -> None:
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            raise RuntimeError("No CUDA device found — use a different backend.")
        # [CUDA Implementation Details Omitted for Brevity]
        self.config = config
        bs = config.block_size
        log.info(
            "CUDA device: %s",
            cv2.cuda.DeviceInfo(cv2.cuda.getDevice()).name(),
        )

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
        # Fallback objects for WLS since ximgproc lacks a native CUDA port
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
        # Perform host-to-device (H2D) memory transfer
        gpu_l = cv2.cuda_GpuMat()
        gpu_r = cv2.cuda_GpuMat()
        gpu_l.upload(frame_l)
        gpu_r.upload(frame_r)

        gray_gpu_l = cv2.cuda.cvtColor(gpu_l, cv2.COLOR_BGR2GRAY)
        gray_gpu_r = cv2.cuda.cvtColor(gpu_r, cv2.COLOR_BGR2GRAY)

        # Execute SGBM parallelization on GPU
        disp_gpu = self.cuda_matcher.compute(gray_gpu_l, gray_gpu_r)
        
        # Device-to-host (D2H) transfer for WLS filtration
        disp_l = disp_gpu.download()
        gray_l = gray_gpu_l.download()
        gray_r = gray_gpu_r.download()
        disp_r = self.right_matcher.compute(gray_r, gray_l)

        filtered = self.wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)
        disparity = filtered.astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0.1

        depth_map = (self.config.focal_length * self.config.baseline) / disparity
        return depth_map, disparity


def create_stereo_processor(
    config: StereoConfig, backend: str = "auto"
) -> BaseStereoProcessor:
    """Hardware prober implementing the Factory pattern for dense matching."""
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
# Object Detection Layer
# ---------------------------------------------------------------------------
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[BBox]:
        pass


class YoloDetector(BaseDetector):
    """
    Handles neural network inference via Ultralytics YOLO architecture.
    Auto-detects tensor hardware acceleration (CUDA/MPS) with a CPU fallback.
    """
    def __init__(
        self,
        model_path: str,
        conf: float = 0.3,
        half_res: bool = True,
    ) -> None:
        from ultralytics import YOLO  

        self.model = YOLO(model_path)
        self.conf = conf
        self.half_res = half_res

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
        """
        Executes the forward pass. Downsamples the input tensor when `half_res` 
        is active to significantly reduce MAC operations and latency on embedded CPUs.
        """
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
            # Map GPU/MPS tensors back to host memory (CPU) numpy arrays
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                # Re-scale bounding box coordinates to align with native frame resolution
                detected_boxes.append((
                    int(x1 * scale),
                    int(y1 * scale),
                    int((x2 - x1) * scale),
                    int((y2 - y1) * scale),
                    confidence,
                ))
        return detected_boxes


class AprilTagDetector(BaseDetector):
    """ArUco library wrapper for fiducial marker identification."""
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
# Utility and Execution Logic
# ---------------------------------------------------------------------------
def get_robust_distance(
    depth_map: np.ndarray, cx: int, cy: int, patch_size: int
) -> float:
    """
    Applies spatial median filtering across a patch within the dense depth map.
    Suppresses transient noise spikes and handles dead-zones (disparity=0) gracefully.
    """
    h, w = depth_map.shape
    half = patch_size >> 1  

    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)

    patch = depth_map[y0:y1, x0:x1]
    valid = patch[patch > 0.5]
    return float(np.median(valid)) if valid.size > 0 else 0.0


def draw_detection(
    frame: np.ndarray, x: int, y: int, w: int, h: int, conf: float, distance: float
) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    label = f"Dist: {distance:.2f}m | Conf: {conf * 100:.0f}%"
    label_y = y - 10 if y > 20 else y + h + 15
    cv2.putText(
        frame, label, (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2,
    )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stereo vision pipeline — edge-optimised"
    )
    p.add_argument(
        "--mode", choices=["yolo", "apriltag"], default=config.DEFAULT_MODE,
        help=f"Detection mode (default: {config.DEFAULT_MODE})",
    )
    p.add_argument(
        "--left", default=config.DEFAULT_LEFT_SRC,
        metavar="SRC",
        help="Left camera source: path to video file, integer device index, "
             "or 'picamN' (e.g. picam0) when using --picam2",
    )
    p.add_argument(
        "--right", default=config.DEFAULT_RIGHT_SRC,
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
        "--model", default=config.DEFAULT_MODEL,
        metavar="PATH",
        help=f"Path to YOLO .pt weights (default: {config.DEFAULT_MODEL})",
    )
    p.add_argument(
        "--no-display", dest="display", action="store_false",
        help="Disable GUI windows (headless / SSH deployment)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core Execution Engine
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Hydrate configuration payload
    pipeline_config = StereoConfig(display=args.display)
    if args.baseline is not None:
        pipeline_config.baseline = args.baseline
    if args.focal_length is not None:
        pipeline_config.focal_length = args.focal_length

    try:
        w_str, h_str = args.resolution.lower().split("x")
        pipeline_config.pi_resolution = (int(w_str), int(h_str))
    except ValueError:
        log.warning("Could not parse --resolution %r, using default %s",
                    args.resolution, pipeline_config.pi_resolution)
    pipeline_config.pi_fps = args.fps

    # Configure graceful teardown hooks mapped to standard UNIX signals
    shutdown = threading.Event()

    def _handle_signal(signum, _frame):
        log.info("Signal %d received — shutting down.", signum)
        shutdown.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Initialize frame acquisition topology
    if args.picam2:
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
            left_idx, right_idx, pipeline_config.baseline, pipeline_config.pi_resolution, pipeline_config.pi_fps,
        )
        cap_left  = PiCamera2Grabber(left_idx, pipeline_config.pi_resolution, pipeline_config.pi_fps).start()
        cap_right = PiCamera2Grabber(right_idx, pipeline_config.pi_resolution, pipeline_config.pi_fps).start()

        synchronize_stereo_cameras(cap_left, cap_right, settle_s=pipeline_config.pi_ae_settle_s)
    else:
        def _src(s: str) -> int | str:
            return int(s) if s.isdigit() else s

        left_src  = _src(args.left)
        right_src = _src(args.right)

        log.info("Opening left  source: %r", left_src)
        log.info("Opening right source: %r", right_src)
        cap_left  = ThreadedVideoGrabber(left_src).start()
        cap_right = ThreadedVideoGrabber(right_src).start()

    # Instantiate logic gates for mapping algorithm hierarchy
    sparse_matcher: Optional[SparseBlockMatcher] = None
    stereo: Optional[BaseStereoProcessor] = None

    if args.dense_depth:
        log.info("Dense depth mode enabled (--dense-depth) — this is the "
                  "heavy path; expect lower FPS on a Pi 5.")
        stereo = create_stereo_processor(pipeline_config, backend=args.backend)
    else:
        log.info("Using lightweight sparse block matcher (default). "
                  "Pass --dense-depth for the full dense SGBM+WLS pipeline.")
        sparse_matcher = SparseBlockMatcher(pipeline_config)

    if args.mode == "yolo":
        detector: BaseDetector = YoloDetector(
            model_path=args.model,
            conf=0.3,
            half_res=pipeline_config.yolo_half_res,
        )
    else:
        detector = AprilTagDetector()
    log.info("Detector mode: %s", args.mode)

    # I/O Buffer stabilization delay
    time.sleep(1.0)

    # Pre-allocate buffer arrays in contiguous memory to prevent heap fragmentation
    disp_vis_buf: Optional[np.ndarray] = None

    log.info("Loading stereo calibration maps from stereo_calib.npz")
    try:
        calib = np.load("stereo_calib.npz")
        map1x, map1y = calib["map1x"], calib["map1y"]
        map2x, map2y = calib["map2x"], calib["map2y"]
    except FileNotFoundError:
        log.error("stereo_calib.npz not found! Ensure it is in the same directory.")
        sys.exit(1)

    fps_counter = 0
    fps_display = 0.0
    fps_timer   = time.monotonic()

    log.info("Pipeline running. Press 'q' to quit.")

    # Main Synchronous Event Loop
    while not shutdown.is_set():
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r or frame_l is None or frame_r is None:
            log.info("Stream ended or camera disconnected.")
            break

        # Execute un-distortion and epipolar row alignment mapping
        frame_l = cv2.remap(frame_l, map1x, map1y, cv2.INTER_LINEAR)
        frame_r = cv2.remap(frame_r, map2x, map2y, cv2.INTER_LINEAR)

        # Stage 1: Localize Objects
        detections = detector.detect(frame_l)

        # Stage 2: Depth Estimation
        if args.dense_depth:
            # O(W*H) Complexity
            depth_map, disparity = stereo.compute_depth(frame_l, frame_r)
            for x, y, w, h, conf in detections:
                cx, cy = x + w // 2, y + h // 2
                dist = get_robust_distance(depth_map, cx, cy, pipeline_config.patch_size)
                draw_detection(frame_l, x, y, w, h, conf, dist)

                if dist > 0.0:
                    img_h, img_w = frame_l.shape[:2]
                    c_x = img_w / 2.0
                    c_y = img_h / 2.0
                    
                    bearing = math.atan2((cx - c_x), pipeline_config.focal_length)
                    
                    X = (cx - c_x) * dist / pipeline_config.focal_length
                    Y = (cy - c_y) * dist / pipeline_config.focal_length
                    true_range = math.sqrt(X**2 + Y**2 + dist**2)
                    
                    # If you have converted this to a ROS 2 node, construct 
                    # and publish your AprilTagDetection message right here.
                # ---------------------------------------------------------
        else:
            # O(N) Complexity
            if detections:
                # Cache grayscale conversions to prevent redundant matrix operations
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                for x, y, w, h, conf in detections:
                    cx, cy = x + w // 2, y + h // 2
                    
                    # Establish inner bounding margin (25%) to mitigate border artifacting
                    mx, my = w // 4, h // 4
                    
                    # 5-point spatial sampling grid
                    points = [
                        (cx, cy),                 
                        (cx, y + my),             
                        (cx, y + h - my),         
                        (x + mx, cy),             
                        (x + w - mx, cy)          
                    ]
                    
                    valid_distances = []
                    for px, py in points:
                        d = sparse_matcher.estimate_distance(gray_l, gray_r, px, py)
                        if d > 0.0:
                            valid_distances.append(d)
                            cv2.circle(frame_l, (px, py), 3, (255, 255, 0), -1)
                        else:
                            cv2.circle(frame_l, (px, py), 3, (0, 0, 255), -1)
                    
                    # Reject structural outliers via median filtering
                    if valid_distances:
                        final_dist = float(np.median(valid_distances))
                    else:
                        final_dist = 0.0
                        
                    draw_detection(frame_l, x, y, w, h, conf, final_dist)

                    # RANGE & BEARING CODE  
                    if final_dist > 0.0:
                        img_h, img_w = frame_l.shape[:2]
                        c_x = img_w / 2.0
                        c_y = img_h / 2.0
                        
                        bearing = math.atan2((cx - c_x), pipeline_config.focal_length)
                        
                        X = (cx - c_x) * final_dist / pipeline_config.focal_length
                        Y = (cy - c_y) * final_dist / pipeline_config.focal_length
                        true_range = math.sqrt(X**2 + Y**2 + final_dist**2)
                    '''
                        # Construct and Publish the ROS 2 Message
                        # Only publish if we have a valid distance, satisfying her safety requirement!
                        msg = AprilTagDetection()
                        
                        # Header
                        msg.header.stamp = node.get_clock().now().to_msg() # Adjust based on your ROS 2 node setup
                        msg.header.frame_id = "camera_link" # Adjust to match your TF tree
                        
                        # Data fields
                        msg.id = int(conf * 100) # Dummy ID for YOLO, or actual ID for AprilTag mode
                        msg.range = float(true_range)
                        msg.bearing = float(bearing)
                        msg.esp32_millis = 0 # As requested
                        
                        # Publish to /apriltag/detections
                        publisher.publish(msg)

                    else:
                        # Target lost or distance is 0.0. 
                        # Do absolutely nothing here. Do not publish.
                        # The node will automatically trigger the safety state after 2 seconds. 
                        pass
                    '''
        # Diagnostics Telemetry
        fps_counter += 1
        now = time.monotonic()
        if now - fps_timer >= 1.0:
            fps_display = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer   = now

        # Stage 3: Visual Output Sink
        if pipeline_config.display:
            draw_fps(frame_l, fps_display)
            cv2.imshow("Stereo Vision — Stream", frame_l)

            if args.dense_depth:
                disp_norm = cv2.normalize(
                    disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                )
                if disp_vis_buf is None or disp_vis_buf.shape != disp_norm.shape:
                    disp_vis_buf = np.empty_like(disp_norm, dtype=np.uint8)
                # Unsafe cast avoids redundant object initialization overhead
                np.copyto(disp_vis_buf, disp_norm, casting="unsafe")
                cv2.imshow(
                    "WLS Filtered Disparity",
                    cv2.applyColorMap(disp_vis_buf, cv2.COLORMAP_JET),
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Resource Teardown
    log.info("Releasing resources.")
    
    cap_left.stop()
    cap_right.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()