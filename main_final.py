"""
Stereo Vision Pipeline — Edge-Optimized Deployment Architecture
===============================================================
Targets: Raspberry Pi 5 (AArch64, VideoCore VII) | NVIDIA Jetson (CUDA)

Architectural Overview:
-----------------------
This pipeline circumvents the O(W*H) computational bottleneck of dense stereo 
matching (StereoSGBM + WLS) by employing a "Detect-Then-Range" heuristic. 
By executing inference solely on the primary (left) optical frame and restricting 
epipolar search to bounded Regions of Interest (RoIs), spatial complexity is 
reduced from millions of pixel operations to a localized O(N) subset.

Key Engineering Paradigms:
  - Configuration Abstraction: State payload decoupled to external `config.py`.
  - Asynchronous I/O: Video capture operates in isolated daemons to prevent 
    blocking the main inference thread, utilizing write-preferring mutex locks.
  - Memory Management: Avoids heap fragmentation via pre-allocated contiguous 
    buffers and strictly bounded array slicing.
  - Sensor Synchronization: Hardware-level ISP locking guarantees photometric 
    consistency (AE/AWB) across the stereo pair, essential for NCC cost functions.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Abstracted configuration state payload injected at runtime
import config
from config import StereoConfig

# ---------------------------------------------------------------------------
# Telemetry & Logging Configuration
# ---------------------------------------------------------------------------
# Establishes a thread-safe stdout logger for asynchronous execution tracing 
# and system diagnostics.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stereo")

# Type alias defining the bounding box tensor schema: (x, y, width, height, confidence)
BBox = Tuple[int, int, int, int, float]


# ---------------------------------------------------------------------------
# ROS2 Bridge — publishes AprilTagDetection for the BlueBoat VS pipeline
# ---------------------------------------------------------------------------
# Interface contract (per blueboat_vs/maneuver_manager_vs.py):
#   topic:  /apriltag/detections
#   type:   nautilus_apriltag_msgs/msg/AprilTagDetection
#     std_msgs/Header header
#     int32  id              -> actual detected AprilTag ID
#     float32 range          -> distance to tag, metres, finite, > 0
#     float32 bearing        -> rad, 0 = centred, positive = tag to the RIGHT
#     uint32 esp32_millis    -> unused here, always 0
#
# Critical behavioural requirement from the downstream integrator: when no
# tag is currently detected, publish NOTHING — never republish a stale
# last-known measurement, and never publish a zero/invalid range. The
# receiving node (maneuver_manager_vs) already treats "no message for >2s"
# as "tag lost" and drives the vehicle to a safe state on its own.
class AprilTagRosBridge:
    """Thin wrapper around a minimal rclpy Node exposing a single publisher.

    Imports rclpy lazily so the rest of this script still runs (e.g. for
    offline testing / --mode yolo without ROS2) when --ros2 isn't passed
    and no ROS2 environment is sourced.
    """

    def __init__(self, topic: str, node_name: str = "stereo_apriltag_bridge") -> None:
        import rclpy
        from rclpy.node import Node
        from nautilus_apriltag_msgs.msg import AprilTagDetection

        self._rclpy = rclpy
        self._msg_cls = AprilTagDetection

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = Node(node_name)
        self.publisher = self.node.create_publisher(AprilTagDetection, topic, 10)
        self.node.get_logger().info(f"AprilTag ROS2 bridge publishing on {topic}")

    def publish_detection(
        self, range_m: float, bearing_rad: float, tag_id: int,
        frame_id: str = "camera_frame", esp32_millis: int = 0,
    ) -> None:
        msg = self._msg_cls()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.id = int(tag_id)
        msg.range = float(range_m)
        msg.bearing = float(bearing_rad)
        msg.esp32_millis = int(esp32_millis)
        self.publisher.publish(msg)

    def shutdown(self) -> None:
        try:
            self.node.destroy_node()
        finally:
            if self._rclpy.ok():
                self._rclpy.shutdown()


# ---------------------------------------------------------------------------
# I/O Acquisition Layer (V4L2)
# ---------------------------------------------------------------------------
class ThreadedVideoGrabber:
    """
    Asynchronous V4L2 video stream handler.
    Decouples I/O blocking operations from the main inference loop. 
    Implements a write-preferring lock to guarantee the inference engine 
    always receives the most recent frame in the hardware buffer, dropping 
    stale frames organically to minimize latency.
    """
    def __init__(self, source: int | str) -> None:
        self.stream = cv2.VideoCapture(source)
        if not self.stream.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

        # Thread-safe event for deterministic teardown sequence
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
        """Spawns the background daemon."""
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
            # O(1) operation prevents blocking the inference loop
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
        """Initiates graceful degradation of the I/O thread."""
        self._stop_event.set()
        self.stream.release()


# ---------------------------------------------------------------------------
# I/O Acquisition Layer (libcamera / DMA)
# ---------------------------------------------------------------------------
class PiCamera2Grabber:
    """
    libcamera (Picamera2) integration for native Raspberry Pi CSI pipelines.
    Provides direct memory access (DMA) to ISP frames, bypassing inefficient V4L2 bridging.
    Crucial for dual-CSI synchronization on Raspberry Pi 5.
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

        # Configure libcamera pipeline for RGB888 (maps natively to OpenCV's BGR layout in memory)
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
        """Overrides internal ISP dynamic algorithms to force static photometric properties."""
        self.picam2.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": int(exposure_time),
            "AnalogueGain": float(analogue_gain),
            "ColourGains": tuple(colour_gains),
        })

    def read_settled_metadata(self) -> dict:
        """Extracts converged state metadata from the ISP."""
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
    Forces photometric consistency across the stereo baseline.
    Correlation algorithms (NCC/SGBM) fail drastically if left/right sensors have 
    divergent exposure or white balance. This function acts as a master/slave 
    synchronizer to lock both ISP units to identical radiometric profiles.
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
# Sparse Distance Estimation Layer (Edge Optimized)
# ---------------------------------------------------------------------------
class SparseBlockMatcher:
    """
    Highly optimized, region-of-interest normalized cross-correlation (NCC) matcher.
    Calculates localized epipolar disparities. Reduces O(W*H) full-frame matrix 
    computations to O(N) operations, where N is the number of valid detections.
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
        Executes localized epipolar search. Extracts a template tensor from the 
        left frame and sweeps across the corresponding epipolar line segment in 
        the right frame. 
        """
        h, w = gray_l.shape[:2]
        half = self.patch_size // 2

        # Define bounded tensor slice for the reference patch (left frame)
        ty0, ty1 = cy - half, cy + half
        tx0, tx1 = cx - half, cx + half
        if ty0 < 0 or tx0 < 0 or ty1 > h or tx1 > w:
            return 0.0

        template = gray_l[ty0:ty1, tx0:tx1]
        if template.size == 0:
            return 0.0

        # Define bounded horizontal search space (right frame).
        # Search constrained strictly leftward to respect epipolar geometry.
        sy0 = max(0, ty0 - self.row_tolerance)
        sy1 = min(h, ty1 + self.row_tolerance)
        sx0 = max(0, cx - self.max_disparity)
        sx1 = min(w, cx - self.min_disparity + self.patch_size)

        if (sx1 - sx0) < self.patch_size or (sy1 - sy0) < self.patch_size:
            return 0.0

        # Execute SIMD-optimized normalized cross-correlation (NCC)
        strip = gray_r[sy0:sy1, sx0:sx1]
        result = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < self.min_confidence:
            return 0.0

        match_x, match_y = max_loc
        matched_x_center = sx0 + match_x + half
        raw_disparity = float(cx - matched_x_center)

        # 1D Parabolic sub-pixel interpolation around the response peak to mitigate integer quantization errors
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
    """Abstract interface enforcing uniform contract across heterogeneous compute backends."""
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
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
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

        # Normalize WLS integer output to standard sub-pixel disparity units
        disparity = filtered.astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0.1  # Prevent ZeroDivisionError in triangulation

        depth_map = (self.config.focal_length * self.config.baseline) / disparity
        return depth_map, disparity


class OpenCLStereoProcessor(BaseStereoProcessor):
    """
    iGPU acceleration using OpenCL Transparent API (T-API).
    Offloads memory-intensive tensor operations (color-space conversions) to the 
    VideoCore VII GPU (Pi 5), freeing CPU cycles for the SGBM matcher.
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

        filtered = self._cpu.wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)
        disparity = filtered.astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0.1

        depth_map = (self._cpu.config.focal_length * self._cpu.config.baseline) / disparity
        return depth_map, disparity


class CUDAStereoProcessor(BaseStereoProcessor):
    """NVIDIA Jetson optimized pipeline shifting core SGBM execution to CUDA cores."""
    def __init__(self, config: StereoConfig) -> None:
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            raise RuntimeError("No CUDA device found — use a different backend.")
        
        self.config = config
        bs = config.block_size
        log.info("CUDA device: %s", cv2.cuda.DeviceInfo(cv2.cuda.getDevice()).name())

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
        
        # WLS post-processing executes on CPU host (no CUDA impl. in ximgproc)
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

        # Execute highly parallelized SIMT operations
        gray_gpu_l = cv2.cuda.cvtColor(gpu_l, cv2.COLOR_BGR2GRAY)
        gray_gpu_r = cv2.cuda.cvtColor(gpu_r, cv2.COLOR_BGR2GRAY)

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
    """Hardware prober implementing the Factory pattern for dense matching backend selection."""
    order = ["cuda", "opencl", "cpu"] if backend == "auto" else [backend]

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
            log.warning("Backend %r not available: %s — probing next layer.", name, exc)

    raise RuntimeError("No stereo processor could be initialized.")


# ---------------------------------------------------------------------------
# Object Detection Layer
# ---------------------------------------------------------------------------
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[BBox]:
        pass


class YoloDetector(BaseDetector):
    """
    Handles universal neural network inference via Ultralytics YOLOv8 architecture.
    Auto-detects tensor hardware acceleration (CUDA/MPS) with a CPU fallback mechanism.
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
        log.info("YOLO inference engine mapped to device: %s", self.device)

    @staticmethod
    def _pick_device() -> str:
        """Probes for optimal tensor processing unit."""
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
        Non-Maximum Suppression (NMS) is inherently handled by the Ultralytics backend.
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
                # Upscale bounding box vectors to align with native frame resolution
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
        # Populated by detect(); parallel to the returned bbox list so
        # callers can recover which physical AprilTag ID each box came
        # from (BaseDetector's bbox schema has no room for it).
        self.last_ids: List[int] = []

    def detect(self, frame: np.ndarray) -> List[BBox]:
        if frame is None:
            self.last_ids = []
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        detected_boxes: List[BBox] = []
        self.last_ids = []
        if ids is not None:
            for c, tag_id in zip(corners, ids.flatten()):
                pts = c[0]
                x1, x2 = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
                y1, y2 = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
                detected_boxes.append((
                    int(x1), int(y1),
                    int(x2 - x1), int(y2 - y1),
                    1.0,
                ))
                self.last_ids.append(int(tag_id))
        return detected_boxes


# ---------------------------------------------------------------------------
# Core Utilities & Execution Logic
# ---------------------------------------------------------------------------
def get_robust_distance(
    depth_map: np.ndarray, cx: int, cy: int, patch_size: int
) -> float:
    """
    Applies spatial median filtering across an N-dimensional patch within the dense map.
    Suppresses transient noise spikes (glare, reflections) and handles dead-zones gracefully.
    """
    h, w = depth_map.shape
    half = patch_size >> 1  # Bitwise shift for minor optimization

    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)

    patch = depth_map[y0:y1, x0:x1]
    valid = patch[patch > 0.5]
    return float(np.median(valid)) if valid.size > 0 else 0.0


def draw_detection(
    frame: np.ndarray, x: int, y: int, w: int, h: int, conf: float, 
    z_dist: float, true_range: float, bearing_rad: float
) -> None:
    """Applies overlay telemetry (including kinematics) to the output buffer."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    # Convert bearing to degrees for human readability on the GUI
    bearing_deg = math.degrees(bearing_rad)

    # Format telemetry strings
    label1 = f"Z-Dist: {z_dist:.2f}m | Rng: {true_range:.2f}m"
    label2 = f"Brg: {bearing_deg:.1f} deg | Conf: {conf * 100:.0f}%"
    
    # Dynamically shift labels above or below the box to prevent clipping
    label_y = y - 25 if y > 35 else y + h + 20
    
    # Draw two lines of text
    cv2.putText(frame, label1, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
    cv2.putText(frame, label2, (x, label_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )


def parse_args() -> argparse.Namespace:
    """Constructs the CLI parser for dynamic runtime configuration."""
    p = argparse.ArgumentParser(
        description="Stereo vision pipeline — edge-optimised deployment architecture"
    )
    p.add_argument(
        "--mode", choices=["yolo", "apriltag"], default=config.DEFAULT_MODE,
        help=f"Detection topology (default: {config.DEFAULT_MODE})",
    )
    p.add_argument(
        "--left", default=config.DEFAULT_LEFT_SRC,
        metavar="SRC",
        help="Left primary source stream mapping",
    )
    p.add_argument(
        "--right", default=config.DEFAULT_RIGHT_SRC,
        metavar="SRC",
        help="Right secondary source stream mapping",
    )
    p.add_argument(
        "--picam2", action="store_true",
        help="Enable native libcamera (Picamera2) integration for CSI interfaces",
    )
    p.add_argument(
        "--baseline", type=float, default=None,
        help="Override intrinsic stereo baseline offset (metres)",
    )
    p.add_argument(
        "--focal-length", type=float, default=None,
        help="Override intrinsic calibrated focal length vector (pixels)",
    )
    p.add_argument(
        "--resolution", default="640x480", metavar="WxH",
        help="Hardware capture resolution mapping",
    )
    p.add_argument(
        "--fps", type=int, default=30,
        help="Hardware capture FPS ceiling",
    )
    p.add_argument(
        "--dense-depth", action="store_true",
        help="Execute exhaustive O(W*H) full-frame matching (debug/demo only)",
    )
    p.add_argument(
        "--backend", choices=["auto", "cuda", "opencl", "cpu"], default="auto",
        help="Dense algorithm execution backend",
    )
    p.add_argument(
        "--model", default=config.DEFAULT_MODEL,
        metavar="PATH",
        help=f"Path to serialized YOLO tensor weights (default: {config.DEFAULT_MODEL})",
    )
    p.add_argument(
        "--no-display", dest="display", action="store_false",
        help="Disable GUI subsystem (critical for headless edge deployment)",
    )
    p.add_argument(
        "--ros2", action="store_true",
        help="Publish detections as nautilus_apriltag_msgs/AprilTagDetection "
             "for the BlueBoat VS pipeline (requires a sourced ROS2 environment)",
    )
    p.add_argument(
        "--ros2-topic", default="/apriltag/detections",
        help="Topic to publish AprilTagDetection messages on",
    )
    p.add_argument(
        "--camera-frame", default="camera_frame",
        help="frame_id to stamp on published detection headers",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main Orchestration Engine
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Hydrate configuration payload from external config module
    pipeline_config = StereoConfig(display=args.display)
    if args.baseline is not None:
        pipeline_config.baseline = args.baseline
    if args.focal_length is not None:
        pipeline_config.focal_length = args.focal_length

    try:
        w_str, h_str = args.resolution.lower().split("x")
        pipeline_config.pi_resolution = (int(w_str), int(h_str))
    except ValueError:
        log.warning("Unrecognized --resolution %r mapping to default %s",
                    args.resolution, pipeline_config.pi_resolution)
    pipeline_config.pi_fps = args.fps

    # Configure graceful teardown hooks mapped to standard UNIX POSIX signals
    shutdown = threading.Event()

    def _handle_signal(signum, _frame):
        log.info("POSIX Signal %d trapped — initiating graceful teardown sequence.", signum)
        shutdown.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Initialize asynchronous hardware acquisition topology
    if args.picam2:
        def _cam_index(s: str, fallback: int) -> int:
            s = s.strip().lower()
            if s.startswith("picam"):
                s = s[len("picam"):]
            return int(s) if s.isdigit() else fallback

        left_idx  = _cam_index(args.left, 0)
        right_idx = _cam_index(args.right, 1)

        log.info(
            "Initializing Pi Camera Module 3 DMA pair: left_idx=%d, right_idx=%d "
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

        log.info("Initializing left primary stream: %r", left_src)
        log.info("Initializing right secondary stream: %r", right_src)
        cap_left  = ThreadedVideoGrabber(left_src).start()
        cap_right = ThreadedVideoGrabber(right_src).start()

    # Instantiate strategy interfaces for runtime algorithm topology
    sparse_matcher: Optional[SparseBlockMatcher] = None
    stereo: Optional[BaseStereoProcessor] = None

    if args.dense_depth:
        log.info("Dense matrix mode activated (--dense-depth) — expect throughput degradation.")
        stereo = create_stereo_processor(pipeline_config, backend=args.backend)
    else:
        log.info("Sparse topology initialized (default).")
        sparse_matcher = SparseBlockMatcher(pipeline_config)

    if args.mode == "yolo":
        detector: BaseDetector = YoloDetector(
            model_path=args.model,
            conf=0.3,
            half_res=pipeline_config.yolo_half_res,
        )
    else:
        detector = AprilTagDetector()
    log.info("Inference topology: %s", args.mode)

    ros_bridge: Optional[AprilTagRosBridge] = None
    if args.ros2:
        try:
            ros_bridge = AprilTagRosBridge(topic=args.ros2_topic)
        except Exception:
            log.exception(
                "Failed to initialize ROS2 bridge — is the ROS2 environment "
                "sourced (e.g. 'source /opt/ros/<distro>/setup.bash') and is "
                "nautilus_apriltag_msgs built/on your PYTHONPATH? Continuing "
                "WITHOUT publishing to ROS2."
            )
            ros_bridge = None

    # I/O Buffer stabilization margin
    time.sleep(1.0)

    # Pre-allocate contiguous memory buffers to strictly prevent heap fragmentation
    disp_vis_buf: Optional[np.ndarray] = None

    log.info("Mapping intrinsic calibration matrices from stereo_calib.npz")
    try:
        calib = np.load("stereo_calib.npz")
        map1x, map1y = calib["map1x"], calib["map1y"]
        map2x, map2y = calib["map2x"], calib["map2y"]
    except FileNotFoundError:
        log.error("stereo_calib.npz mapping failed! Halting execution.")
        sys.exit(1)

    fps_counter = 0
    fps_display = 0.0
    fps_timer   = time.monotonic()

    # Rate-limited detection logging: at 30fps we never want to log every
    # frame (floods the terminal now, and would flood a ROS2 topic/log
    # later) — log a lightweight heartbeat every second regardless of
    # detections, and full detection details throttled separately so a
    # human (or log aggregator) can actually read it.
    DETECTION_LOG_INTERVAL_S = 2.0
    last_detect_log = time.monotonic()

    log.info("Orchestration engine online. Awaiting data...")

    # Main Synchronous Event Loop
    while not shutdown.is_set():
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r or frame_l is None or frame_r is None:
            log.info("Stream exhausted or interface fault detected.")
            break

        # Execute un-distortion and epipolar row alignment mapping
        frame_l = cv2.remap(frame_l, map1x, map1y, cv2.INTER_LINEAR)
        frame_r = cv2.remap(frame_r, map2x, map2y, cv2.INTER_LINEAR)

        # Stage 1: Spatial Inference
        raw_detections = detector.detect(frame_l)
        raw_ids = getattr(detector, "last_ids", None)  # only meaningful for AprilTagDetector

        # Spatial filtering: Suppress false-positives utilizing bounding tensor constraints
        detections = []
        for i, (x, y, w, h, conf) in enumerate(raw_detections):
            if (w * h) >= pipeline_config.min_object_area:
                tag_id = raw_ids[i] if raw_ids is not None and i < len(raw_ids) else -1
                detections.append((x, y, w, h, conf, tag_id))

# Stage 2: Depth Estimation
        frame_results = []  # collected for rate-limited logging below
        if args.dense_depth:
            # O(W*H) Matrix Complexity Pathway
            depth_map, disparity = stereo.compute_depth(frame_l, frame_r)
            for x, y, w, h, conf, tag_id in detections:
                cx, cy = x + w // 2, y + h // 2
                dist = get_robust_distance(depth_map, cx, cy, pipeline_config.patch_size)
                
                # Pose Translation (Range & Bearing)
                true_range = 0.0
                bearing = 0.0
                if dist > 0.0:
                    img_h, img_w = frame_l.shape[:2]
                    c_x = img_w / 2.0
                    c_y = img_h / 2.0
                    
                    bearing = math.atan2((cx - c_x), pipeline_config.focal_length)
                    X = (cx - c_x) * dist / pipeline_config.focal_length
                    Y = (cy - c_y) * dist / pipeline_config.focal_length
                    true_range = math.sqrt(X**2 + Y**2 + dist**2)

                # Draw overlay with newly calculated kinematics
                draw_detection(frame_l, x, y, w, h, conf, dist, true_range, bearing)
                frame_results.append(dict(x=cx, y=cy, dist=dist, true_range=true_range,
                                           bearing=bearing, conf=conf, tag_id=tag_id))

        else:
            # O(N) Sub-tensor Complexity Pathway
            if detections:
                # Cache grayscale conversions to prevent redundant matrix operations
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                for x, y, w, h, conf, tag_id in detections:
                    cx, cy = x + w // 2, y + h // 2
                    
                    # Establish inner bounding margin (25%) to mitigate border artifacting
                    mx, my = w // 4, h // 4
                    
                    # 5-point spatial sampling cross-grid
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
                        
                    # Pose Translation (Range & Bearing)
                    true_range = 0.0
                    bearing = 0.0
                    if final_dist > 0.0:
                        img_h, img_w = frame_l.shape[:2]
                        c_x = img_w / 2.0
                        c_y = img_h / 2.0
                        
                        bearing = math.atan2((cx - c_x), pipeline_config.focal_length)
                        X = (cx - c_x) * final_dist / pipeline_config.focal_length
                        Y = (cy - c_y) * final_dist / pipeline_config.focal_length
                        true_range = math.sqrt(X**2 + Y**2 + final_dist**2)

                    # Draw overlay with newly calculated kinematics
                    draw_detection(frame_l, x, y, w, h, conf, final_dist, true_range, bearing)
                    frame_results.append(dict(x=cx, y=cy, dist=final_dist, true_range=true_range,
                                               bearing=bearing, conf=conf, tag_id=tag_id))

        # Diagnostics & Throughput Telemetry
        fps_counter += 1
        now = time.monotonic()
        if now - fps_timer >= 1.0:
            fps_display = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer   = now
            log.info("stereo: heartbeat fps=%.1f objects_in_frame=%d",
                      fps_display, len(frame_results))

        # Rate-limited detection detail logging — full distance/bearing per
        # object, throttled so it stays readable even at 30fps input.
        if frame_results and (now - last_detect_log) >= DETECTION_LOG_INTERVAL_S:
            for i, r in enumerate(frame_results):
                log.info(
                    "stereo: object[%d] tag_id=%d px=(%d,%d) dist=%.2fm range=%.2fm "
                    "bearing=%.1fdeg conf=%.2f",
                    i, r["tag_id"], r["x"], r["y"], r["dist"], r["true_range"],
                    math.degrees(r["bearing"]), r["conf"],
                )
            last_detect_log = now

        # ROS2 publish — BlueBoat VS interface (/apriltag/detections).
        # Pick the single closest VALID detection as "the" target for this
        # frame; downstream (maneuver_manager_vs) tracks one target at a
        # time and has no concept of multiple simultaneous tags. Publish
        # nothing at all when there's no valid detection this frame — per
        # spec, never republish a stale reading or a zero/invalid range.
        if ros_bridge is not None:
            valid_results = [
                r for r in frame_results
                if r["true_range"] > 0.0 and math.isfinite(r["true_range"])
                and math.isfinite(r["bearing"])
            ]
            if valid_results:
                target = min(valid_results, key=lambda r: r["true_range"])
                try:
                    ros_bridge.publish_detection(
                        range_m=target["true_range"],
                        bearing_rad=target["bearing"],
                        tag_id=target["tag_id"],
                        frame_id=args.camera_frame,
                        esp32_millis=0,
                    )
                except Exception:
                    log.exception("Failed to publish AprilTagDetection — dropping this frame's ROS2 message.")
            # else: no valid detection this frame -> intentionally publish nothing.

        # Stage 3: GUI Rendering Pipeline
        if pipeline_config.display:
            draw_fps(frame_l, fps_display)
            cv2.imshow("Stereo Vision — Stream", frame_l)

            if args.dense_depth:
                disp_norm = cv2.normalize(
                    disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                )
                if disp_vis_buf is None or disp_vis_buf.shape != disp_norm.shape:
                    disp_vis_buf = np.empty_like(disp_norm, dtype=np.uint8)
                # Unsafe cast bypasses redundant object initialization overhead
                np.copyto(disp_vis_buf, disp_norm, casting="unsafe")
                cv2.imshow(
                    "WLS Filtered Disparity",
                    cv2.applyColorMap(disp_vis_buf, cv2.COLORMAP_JET),
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Resource Teardown & Buffer De-allocation
    log.info("Initiating resource de-allocation.")
    
    cap_left.stop()
    cap_right.stop()
    cv2.destroyAllWindows()
    if ros_bridge is not None:
        ros_bridge.shutdown()


if __name__ == "__main__":
    main()