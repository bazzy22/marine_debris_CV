# Stereo Vision Pipeline: Documentation & User Guide

---

## PART 1: System Architecture Documentation

### 1. Executive Summary
This document outlines the architecture and technical specifications of the stereo vision pipeline designed for edge deployments (specifically targeting AArch64 architectures like the Raspberry Pi 5 and NVIDIA Jetson). 

The core engineering philosophy of this pipeline is to circumvent the severe O(W*H) computational bottleneck inherent in dense stereo matching (StereoSGBM) by implementing a **"Detect-Then-Range"** heuristic[cite: 1]. By restricting epipolar searches to bounded Regions of Interest (RoIs) identified by a spatial detection layer, the system reduces matrix operations from millions of pixels to a localized O(N) subset[cite: 1].

### 2. Architectural Design & Data Flow
The system is decoupled into four primary layers, ensuring thread safety, asynchronous execution, and hardware-accelerated routing.

#### 2.1. Configuration & State Management
To prevent monolithic coupling, all hardware intrinsic parameters, algorithm hyperparameters, and file paths are injected at runtime via an external payload (`config.py`)[cite: 1, 2]. 
*   **Key Parameters**: Intrinsic stereo baseline, calibrated focal length, bounding tensor constraints (`min_object_area`), and sub-pixel matcher tolerances[cite: 2].

#### 2.2. I/O Acquisition Layer
Video acquisition is strictly decoupled from the inference loop to prevent I/O blocking[cite: 1].
*   **V4L2 Interface (`ThreadedVideoGrabber`)**: Runs a background daemon polling the hardware buffer[cite: 1]. It utilizes a write-preferring mutex lock to ensure the main inference thread always receives the most recent frame, organically dropping stale frames to minimize latency[cite: 1].
*   **DMA / libcamera Interface (`PiCamera2Grabber`)**: Integrates native libcamera (Picamera2) pipelines for CSI sensors[cite: 1]. This bypasses inefficient V4L2 bridging and provides Direct Memory Access (DMA) to the ISP frames[cite: 1].
*   **Photometric Synchronization**: The `synchronize_stereo_cameras` function acts as a master/slave synchronizer[cite: 1]. Because correlation algorithms (NCC/SGBM) degrade drastically with divergent radiometry, this module forces a hard lock on auto-exposure (AE) and auto-white-balance (AWB) across both ISP units after a brief convergence period[cite: 1].

#### 2.3. Spatial Detection Layer
The primary frame (left) is routed through a spatial inference engine to localize Regions of Interest (RoIs).
*   **YOLOv8 Engine (`YoloDetector`)**: Implements dynamic hardware probing (CUDA/MPS/CPU)[cite: 1]. It employs a tensor downsampling mechanism (`half_res`) to reduce Multiply-Accumulate (MAC) operations by ~4x, significantly decreasing inference latency on embedded CPUs[cite: 1, 2].
*   **Fiducial Engine (`AprilTagDetector`)**: A lightweight ArUco-based wrapper for deterministic marker identification[cite: 1].
*   **Spatial Filtering**: Extracted bounding tensors are filtered against `min_object_area` to suppress environmental false positives before they consume epipolar matching cycles[cite: 1, 2].

#### 2.4. Depth Estimation Layer
The pipeline dynamically routes depth computations based on runtime configuration (`--dense-depth`)[cite: 1].

*   **Sparse Sub-Tensor Pathway (Default)**: Evaluated strictly within detection RoIs[cite: 1].
    *   Extracts a spatial template from the reference frame and sweeps across the corresponding epipolar line segment in the target frame[cite: 1].
    *   Utilizes SIMD-optimized Normalized Cross-Correlation (NCC)[cite: 1].
    *   Applies a 1D parabolic sub-pixel interpolation around the response peak to mitigate integer quantization errors at longer ranges[cite: 1].
    *   Utilizes a 5-point spatial sampling cross-grid to establish structural consensus, rejecting anomalies via median filtering[cite: 1].
*   **Dense Matrix Pathway (Opt-In)**: Utilizes a Strategy Pattern (`BaseStereoProcessor`) to abstract underlying hardware[cite: 1].
    *   **CPU (`CPUStereoProcessor`)**: Standard AVX/SSE vectorization[cite: 1].
    *   **OpenCL (`OpenCLStereoProcessor`)**: Uses Transparent API (T-API) `cv2.UMat` wrapping to offload memory-intensive color space conversions to the iGPU (e.g., VideoCore VII), freeing host CPU cycles[cite: 1].
    *   **CUDA (`CUDAStereoProcessor`)**: Offloads core SGBM execution and H2D/D2H memory transfers to NVIDIA CUDA cores[cite: 1].

#### 2.5. Memory & Thread Management Guidelines
*   **Heap Fragmentation**: The GUI pipeline utilizes pre-allocated, contiguous `numpy` arrays (`disp_vis_buf`) and executes `unsafe` casting during color mapping to prevent garbage collection spikes and redundant object initialization overhead[cite: 1].
*   **Teardown Determinism**: Execution utilizes `threading.Event` coupled with standard UNIX POSIX signal trapping (SIGINT/SIGTERM) to guarantee graceful de-allocation of hardware streams and GUI subsystems without zombie processes[cite: 1].

---

## PART 2: Quick-Start & User Guide

### 1. Pre-Flight Checklist
Before executing the script, ensure the working directory contains the following critical files:
*   **`config.py`**: The external configuration file containing your baseline, focal length, and minimum area constraints[cite: 1, 2].
*   **`stereo_calib.npz`**: The NumPy archive containing your intrinsic calibration matrices (`map1x`, `map1y`, `map2x`, `map2y`)[cite: 1]. The pipeline will fatally crash if this is missing[cite: 1].
*   **YOLO Weights**: If using YOLO mode, ensure your `.pt` model file (e.g., `best.pt`) is in the correct path specified by your configuration[cite: 1, 2].

### 2. Command-Line Arguments (Tags)
The pipeline is designed to be dynamically configured at runtime via the terminal[cite: 1]. Here is the complete list of available tags:

| Argument Flag | Accepted Values | Description |
| :--- | :--- | :--- |
| `--mode` | `yolo`, `apriltag` | Sets the primary detection engine[cite: 1]. |
| `--left` | File path, Int, `picamN` | Maps the left primary video source (e.g., `0`, `video/left.mp4`, or `picam0`)[cite: 1]. |
| `--right` | File path, Int, `picamN` | Maps the right secondary video source[cite: 1]. |
| `--picam2` | *Flag (No value)* | Enables direct memory access (DMA) via libcamera/Picamera2[cite: 1]. Required when using Raspberry Pi Camera Module 3 sensors directly via CSI ports[cite: 1]. |
| `--resolution` | String (e.g., `640x480`) | Hardware capture resolution mapping, strictly for `--picam2` mode[cite: 1]. |
| `--fps` | Integer (e.g., `30`) | Hardware capture frames-per-second ceiling, strictly for `--picam2` mode[cite: 1]. |
| `--baseline` | Float (e.g., `0.12`) | Overrides the physical stereo baseline distance (in meters) defined in `config.py`[cite: 1]. |
| `--focal-length` | Float (e.g., `500.0`) | Overrides the calibrated focal length (in pixels) defined in `config.py`[cite: 1]. |
| `--model` | File path | Overrides the path to the YOLO `.pt` weights[cite: 1]. |
| `--dense-depth` | *Flag (No value)* | Switches from the default `O(N)` sparse matcher to the heavy `O(W*H)` full-frame StereoSGBM+WLS matrix computation[cite: 1]. Useful for debugging but causes major FPS drops on edge devices[cite: 1]. |
| `--backend` | `auto`, `cuda`, `opencl`, `cpu` | Selects the hardware acceleration backend for dense depth mapping[cite: 1]. Ignored unless `--dense-depth` is active[cite: 1]. |
| `--no-display` | *Flag (No value)* | Disables the GUI (`cv2.imshow`)[cite: 1]. Critical for deploying headless on an edge device via SSH[cite: 1]. |

### 3. Practical Deployment Examples

**Example A: Desktop Testing (AprilTags via Video Files)**
Running the pipeline locally using pre-recorded videos to verify the logic.
> `python main_pi_gemini_5anchor.py --mode apriltag --left video/left.mp4 --right video/right.mp4`

**Example B: Raspberry Pi 5 Deployment (Headless YOLO)**
Deploying the pipeline on the physical robot using two Pi Camera Module 3 sensors via CSI ports. The GUI is disabled to save CPU cycles.
> `python main_pi_gemini_5anchor.py --mode yolo --picam2 --left picam0 --right picam1 --no-display`

**Example C: Jetson Nano / Orin (Dense Depth Debugging)**
Forcing the pipeline to calculate a full-frame depth map using CUDA acceleration to verify physical stereo alignment.
> `python main_pi_gemini_5anchor.py --mode yolo --left 0 --right 1 --dense-depth --backend cuda`