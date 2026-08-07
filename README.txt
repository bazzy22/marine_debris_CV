1.0 Overview
This is a lightweight, edge-optimized stereo vision pipeline designed for real-time object detection and spatial distance estimation. It is built primarily for embedded systems like the Raspberry Pi 5 and leverages dual camera streams to detect objects (via YOLO or AprilTags) and calculate their physical distance using stereo disparity. It is highly adaptable for various applications, such as automated marine debris tracking, robotics navigation, or general spatial awareness tasks. By default, it utilizes a CPU-friendly Sparse Block Matching algorithm optimized for bounding-box regions rather than computing expensive, full-frame dense depth maps.

2.0 Features
Universal Object Detection: Supports interchangeable YOLO models (e.g., best.pt for marine entities, trash, etc.) and OpenCV ArUco AprilTag detection.

Edge-Optimized Distance Estimation: Evaluates a 5-point cross-pattern grid on detected bounding boxes, dropping outliers and using median values to ensure robust distance locks without the overhead of full-frame processing.

Native Pi Camera Support: Fully integrated with picamera2 (libcamera) for simultaneous dual Camera Module 3 capture on a single Raspberry Pi 5.

Hardware-Accelerated Dense Depth: Includes fallback support for OpenCL (Pi 5 GPU) and CUDA (NVIDIA Jetson) dense StereoSGBM + WLS pipelines when full-frame depth visualization is required.

Auto-Exposure Synchronization: Automatically settles and locks AE/AWB across the stereo pair to prevent photometric inconsistencies during template matching.

3.0 Prerequisites
Compute Hardware: Raspberry Pi 5 (recommended) or NVIDIA Jetson.

Sensor Hardware: 2x Raspberry Pi Camera Module 3 (connected to CSI ports 0 and 1).

Rig Hardware: A physically fixed stereo rig (default baseline is set to 12 cm).

Software Base: Python 3.9+.

Python Dependencies: opencv-python, opencv-contrib-python, ultralytics, and picamera2.

Required Files: stereo_calib.npz (A generated stereo camera calibration matrix containing map1x, map1y, map2x, and map2y).

4.0 Installation and Setup
Clone the repository and navigate to the directory.

Install the required standard Python packages via pip: opencv-contrib-python and ultralytics.

Install picamera2 via apt (Raspberry Pi OS specific): python3-picamera2.

Ensure the stereo_calib.npz file is placed in the root directory.

5.0 Configuration
The pipeline's default parameters and model paths are centralized in config.py. Update the following core variables in this file as needed:

DEFAULT_MODEL: Set to the path of your chosen weights (e.g., trained_model/best.pt).

DEFAULT_MODE: Set the detection framework (e.g., yolo).

baseline: Set the rig inter-camera distance in meters (default is 0.12).

focal_length: You must update this variable with the focal length in pixels obtained during your stereo calibration process to get metrically accurate distances.

pi_resolution: Set the capture resolution (default is 640, 480).

6.0 Usage Commands
Standard Execution (Headless Edge Deployment): Run the main script with the picam2 flag, yolo mode, and no-display flag to optimize performance.

Video File Testing: Pass the file paths to the left and right arguments to test the pipeline on pre-recorded video files instead of live camera feeds.

7.0 Command-Line Arguments Reference
--mode: Detection framework (yolo or apriltag). Default is yolo.

--model: Path to the .pt weights for YOLO inference. Default is config.py value.

--picam2: Flag to use libcamera/picamera2 for Pi Camera Module 3. Default is False.

--left / --right: Input sources (file path, integer index, or picamN). Defaults are config.py values.

--dense-depth: Opt-in flag to run full-frame StereoSGBM + WLS matching. Default is False.

--backend: Dense matching hardware backend (auto, cuda, opencl, cpu). Default is auto.

--no-display: Disables GUI windows for headless operation. Default is False.

--resolution: Pi camera capture resolution. Default is 640x480.

8.0 Pipeline Architecture Overview
Frame Acquisition: Dedicated threads continuously pull the latest frames from both cameras, dropping stale frames to prevent buffer lag.

Rectification: Both frames are undistorted and row-aligned using the stereo_calib.npz mapping.

Detection: A half-resolution copy of the left frame is passed to the YOLO/AprilTag detector for rapid inference.

Distance Estimation (Sparse Mode): For each bounding box, a 5-point template matching grid is applied between the left and right frames. A 1D parabolic fit provides sub-pixel disparity accuracy.

Distance Estimation (Dense Mode): Computes a complete depth map across the entire frame matrix before sampling the bounding box region.

Output: Distances are drawn onto the live feed (if display is enabled) or logged via standard output.