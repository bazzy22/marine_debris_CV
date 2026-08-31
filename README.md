# Stereo Vision Pipeline — Documentation & User Guide

A stereo-camera pipeline that detects objects (marine debris via YOLO, or
AprilTags for testing/calibration), estimates their distance and bearing,
and — optionally — publishes that information to ROS2 for the BlueBoat
autonomous navigation stack.

---

## 1. How it works

The pipeline is built from four stages that run in sequence, once per frame:

1. **Camera input.** Two synchronized camera feeds (left/right) are read
   continuously in a background thread, so a slow frame never blocks the
   rest of the pipeline. On Raspberry Pi, `--picam2` uses `libcamera`
   directly for low-latency access to the Camera Module 3 pair; on any
   other machine, `--left`/`--right` can point at video files, USB webcam
   indices, or other sources instead. Auto-exposure and white-balance are
   locked on both cameras shortly after startup — stereo matching
   degrades badly if the two images don't agree on brightness/color.

2. **Detection.** The left frame is run through whichever detector
   `--mode` selects — YOLO for real-world debris detection, or ArUco for
   AprilTags (used for testing and camera calibration work). Each
   detection is a bounding box; anything smaller than the configured
   minimum area is discarded as noise.

3. **Distance & bearing.** For each detection, the pipeline estimates how
   far away it is using the two camera views (basic stereo triangulation
   — the same principle as human depth perception from two eyes) and
   computes the bearing angle relative to the camera's centerline.
   - **Sparse mode (default)**: only searches for matches near each
     detected object, which is fast and works well on a Raspberry Pi.
   - **Dense mode (`--dense-depth`)**: computes a full-frame depth map
     instead, which is much slower but useful for debugging camera
     alignment.

4. **Output.** Results are drawn on-screen (unless `--no-display` is
   set), logged to the terminal, and — if `--ros2` is set — published to
   ROS2 for the BlueBoat's navigation system to consume. See
   [Section 4](#4-ros2-integration) below.

---

## 2. Before running: what needs to already exist

| File | Required for | Notes |
|---|---|---|
| `config.py` | Always | Default baseline, focal length, minimum detection area, model path. Any of these can be overridden with CLI flags. |
| `stereo_calib.npz` | Always | Camera calibration data (rectification maps). **The script exits immediately if this is missing.** Produced by `stereo_calibrate.py` — see [Section 5](#5-calibration). |
| YOLO weights (`.pt` file) | `--mode yolo` only | Path set in `config.py`, or overridden with `--model`. Not needed for `--mode apriltag`. |
| `nautilus_apriltag_msgs` (built) | `--ros2` only | A ROS2 workspace with this message package built and sourced. See [Section 4](#4-ros2-integration). |

`calib_pairs/` (the folder of raw calibration photos) is **not** needed
at runtime — only `stereo_calibrate.py` reads it, to *produce*
`stereo_calib.npz`. Once that file exists, the photos can be archived or
deleted without affecting normal runs.

---

## 3. Command-line arguments

| Flag | Values | Description |
|---|---|---|
| `--mode` | `yolo`, `apriltag` | Detection engine to use. |
| `--left` | path, integer, or `picamN` | Left camera source. Ignored if `--picam2` is set. |
| `--right` | path, integer, or `picamN` | Right camera source. Ignored if `--picam2` is set. |
| `--picam2` | flag | Use Raspberry Pi Camera Module 3 sensors directly via `libcamera`. |
| `--resolution` | `WxH`, e.g. `640x480` | Capture resolution. Only applies with `--picam2`. |
| `--fps` | integer | Capture frame rate ceiling. Only applies with `--picam2`. |
| `--baseline` | float (metres) | Overrides the stereo baseline from `config.py`. Use the value `stereo_calibrate.py` prints out. |
| `--focal-length` | float (pixels) | Overrides the calibrated focal length from `config.py`. Also from `stereo_calibrate.py`'s output. |
| `--model` | file path | Overrides the YOLO `.pt` weights path. Only used with `--mode yolo`. |
| `--dense-depth` | flag | Use full-frame depth matching instead of the default fast per-detection matching. Debugging only — noticeably slower. |
| `--backend` | `auto`, `cuda`, `opencl`, `cpu` | Hardware backend for dense depth. Ignored unless `--dense-depth` is set. |
| `--no-display` | flag | Disable the GUI preview window. Required for headless/SSH deployment. |
| `--ros2` | flag | Publish detections to ROS2 as `nautilus_apriltag_msgs/msg/AprilTagDetection`. See [Section 4](#4-ros2-integration). |
| `--ros2-topic` | string | Topic to publish on. Default: `/apriltag/detections`. |
| `--camera-frame` | string | `frame_id` stamped on published message headers. Default: `camera_frame`. |

---

## 4. ROS2 integration

When `--ros2` is set, this script becomes the first node in the BlueBoat
vision-to-control pipeline, publishing directly to the topic the rest of
the team's nodes already expect:

```
main_final.py  →  /apriltag/detections  →  maneuver_manager_vs  →  /tracking_reference  →  controller_mavros_vs  →  MAVROS  →  boat
   (this script,                                      (both run on the boat's
    Raspberry Pi)                                       onboard computer)
```

It replaces the team's earlier `apriltag_bridge_vs.py` (a simulated
stand-in) and `nautilus_apriltag_bridge/udp_bridge_node.py` (an
ESP32-based bridge) — this script does the same job, publishing the same
message type on the same topic, using the actual stereo cameras instead.

**Behavior notes:**
- Only the **single closest valid detection** is published per frame —
  `maneuver_manager_vs` tracks one target at a time, so if multiple tags
  are visible, the nearest one is treated as the target.
- **Nothing is published when there's no valid detection.** This is
  intentional — the downstream node treats "no message for >2 seconds"
  as "target lost" and drives to a safe state on its own. Publishing a
  stale or zero-range reading instead would defeat that safety behavior.
- `bearing` follows REP-103 convention: `0` = tag centered, **positive =
  tag to the right** of the camera's optical axis, radians.

### Prerequisites (one-time setup)

1. ROS2 installed on the Pi (this project uses **Jazzy Jalisco**, matched
   to Ubuntu 24.04).
2. `nautilus_apriltag_msgs` — just the message-definition package, not
   the full team workspace — built in a local workspace:
   ```bash
   mkdir -p ~/ros2_ws/src
   # copy the nautilus_apriltag_msgs folder into ~/ros2_ws/src/
   cd ~/ros2_ws
   source /opt/ros/jazzy/setup.bash
   colcon build --packages-select nautilus_apriltag_msgs
   ```
3. `ROS_DOMAIN_ID` set to match the boat's onboard computer (both
   machines must use the same value to discover each other over the
   network). Set this once in `~/.bashrc`.

### Running it

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/marine_debris_CV
source .venv/bin/activate
python3 main_final.py --picam2 --baseline 0.1217 --focal-length 959.9 --mode apriltag --no-display --ros2
```

Each line matters and depends on the one before it: the first two load
ROS2 itself and the custom message package into the shell; the venv
activation brings in `picamera2`/OpenCV; the final command needs all of
the above already active.

### Verifying it's working

```bash
# in a second terminal, same machine
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 topic echo /apriltag/detections
```

With a tag in view, messages should stream with a real `id`, a plausible
`range`, and a `bearing` that flips sign correctly as the tag moves left
vs. right of the camera's centerline. Hide the tag — the topic should go
completely silent, not repeat a stale value.

The final check is cross-machine: on the **boat's** computer (same
`ROS_DOMAIN_ID`, on the same network), `ros2 topic list` should also show
`/apriltag/detections`.

---

## 5. Calibration

`stereo_calib.npz` is produced by a separate one-time (or
occasional) process, not part of this script:

1. `capture_calibration_pairs_web.py` — capture synchronized left/right
   photos of a ChArUco calibration board.
2. `stereo_calibrate.py` — processes those photos into `stereo_calib.npz`,
   and prints the measured baseline and focal length to use with
   `--baseline`/`--focal-length`.

Recalibrate whenever the camera rig is physically adjusted, or if
distance estimates start looking consistently off.

---

## 6. Deployment examples

**Local testing with recorded video (no camera hardware needed):**
```bash
python3 main_final.py --mode apriltag --left video/left.mp4 --right video/right.mp4
```

**Raspberry Pi 5, headless, YOLO debris detection:**
```bash
python3 main_final.py --mode yolo --picam2 --no-display
```

**Raspberry Pi 5, headless, AprilTag mode with ROS2 publishing (full deployment):**
```bash
python3 main_final.py --picam2 --baseline 0.1217 --focal-length 959.9 --mode apriltag --no-display --ros2
```

**Debugging camera alignment with a full dense depth map (any machine with a GPU):**
```bash
python3 main_final.py --mode yolo --left 0 --right 1 --dense-depth --backend cuda
```

**Setup run example:**
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/marine_debris_CV
source .venv/bin/activate
python3 main_final.py --picam2 --baseline 0.1217 --focal-length 959.9 --mode apriltag --no-display --ros2
```