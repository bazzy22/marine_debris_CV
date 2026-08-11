#!/usr/bin/env python3
"""
capture_calibration_pairs_web.py  (headless / browser version)
================================================================
Same job as capture_calibration_pairs.py — capture synchronized ChArUco
board image pairs from the two Pi Camera Module 3s for stereo_calibrate.py
— but with the live preview served over HTTP instead of cv2.imshow, so it
works on a headless Ubuntu Server install with no X display at all.

Run on the Raspberry Pi 5 itself:

    python3 capture_calibration_pairs_web.py \\
        --squares-x 8 --squares-y 11 --square-length 0.015 --marker-length 0.011

Then, from ANY device on the same network (phone, laptop, whatever),
open a browser to:

    http://<pi-ip-address>:8000

You'll see the live left|right feed with the same green/red "READY"
overlay and marker/corner counts as before. Click "Capture Pair" (or
press SPACE/Enter while the page is focused) to save a pair once it's
ready. Ctrl+C in the terminal to stop.

Same tips as before: aim for 25-40 pairs, prioritize TILT VARIETY
(angle the board up/down/left/right, not just move it around flat-on)
— weak tilt variety is the most common cause of a calibration that
looks fine but gives wrong distances.
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from charuco_common import CharucoSetup, DICT_NAMES, detect_markers


def open_camera(camera_num, resolution):
    from picamera2 import Picamera2

    picam2 = Picamera2(camera_num=camera_num)
    cfg = picam2.create_video_configuration(main={"size": resolution, "format": "RGB888"})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.5)
    return picam2


def auto_detect_dict(gray):
    best = (None, 0)
    for name in DICT_NAMES:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        corners, ids, _ = detect_markers(gray, dictionary)
        n = 0 if ids is None else len(ids)
        print(f"  {name}: {n} markers detected")
        if n > best[1]:
            best = (name, n)
    return best


class CaptureState:
    """Shared state between the camera-processing thread and the Flask routes."""

    def __init__(self, args, setup, outdir):
        self.args = args
        self.setup = setup
        self.outdir = outdir
        self.lock = threading.Lock()
        self.jpeg_bytes = None
        self.ready = False
        self.status_text = "starting..."
        self.saved_count = len([f for f in os.listdir(outdir) if f.startswith("left_")])
        self.raw_frame_l = None
        self.raw_frame_r = None
        self.stop_flag = False
        self.capture_request = False


def camera_loop(state: CaptureState, cam_l, cam_r):
    args = state.args
    setup = state.setup

    while not state.stop_flag:
        frame_l = cam_l.capture_array()
        frame_r = cam_r.capture_array()
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        corners_l, ids_l, markers_l = setup.detect(gray_l)
        corners_r, ids_r, markers_r = setup.detect(gray_r)

        n_l = 0 if ids_l is None else len(ids_l)
        n_r = 0 if ids_r is None else len(ids_r)
        shared = 0
        if ids_l is not None and ids_r is not None:
            shared = len(set(ids_l.flatten()) & set(ids_r.flatten()))

        ready = shared >= args.min_shared_corners

        disp_l, disp_r = frame_l.copy(), frame_r.copy()
        setup.draw(disp_l, corners_l, ids_l)
        setup.draw(disp_r, corners_r, ids_r)

        combined = np.hstack([disp_l, disp_r])
        color = (0, 255, 0) if ready else (0, 0, 255)
        cv2.putText(combined, "READY - click Capture" if ready else "not enough shared corners",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(combined,
                    f"L: {markers_l} markers / {n_l} corners   "
                    f"R: {markers_r} markers / {n_r} corners   shared: {shared}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        with state.lock:
            cv2.putText(combined, f"saved: {state.saved_count}",
                        (10, combined.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                state.jpeg_bytes = buf.tobytes()
            state.ready = ready
            state.status_text = (
                f"L: {markers_l} markers / {n_l} corners   "
                f"R: {markers_r} markers / {n_r} corners   shared: {shared}"
            )
            state.raw_frame_l = frame_l
            state.raw_frame_r = frame_r

            if state.capture_request and ready:
                idx = state.saved_count
                lpath = os.path.join(state.outdir, f"left_{idx:03d}.png")
                rpath = os.path.join(state.outdir, f"right_{idx:03d}.png")
                cv2.imwrite(lpath, state.raw_frame_l)
                cv2.imwrite(rpath, state.raw_frame_r)
                state.saved_count += 1
                print(f"Saved pair {idx}: shared corners={shared}")
            state.capture_request = False

        time.sleep(0.03)  # ~30fps cap on the processing loop


def build_app(state: CaptureState):
    app = Flask(__name__)

    PAGE = """
    <!doctype html>
    <html>
    <head>
      <title>ChArUco Calibration Capture</title>
      <style>
        body { font-family: sans-serif; background: #111; color: #eee; text-align: center; }
        img { max-width: 95vw; border: 2px solid #444; margin-top: 10px; }
        button { font-size: 1.3em; padding: 12px 28px; margin: 16px; cursor: pointer; }
        #status { font-size: 1.1em; margin-top: 8px; }
      </style>
    </head>
    <body>
      <h2>ChArUco Calibration Capture</h2>
      <img src="/stream" />
      <div id="status">connecting...</div>
      <div>
        <button onclick="capture()">Capture Pair (Space)</button>
      </div>
      <script>
        function capture() {
          fetch('/capture', {method: 'POST'})
            .then(r => r.json())
            .then(d => { document.getElementById('status').innerText = d.message; });
        }
        document.addEventListener('keydown', function(e) {
          if (e.code === 'Space' || e.key === 'Enter') { e.preventDefault(); capture(); }
        });
        setInterval(function() {
          fetch('/status').then(r => r.json()).then(d => {
            document.getElementById('status').innerText =
              (d.ready ? '✅ READY — ' : '❌ ') + d.status_text + '   |   saved: ' + d.saved_count;
          });
        }, 500);
      </script>
    </body>
    </html>
    """

    @app.route("/")
    def index():
        return PAGE

    @app.route("/stream")
    def stream():
        def gen():
            while True:
                with state.lock:
                    frame = state.jpeg_bytes
                if frame is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.05)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        with state.lock:
            return jsonify(ready=state.ready, status_text=state.status_text,
                            saved_count=state.saved_count)

    @app.route("/capture", methods=["POST"])
    def capture():
        with state.lock:
            if not state.ready:
                return jsonify(message="Not ready — not enough shared corners yet.")
            state.capture_request = True
        time.sleep(0.1)  # let the camera loop process the request
        with state.lock:
            return jsonify(message=f"Saved pair {state.saved_count - 1}.")

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left-cam", type=int, default=0)
    ap.add_argument("--right-cam", type=int, default=1)
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--squares-x", type=int, default=8)
    ap.add_argument("--squares-y", type=int, default=11)
    ap.add_argument("--square-length", type=float, default=0.015,
                     help="metres (calib.io filename '..._15_11_...' -> 0.015)")
    ap.add_argument("--marker-length", type=float, default=0.011,
                     help="metres (calib.io filename '..._15_11_...' -> 0.011)")
    ap.add_argument("--dict", default="DICT_4X4_50", choices=DICT_NAMES)
    ap.add_argument("--no-legacy-pattern", action="store_true",
                     help="Disable legacy ChArUco corner numbering (try this if "
                          "markers are detected but corners aren't)")
    ap.add_argument("--force-legacy-interp", action="store_true",
                     help="Force the older detectMarkers()+interpolateCornersCharuco() "
                          "path instead of the newer CharucoDetector class, which has "
                          "several confirmed upstream bugs where markers are found but "
                          "corners silently come back empty. Try this if markers show "
                          ">0 but corners stay at 0 no matter what.")
    ap.add_argument("--min-shared-corners", type=int, default=8,
                     help="Minimum charuco corners shared between L/R to allow capture")
    ap.add_argument("--outdir", default="calib_pairs")
    ap.add_argument("--auto-dict", action="store_true",
                     help="Test all dictionary sizes on the first frame and exit")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0",
                     help="0.0.0.0 lets other devices on your network connect")
    args = ap.parse_args()

    w, h = (int(x) for x in args.resolution.lower().split("x"))
    resolution = (w, h)

    print(f"Opening left=camera_num {args.left_cam}, right=camera_num {args.right_cam} "
          f"at {resolution} ...")
    try:
        cam_l = open_camera(args.left_cam, resolution)
        cam_r = open_camera(args.right_cam, resolution)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.auto_dict:
        frame = cam_l.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Testing dictionaries against left camera frame:")
        name, n = auto_detect_dict(gray)
        print(f"\nBest match: {name} ({n} markers). Re-run with --dict {name}")
        cam_l.stop(); cam_r.stop()
        return

    setup = CharucoSetup(
        args.squares_x, args.squares_y, args.square_length, args.marker_length,
        dict_name=args.dict, legacy_pattern=not args.no_legacy_pattern,
        force_legacy_interp=args.force_legacy_interp,
    )

    os.makedirs(args.outdir, exist_ok=True)
    state = CaptureState(args, setup, args.outdir)

    print(f"Board: {args.squares_x}x{args.squares_y} squares, dict={args.dict}, "
          f"legacy_pattern={not args.no_legacy_pattern}")

    cam_thread = threading.Thread(target=camera_loop, args=(state, cam_l, cam_r), daemon=True)
    cam_thread.start()

    app = build_app(state)

    print("\n" + "=" * 60)
    print(f"  Open this URL in a browser on any device on your network:")
    print(f"  http://<this-pi's-ip-address>:{args.port}")
    print("=" * 60 + "\n")

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_flag = True
        cam_thread.join(timeout=2)
        cam_l.stop(); cam_r.stop()
        print(f"\nDone. Captured {state.saved_count} pairs in '{args.outdir}/'.")
        print("Then run: python3 stereo_calibrate.py --dir", args.outdir,
              f"--squares-x {args.squares_x} --squares-y {args.squares_y} "
              f"--square-length {args.square_length} --marker-length {args.marker_length} "
              f"--dict {args.dict}" + (" --no-legacy-pattern" if args.no_legacy_pattern else ""))


if __name__ == "__main__":
    main()
