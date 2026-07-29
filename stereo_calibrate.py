#!/usr/bin/env python3
"""
stereo_calibrate.py  (ChArUco version)
========================================
Run a full stereo calibration from ChArUco pairs captured by
capture_calibration_pairs.py, and produce:

  1. stereo_calib.npz  — camera matrices, distortion coeffs, R, T, rectification
                          maps, etc. (load this at runtime for undistort+rectify)
  2. Printed --baseline / --focal-length values to plug into main_pi.py

Usage (defaults match calib_io_charuco_200x150_8x11_15_11_DICT_4X4.pdf):
    python3 stereo_calibrate.py --dir calib_pairs

What "good" looks like
-----------------------
- Per-camera reprojection error: well under 1.0 px (ideally < 0.5 px).
- Stereo reprojection error: well under 1.0 px.
Higher than that -> reshoot with more coverage (corners/edges of frame,
varied angles/distances).
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

from charuco_common import CharucoSetup


def collect_detections(setup, left_paths):
    """Detect the board in every pair. Returns per-image detections (used
    for per-camera intrinsics, which can use partial/unshared corners) plus
    per-pair shared-corner correspondences (needed for stereoCalibrate)."""
    all_corners_l, all_ids_l = [], []
    all_corners_r, all_ids_r = [], []
    pair_objp, pair_imgp_l, pair_imgp_r = [], [], []
    image_size = None
    used, skipped = 0, 0

    obj_pts_by_id = setup.object_points  # (N,3), index == charuco corner id

    for lpath in left_paths:
        rpath = lpath.replace("left_", "right_")
        if not os.path.exists(rpath):
            continue
        img_l, img_r = cv2.imread(lpath), cv2.imread(rpath)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray_l.shape[::-1]

        corners_l, ids_l, _ = setup.detect(gray_l)
        corners_r, ids_r, _ = setup.detect(gray_r)

        if ids_l is not None and len(ids_l) >= 6:
            all_corners_l.append(corners_l)
            all_ids_l.append(ids_l)
        if ids_r is not None and len(ids_r) >= 6:
            all_corners_r.append(corners_r)
            all_ids_r.append(ids_r)

        if ids_l is None or ids_r is None:
            skipped += 1
            print(f"  skip {os.path.basename(lpath)}: no board detected in "
                  f"{'left' if ids_l is None else ''}{' & ' if ids_l is None and ids_r is None else ''}"
                  f"{'right' if ids_r is None else ''}")
            continue

        shared = sorted(set(ids_l.flatten()) & set(ids_r.flatten()))
        if len(shared) < 6:
            skipped += 1
            print(f"  skip {os.path.basename(lpath)}: only {len(shared)} shared corners (need >=6)")
            continue

        map_l = {int(i): c for i, c in zip(ids_l.flatten(), corners_l.reshape(-1, 2))}
        map_r = {int(i): c for i, c in zip(ids_r.flatten(), corners_r.reshape(-1, 2))}
        objp = np.array([obj_pts_by_id[i] for i in shared], dtype=np.float32)
        imgp_l = np.array([map_l[i] for i in shared], dtype=np.float32).reshape(-1, 1, 2)
        imgp_r = np.array([map_r[i] for i in shared], dtype=np.float32).reshape(-1, 1, 2)

        pair_objp.append(objp)
        pair_imgp_l.append(imgp_l)
        pair_imgp_r.append(imgp_r)
        used += 1

    return (all_corners_l, all_ids_l, all_corners_r, all_ids_r,
            pair_objp, pair_imgp_l, pair_imgp_r, image_size, used, skipped)


def calibrate_mono(setup, all_corners, all_ids, image_size, label):
    objp_list, imgp_list = [], []
    for corners, ids in zip(all_corners, all_ids):
        objp, imgp = setup.board.matchImagePoints(corners, ids)
        objp_list.append(objp)
        imgp_list.append(imgp)
    rms, mtx, dist, _, _ = cv2.calibrateCamera(objp_list, imgp_list, image_size, None, None)
    print(f"[{label}] per-camera reprojection RMS error: {rms:.3f} px "
          f"{'OK' if rms < 1.0 else '(HIGH — recalibrate with more/better coverage)'}"
          f"  ({len(objp_list)} images used)")
    return mtx, dist


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="calib_pairs")
    ap.add_argument("--squares-x", type=int, default=8)
    ap.add_argument("--squares-y", type=int, default=11)
    ap.add_argument("--square-length", type=float, default=0.015)
    ap.add_argument("--marker-length", type=float, default=0.011)
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--no-legacy-pattern", action="store_true")
    ap.add_argument("--out", default="stereo_calib.npz")
    args = ap.parse_args()

    setup = CharucoSetup(
        args.squares_x, args.squares_y, args.square_length, args.marker_length,
        dict_name=args.dict, legacy_pattern=not args.no_legacy_pattern,
    )

    left_paths = sorted(glob.glob(os.path.join(args.dir, "left_*.png")))
    if not left_paths:
        print(f"No 'left_*.png' files found in {args.dir}", file=sys.stderr)
        sys.exit(1)

    (all_corners_l, all_ids_l, all_corners_r, all_ids_r,
     pair_objp, pair_imgp_l, pair_imgp_r, image_size, used, skipped) = \
        collect_detections(setup, left_paths)

    print(f"\nUsable stereo pairs: {used}, skipped: {skipped}")
    if used < 10:
        print("WARNING: fewer than 10 usable pairs — calibration will likely be poor.")
    if used < 4:
        print("Too few pairs to proceed (need at least a handful with shared corners).")
        print("If marker/corner counts were 0 during capture, double-check --dict and "
              "--no-legacy-pattern against your board.")
        sys.exit(1)

    # ── Per-camera intrinsics + distortion (uses ALL detections, even in
    #    images that didn't have enough overlap for the stereo pair list) ──
    mtx_l, dist_l = calibrate_mono(setup, all_corners_l, all_ids_l, image_size, "left")
    mtx_r, dist_r = calibrate_mono(setup, all_corners_r, all_ids_r, image_size, "right")

    # ── Stereo calibration (fix intrinsics just computed, solve R/T) ──────
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        pair_objp, pair_imgp_l, pair_imgp_r,
        mtx_l, dist_l, mtx_r, dist_r,
        image_size, criteria=criteria, flags=flags,
    )
    print(f"\nStereo reprojection RMS error: {rms_stereo:.3f} px "
          f"{'OK' if rms_stereo < 1.0 else '(HIGH — recalibrate)'}")

    # ── Rectification ─────────────────────────────────────────────────────
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1)

    # For the "quick plug-in" numbers (main_pi.py --focal-length/--baseline),
    # use the raw calibrated intrinsics/translation rather than the
    # post-stereoRectify P1/T[0]: main_pi.py's simple formula doesn't
    # actually rectify frames at runtime, and P1 can amplify small
    # extrinsic-calibration noise into a very wrong-looking focal length
    # even when the underlying calibration is otherwise fine. The raw
    # values are the physically-consistent ones for an un-rectified pair.
    focal_length_px = float((mtx_l[0, 0] + mtx_r[0, 0]) / 2)
    baseline_m = float(np.linalg.norm(T))

    np.savez(
        args.out,
        image_size=image_size,
        mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        focal_length_px=focal_length_px, baseline_m=baseline_m,
    )

    print("\n" + "=" * 60)
    print(f"Saved full calibration to: {args.out}")
    print("=" * 60)
    if rms_stereo >= 1.0:
        print("*** Stereo reprojection error is HIGH — the numbers below are not")
        print("*** trustworthy. Recapture with more pose variety (tilt the board")
        print("*** noticeably in different directions, not just move it around) and")
        print("*** more pairs before using these values.")
    print(f"Calibrated focal length : {focal_length_px:.1f} px  (@ {image_size[0]}x{image_size[1]})")
    print(f"Calibrated baseline     : {baseline_m*100:.2f} cm  (rig spec was 12 cm)")
    print("\nQuick plug-in (no rectification, partial improvement):")
    print(f"  python3 main_pi.py --picam2 --baseline {baseline_m:.4f} "
          f"--focal-length {focal_length_px:.1f} --resolution {image_size[0]}x{image_size[1]}")
    print("\nFor full accuracy, use stereo_calib.npz to undistort+rectify both frames")
    print("BEFORE AprilTag detection / block matching — ask me to wire this into")
    print("main_pi.py directly if you'd like.")

    if abs(baseline_m - 0.12) > 0.02:
        print(f"\nNOTE: calibrated baseline ({baseline_m*100:.1f} cm) differs from the "
              f"12 cm you measured by more than 2 cm — double check --square-length is "
              f"correct for your printed board, or physically re-measure the rig.")


if __name__ == "__main__":
    main()
