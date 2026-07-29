"""
charuco_common.py
==================
Shared ChArUco board + detector setup for capture_calibration_pairs.py and
stereo_calibrate.py. Handles both the new (OpenCV >= 4.7) and legacy
cv2.aruco APIs, since the Pi's system python3-opencv package can lag well
behind pip's opencv-contrib-python.

Board parameters default to what's encoded in the filename of a calib.io
ChArUco PDF, e.g.:

    calib_io_charuco_200x150_8x11_15_11_DICT_4X4.pdf
                       ^board mm  ^squares  ^square/marker mm  ^dictionary

    -> squares_x=8, squares_y=11, square_length=15mm, marker_length=11mm,
       dictionary DICT_4X4

If your filename differs, adjust --squares-x/--squares-y/--square-length/
--marker-length/--dict accordingly — get these wrong (especially the
square/marker size ratio) and detection will silently fail or be unstable.
"""

import sys

import cv2
import numpy as np

DICT_NAMES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
]


def require_aruco():
    if not hasattr(cv2, "aruco"):
        print(
            "ERROR: cv2.aruco is not available. Your OpenCV build doesn't "
            "include the contrib/aruco module. Install it with:\n"
            "  python3 -m pip install --break-system-packages opencv-contrib-python",
            file=sys.stderr,
        )
        sys.exit(1)


def detect_markers(gray, dictionary, params=None):
    """Version-robust cv2.aruco marker detection. Newer OpenCV (>=4.7-ish,
    depending on build) removed the old free function cv2.aruco.detectMarkers
    in favor of the cv2.aruco.ArucoDetector class; older builds don't have
    that class yet. Try both."""
    if params is None:
        params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids, rejected


class CharucoSetup:
    """Wraps board creation + per-frame detection across OpenCV API versions."""

    def __init__(self, squares_x, squares_y, square_length, marker_length,
                 dict_name="DICT_4X4_50", legacy_pattern=True):
        require_aruco()
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length

        dict_id = getattr(cv2.aruco, dict_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        self._new_api = hasattr(cv2.aruco, "CharucoBoard") and hasattr(cv2.aruco, "CharucoDetector")
        if self._new_api:
            self.board = cv2.aruco.CharucoBoard(
                (squares_x, squares_y), square_length, marker_length, self.dictionary
            )
            # calib.io's generator predates OpenCV's 4.6 charuco numbering
            # change; "legacy" pattern matches what it produces. If corners
            # aren't found (or IDs look scrambled), try legacy_pattern=False.
            try:
                self.board.setLegacyPattern(legacy_pattern)
            except AttributeError:
                pass
            det_params = cv2.aruco.DetectorParameters()
            charuco_params = cv2.aruco.CharucoParameters()
            self.detector = cv2.aruco.CharucoDetector(self.board, charuco_params, det_params)
        else:
            self.board = cv2.aruco.CharucoBoard_create(
                squares_x, squares_y, square_length, marker_length, self.dictionary
            )
            self.detector = None

        # 3D position (board coordinate frame) of every possible charuco
        # corner, indexed by charuco corner id.
        try:
            self.object_points = self.board.getChessboardCorners()
        except AttributeError:
            self.object_points = self.board.chessboardCorners

    def detect(self, gray):
        """Returns (charuco_corners, charuco_ids, marker_count) or
        (None, None, marker_count) if too few corners found."""
        if self._new_api:
            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                self.detector.detectBoard(gray)
            marker_count = 0 if marker_ids is None else len(marker_ids)
        else:
            marker_corners, marker_ids, _ = detect_markers(gray, self.dictionary)
            marker_count = 0 if marker_ids is None else len(marker_ids)
            charuco_corners, charuco_ids = None, None
            if marker_ids is not None and len(marker_ids) > 0:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.board
                )

        if charuco_ids is None or len(charuco_ids) == 0:
            return None, None, marker_count
        return charuco_corners, charuco_ids, marker_count

    def draw(self, img, charuco_corners, charuco_ids):
        if charuco_ids is None or charuco_corners is None:
            return
        if len(charuco_ids) == 0 or len(charuco_corners) != len(charuco_ids):
            return
        try:
            cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids, (0, 255, 0))
        except cv2.error:
            # Purely cosmetic — never let an overlay-drawing glitch kill capture.
            pass
