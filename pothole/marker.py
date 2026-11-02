"""ArUco marker detection for the 10 cm scale reference."""

import cv2
import numpy as np

ARUCO_SIZE_CM = 10.0  # outer square; confirmed physical measurement
ARUCO_ID = 0
ARUCO_DICT = cv2.aruco.DICT_4X4_50


def make_detector(aruco_dict=ARUCO_DICT):
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    params = cv2.aruco.DetectorParameters()
    # Corner errors are magnified across the whole homography; refining them
    # cut oblique-view length error from ~3.5% to <1% in tests/test_measure.py.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_marker(image, marker_id=ARUCO_ID, detector=None):
    """Return the marker's 4 corners as a (4, 2) float32 array, or None."""
    detector = detector or make_detector()
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or marker_id not in ids.flatten().tolist():
        return None
    index = ids.flatten().tolist().index(marker_id)
    return corners[index].reshape(4, 2).astype(np.float32)


def draw_marker(image, marker_px, marker_id=ARUCO_ID):
    cv2.aruco.drawDetectedMarkers(image, [marker_px.reshape(1, 4, 2)],
                                  np.array([[marker_id]], dtype=np.int32))
