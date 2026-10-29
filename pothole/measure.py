"""Planar measurement: map image pixels onto the ArUco marker's cm plane.

The marker and the measured object must lie on the same physical plane for
the homography to be valid. Outputs are 2D only (no depth or volume).
"""

import cv2
import numpy as np

from pothole.marker import ARUCO_SIZE_CM


def marker_homography(marker_px, marker_size_cm=ARUCO_SIZE_CM):
    """Homography from image pixels to cm, with the marker as a square."""
    size = marker_size_cm
    cm_marker = np.array([
        [0, 0], [size, 0], [size, size], [0, size]
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(marker_px.astype(np.float32), cm_marker)


def to_cm(H, points_px):
    points = np.asarray(points_px, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(points, H)[0]


def measure_rectangle(marker_px, rectangle_px, marker_size_cm=ARUCO_SIZE_CM):
    """Milestone 2: four ordered corners -> (long side, short side, area).

    Opposite sides are averaged, which suits a manually clicked rectangle.
    """
    cm_points = to_cm(marker_homography(marker_px, marker_size_cm), rectangle_px)
    # Adjacent sides: [0-1, 1-2, 2-3, 3-0]
    edges = np.linalg.norm(np.roll(cm_points, -1, axis=0) - cm_points, axis=1)
    pair_1 = float((edges[0] + edges[2]) / 2.0)
    pair_2 = float((edges[1] + edges[3]) / 2.0)
    longer_cm, shorter_cm = sorted((pair_1, pair_2), reverse=True)
    area_cm2 = float(cv2.contourArea(cm_points))
    return longer_cm, shorter_cm, area_cm2


def measure_polygon(marker_px, polygon_px, marker_size_cm=ARUCO_SIZE_CM):
    """Any outline (e.g. a pothole contour) -> (length, width, area).

    Length/width are the sides of the minimum-area rotated rectangle in the cm
    plane, so they follow the pothole's own orientation, not the image axes.
    """
    cm_points = to_cm(marker_homography(marker_px, marker_size_cm), polygon_px)
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(cm_points)
    length_cm, width_cm = sorted((float(side_a), float(side_b)), reverse=True)
    area_cm2 = float(cv2.contourArea(cm_points))
    return length_cm, width_cm, area_cm2


def mask_contours(mask, min_area_px=50):
    """External contours of a binary mask, largest first, tiny blobs dropped."""
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    return sorted(contours, key=cv2.contourArea, reverse=True)


def measure_mask(marker_px, mask, marker_size_cm=ARUCO_SIZE_CM):
    """Measure the largest region of a binary mask; None if the mask is empty."""
    contours = mask_contours(mask)
    if not contours:
        return None
    return measure_polygon(marker_px, contours[0].reshape(-1, 2), marker_size_cm)
