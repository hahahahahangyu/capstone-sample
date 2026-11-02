"""Synthetic checks of the planar measurement maths (no model needed)."""

import math

import cv2
import numpy as np
import pytest

from pothole.marker import detect_marker
from pothole.measure import (marker_homography, measure_mask, measure_polygon,
                             measure_rectangle, to_cm)

PX_PER_CM = 20  # synthetic top-down scene: marker 10 cm = 200 px


def top_down_scene():
    """White road plane with marker ID 0 and a filled 'pothole' ellipse."""
    canvas = np.full((700, 1000, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 10 * PX_PER_CM)
    canvas[60:260, 60:260] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    # Semi-axes 12 cm x 6 cm, rotated 30 degrees -> 24 x 12 cm, area 72*pi.
    cv2.ellipse(mask, (620, 400), (12 * PX_PER_CM, 6 * PX_PER_CM), 30, 0, 360,
                255, -1)
    canvas[mask > 0] = (60, 60, 60)
    return canvas, mask


def warp(image, mask):
    """Simulate a camera looking at the road from an oblique angle."""
    h, w = mask.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[150, 120], [w - 120, 0], [w, h], [0, h - 40]])
    H = cv2.getPerspectiveTransform(src, dst)
    return (cv2.warpPerspective(image, H, (w, h), borderValue=(255, 255, 255)),
            cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST))


@pytest.mark.parametrize("oblique", [False, True])
def test_ellipse_dimensions(oblique):
    image, mask = top_down_scene()
    if oblique:
        image, mask = warp(image, mask)
    marker_px = detect_marker(image)
    assert marker_px is not None
    length, width, area = measure_mask(marker_px, mask)
    assert length == pytest.approx(24.0, rel=0.03)
    assert width == pytest.approx(12.0, rel=0.03)
    assert area == pytest.approx(72 * math.pi, rel=0.03)


def test_polygon_matches_rectangle_for_four_corners():
    """A real 29.7 x 21 cm sheet seen in perspective: both methods agree."""
    marker_px = np.float32([[100, 100], [300, 110], [295, 300], [105, 290]])
    H = marker_homography(marker_px)
    rect_cm = np.float32([[15, 5], [44.7, 5], [44.7, 26], [15, 26]])
    rect_px = to_cm(np.linalg.inv(H), rect_cm)
    for measure in (measure_rectangle, measure_polygon):
        length, width, area = measure(marker_px, rect_px)
        assert length == pytest.approx(29.7, rel=1e-3)
        assert width == pytest.approx(21.0, rel=1e-3)
        assert area == pytest.approx(29.7 * 21.0, rel=1e-3)


def test_empty_mask_returns_none():
    marker_px = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert measure_mask(marker_px, np.zeros((50, 50), np.uint8)) is None
