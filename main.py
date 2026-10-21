"""First FYP milestone: ArUco calibration + synthetic pothole dimensions.

This image is artificially generated with no perspective distortion. The output
is only a unit-style test, NOT a validated real-world pothole measurement.
"""
from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
MARKER_SIZE_CM = 10.0   # Nominal marker side in the synthetic example


def build_test_image():
    canvas = np.full((500, 850, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 200)
    canvas[60:260, 60:260] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR) #ArUco marker 
    # Blue ellipse = MOCK pothole, not a trained segmentation result.
    cv2.ellipse(canvas, (550, 290), (120, 190), 0, 0, 360, (255, 0, 0), -1)
    cv2.imwrite(str(OUTPUT_DIR / 'aruco_marker.png'), marker)
    return canvas


def measure(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or 7 not in ids.flatten():
        raise RuntimeError('ArUco marker ID 7 not found')

    index = int(np.where(ids.flatten() == 7)[0][0])
    pts = corners[index].reshape(4, 2)
    edges_px = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    pixels_per_cm = float(np.mean(edges_px) / MARKER_SIZE_CM)

    # Threshold the BLUE color to stand in for a DeepLab segmentation mask.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 100, 60), (130, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError('Mock pothole mask not found')
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    area_pixels = int(cv2.countNonZero(mask))

    length_cm = w / pixels_per_cm
    width_cm = h / pixels_per_cm
    area_cm2 = area_pixels / (pixels_per_cm ** 2)
    annotated = image.copy()
    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 3)
    cv2.rectangle(annotated, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 2)
    cv2.putText(annotated, f'L={length_cm:.1f}cm W={width_cm:.1f}cm',
                (360, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.imwrite(str(OUTPUT_DIR / 'result.png'), annotated)
    cv2.imwrite(str(OUTPUT_DIR / 'mock_mask.png'), mask)
    return pixels_per_cm, length_cm, width_cm, area_cm2


if __name__ == '__main__':
    image = build_test_image()
    scale, length, width, area = measure(image)
    print(f'Scale: {scale:.2f} pixels/cm')
    print(f'Mock pothole length: {length:.2f} cm')
    print(f'Mock pothole width:  {width:.2f} cm')
    print(f'Mock pothole area:   {area:.2f} cm^2')
    print(f'Images saved to: {OUTPUT_DIR}')
