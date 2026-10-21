"""Milestone 2: planar size measurement using a 10 cm ArUco marker.

Inputs: A4_images/*.jpeg (also .jpg, .png)
Interaction: click 4 rectangle corners in clockwise OR counterclockwise order.
Outputs: A4_results/<stem>_mask.png, <stem>_measured.jpg, measurements.csv

This is a manual reference-object boundary, not YOLO or DeepLab segmentation.
The ArUco marker's measured black-border square and the reference rectangle
must lie on the same physical plane for a valid planar homography.
"""

from pathlib import Path
import csv

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
IMAGE_DIR = BASE / "A4_images"
OUTPUT_DIR = BASE / "A4_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARUCO_SIZE_CM = 10.0  # outer square; confirmed physical measurement
ARUCO_ID = 0
ARUCO_DICT = cv2.aruco.DICT_4X4_50


def select_corners(image, marker_corners, filename):
    """Select 4 corners around the rectangle; return original-image pixels."""
    height, width = image.shape[:2]
    scale = min(1.0, 1150.0 / max(height, width))
    preview_base = cv2.resize(image, None, fx=scale, fy=scale)
    points = []
    window = "Select rectangle: " + filename

    def mouse_callback(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"  Corner {len(points)} selected: ({x}, {y}) in preview")

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse_callback)
    print("  Click the four OUTER rectangle corners around its boundary.")
    print("  Controls: S = save selection, R = restart clicks, Esc = skip photo")

    while True:
        view = preview_base.copy()
        # Marker outline is visualized separately from the rectangle annotation.
        mark = np.rint(marker_corners * scale).astype(np.int32)
        cv2.polylines(view, [mark], True, (0, 255, 0), 2)
        if points:
            path = np.array(points, dtype=np.int32)
            cv2.polylines(view, [path], len(points) == 4, (255, 0, 0), 2)
            for p in points:
                cv2.circle(view, p, 5, (0, 0, 255), -1)
        cv2.putText(view, f"Points {len(points)}/4 | S save | R redo | Esc skip",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(view, f"Points {len(points)}/4 | S save | R redo | Esc skip",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(window, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == ord("s") and len(points) == 4:
            cv2.destroyWindow(window)
            return np.array(points, dtype=np.float32) / scale
        elif key == 27:
            cv2.destroyWindow(window)
            return None


def measure_on_plane(marker_px, rectangle_px):
    """Project a four-corner rectangle onto the ArUco marker's cm plane."""
    size = ARUCO_SIZE_CM
    cm_marker = np.array([
        [0, 0], [size, 0], [size, size], [0, size]
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(marker_px.astype(np.float32), cm_marker)
    cm_points = cv2.perspectiveTransform(
        rectangle_px.reshape(1, 4, 2).astype(np.float32), H
    )[0]

    # Adjacent sides: [0-1, 1-2, 2-3, 3-0]
    edges = np.linalg.norm(np.roll(cm_points, -1, axis=0) - cm_points, axis=1)
    pair_1 = float((edges[0] + edges[2]) / 2.0)
    pair_2 = float((edges[1] + edges[3]) / 2.0)
    longer_cm, shorter_cm = sorted((pair_1, pair_2), reverse=True)
    area_cm2 = float(cv2.contourArea(cm_points))
    return longer_cm, shorter_cm, area_cm2


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Create your image folder first: {IMAGE_DIR}")

    image_paths = sorted(p for p in IMAGE_DIR.iterdir()
                         if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    if not image_paths:
        raise FileNotFoundError(f"Put your new photographs into: {IMAGE_DIR}")

    # Ruler measurements are independent ground truth, not used to calculate
    # the predicted dimensions.
    actual_long_cm = float(input("Ruler-measured LONG side in cm: "))
    actual_short_cm = float(input("Ruler-measured SHORT side in cm: "))
    if actual_long_cm <= 0 or actual_short_cm <= 0:
        raise ValueError("Ruler measurements must be positive numbers")
    actual_long_cm, actual_short_cm = sorted(
        (actual_long_cm, actual_short_cm), reverse=True
    )
    actual_area_cm2 = actual_long_cm * actual_short_cm

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(
        dictionary, cv2.aruco.DetectorParameters()
    )

    csv_path = OUTPUT_DIR / "measurements.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image", "length_cm", "width_cm", "area_cm2",
            "length_error_pct", "width_error_pct", "area_error_pct"
        ])

        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"{path.name}: image could not be read. Skipping.")
                continue

            corners, ids, _ = detector.detectMarkers(image)
            if ids is None or ARUCO_ID not in ids.flatten().tolist():
                print(f"{path.name}: ArUco ID {ARUCO_ID} not detected. Skipping.")
                continue
            index = ids.flatten().tolist().index(ARUCO_ID)
            marker_px = corners[index].reshape(4, 2).astype(np.float32)
            print(f"\n{path.name}: ArUco marker detected.")

            rectangle_px = select_corners(image, marker_px, path.name)
            if rectangle_px is None:
                print(f"{path.name}: skipped by user.")
                continue

            length_cm, width_cm, area_cm2 = measure_on_plane(
                marker_px, rectangle_px
            )
            length_err = abs(length_cm - actual_long_cm) / actual_long_cm * 100
            width_err = abs(width_cm - actual_short_cm) / actual_short_cm * 100
            area_err = abs(area_cm2 - actual_area_cm2) / actual_area_cm2 * 100

            # Save the manual mask and annotated full-resolution image.
            height, width = image.shape[:2]
            polygon = np.rint(rectangle_px).astype(np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            annotated = image.copy()
            cv2.polylines(annotated, [polygon], True, (0, 255, 0), 8)
            cv2.aruco.drawDetectedMarkers(annotated, [corners[index]],
                                         np.array([[ARUCO_ID]], dtype=np.int32))
            cv2.imwrite(str(OUTPUT_DIR / f"{path.stem}_mask.png"), mask)
            cv2.imwrite(str(OUTPUT_DIR / f"{path.stem}_measured.jpg"), annotated)

            writer.writerow([
                path.name, f"{length_cm:.3f}", f"{width_cm:.3f}",
                f"{area_cm2:.3f}", f"{length_err:.2f}",
                f"{width_err:.2f}", f"{area_err:.2f}"
            ])
            print(f"  Long side: {length_cm:.2f} cm (error {length_err:.2f}%)")
            print(f"  Short side: {width_cm:.2f} cm (error {width_err:.2f}%)")
            print(f"  Area: {area_cm2:.2f} cm^2 (error {area_err:.2f}%)")

    print(f"\nFinished. Review results at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
