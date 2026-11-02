"""Milestone 3: automatic pothole measurement with YOLO-seg + ArUco marker.

Inputs: pothole_images/*.jpg|jpeg|png, each with ArUco marker ID 0 (10 cm)
        lying on the road surface next to the pothole.
Outputs: pothole_results/<stem>_mask.png, <stem>_measured.jpg, measurements.csv

Each pothole mask is projected onto the marker's cm plane, so the marker and
pothole rim must be on the same plane. Length/width are the sides of the
minimum-area rotated rectangle; results are 2D only (no depth or volume).

Optional ground truth CSV (--ground-truth) with columns
    image,length_cm,width_cm
adds error % columns for the largest pothole in each image. Ruler values are
never used to compute the predictions.
"""

from pathlib import Path
import argparse
import csv

import cv2
import numpy as np

from pothole.marker import ARUCO_ID, detect_marker, draw_marker, make_detector
from pothole.measure import mask_contours, measure_polygon
from pothole.segment import DEFAULT_WEIGHTS, load_model, predict_masks

BASE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpeg", ".jpg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=BASE / "pothole_images")
    parser.add_argument("--out", type=Path, default=BASE / "pothole_results")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="mps, cpu, ...")
    parser.add_argument("--ground-truth", type=Path, default=None)
    return parser.parse_args()


def load_ground_truth(path):
    if path is None:
        return {}
    with path.open(newline="") as file:
        return {
            row["image"]: sorted((float(row["length_cm"]),
                                  float(row["width_cm"])), reverse=True)
            for row in csv.DictReader(file)
        }


def pct_error(value, actual):
    return abs(value - actual) / actual * 100


def main():
    args = parse_args()
    image_paths = sorted(p for p in args.images.glob("*")
                         if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"Put photographs into: {args.images}")
    args.out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights)
    detector = make_detector()
    truth = load_ground_truth(args.ground_truth)

    csv_path = args.out / "measurements.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image", "pothole_idx", "confidence", "length_cm", "width_cm",
            "area_cm2", "length_error_pct", "width_error_pct"
        ])

        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"{path.name}: image could not be read. Skipping.")
                continue
            marker_px = detect_marker(image, ARUCO_ID, detector)
            if marker_px is None:
                print(f"{path.name}: ArUco ID {ARUCO_ID} not detected. Skipping.")
                continue

            detections = predict_masks(model, image, args.conf, args.device)
            if not detections:
                print(f"{path.name}: no pothole detected.")
                continue

            # Measure each pothole; largest area first so idx 0 is the main one.
            results = []
            for mask, score in detections:
                contours = mask_contours(mask)
                if not contours:
                    continue
                contour = contours[0].reshape(-1, 2)
                dims = measure_polygon(marker_px, contour)
                results.append((dims, score, contour, mask))
            results.sort(key=lambda r: r[0][2], reverse=True)

            combined = np.zeros(image.shape[:2], dtype=np.uint8)
            annotated = image.copy()
            draw_marker(annotated, marker_px)
            thickness = max(2, image.shape[1] // 500)
            font_scale = max(0.6, image.shape[1] / 1500)
            print(f"\n{path.name}: {len(results)} pothole(s)")
            for idx, ((length, width, area), score, contour, mask) in enumerate(results):
                combined |= mask
                cv2.polylines(annotated, [contour.astype(np.int32)], True,
                              (0, 255, 0), thickness)
                box = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32)))
                cv2.polylines(annotated, [box.astype(np.int32)], True,
                              (0, 0, 255), thickness)
                x, y = contour.min(axis=0)
                cv2.putText(annotated, f"#{idx} L={length:.1f} W={width:.1f}cm",
                            (int(x), max(int(y) - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                            thickness, cv2.LINE_AA)

                length_err = width_err = ""
                if idx == 0 and path.name in truth:
                    true_long, true_short = truth[path.name]
                    length_err = f"{pct_error(length, true_long):.2f}"
                    width_err = f"{pct_error(width, true_short):.2f}"
                writer.writerow([
                    path.name, idx, f"{score:.3f}", f"{length:.3f}",
                    f"{width:.3f}", f"{area:.3f}", length_err, width_err
                ])
                print(f"  #{idx} conf {score:.2f}: {length:.2f} x {width:.2f} cm, "
                      f"area {area:.2f} cm^2")

            cv2.imwrite(str(args.out / f"{path.stem}_mask.png"), combined)
            cv2.imwrite(str(args.out / f"{path.stem}_measured.jpg"), annotated)

    print(f"\nFinished. Review results at: {args.out}")


if __name__ == "__main__":
    main()
