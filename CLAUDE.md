# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Final-year project (FYP) prototype for measuring potholes from photos. An ArUco marker of known size lies on the road next to the pothole and sets the scale. A YOLO-seg model outlines each pothole, and its 2D length, width and area are computed in cm. It's built in milestones: Milestone 1 (synthetic) and Milestone 2 (manual clicks on A4 paper) validated the measurement, and Milestone 3 adds the trained model.

## Environment and commands

Python 3.9 virtualenv in `.venv/` (opencv-contrib-python 4.11, numpy 1.26, ultralytics/torch). These pins are deliberate (see `requirements.txt`): ultralytics on macOS rejects the numpy 2.x versions Python 3.9 can install. Training and inference run on the Apple GPU via `device="mps"`.

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest                                    # all tests
pytest tests/test_measure.py -k oblique   # a single test
```

**OpenCV gotcha:** ultralytics installs `opencv-python`, which shadows `cv2.aruco` from `opencv-contrib-python`. After installing, uninstall `opencv-python` and force-reinstall the contrib package (the commands are in the README).

Pipeline (each script is run from the repo root):

| Script | Purpose | Reads → Writes |
|---|---|---|
| `scripts/prepare_potholemix.py` | SHREC 2022 Pothole Mix masks → YOLO-seg polygons, keeping the official train/val/test split | `~/Documents/fyp datasets/SHREC2022/Pothole-Mix/pothole-mix/` → `data/processed/potholemix/`, `data/manifests/`, `data/pothole_seg.yaml` |
| `scripts/prepare_rdd2022.py` | RDD2022 D40 (pothole) boxes → YOLO detect labels, read straight from the 13 GB zip | `~/Documents/fyp datasets/RDD2022_fullDataset/*.zip` → `data/processed/rdd2022_d40/`, `data/manifests/rdd2022_d40.csv` |
| `scripts/train.py` | Fine-tunes `yolo11n-seg.pt` | → `runs/<name>/`, best checkpoint copied to `weights/pothole_seg.pt` |
| `scripts/evaluate.py` | Box and mask mAP on the test split | `weights/`, yaml → `runs/eval_test/` |
| `measure_pothole.py` | Milestone 3 end to end, headless; optional `--ground-truth` CSV (`image,length_cm,width_cm`) | `pothole_images/` → `pothole_results/` |
| `measure_A4.py` | Milestone 2: manual 4-corner click (GUI + stdin, so the user runs it) | `A4_images/` → `A4_results/` |
| `main.py` | Milestone 1: synthetic image, marker ID **7** (leave as the historical record) | → `outputs/` |
| `test_aruco.py`, `make_rectangle_mask.py`, `check_images.py` | Early ad-hoc scripts (not pytest) | `test_images/` |

`data/raw/`, `data/processed/`, `runs/`, `weights/`, `pothole_results/` and `*.pt` are gitignored.

## Architecture: `pothole/` package

The shared logic lives in `pothole/`, so the milestone scripts stay thin:
- `marker.py`: `DICT_4X4_50`, marker ID 0, 10 cm outer square. `detect_marker()` returns (4, 2) corners or None.
- `measure.py`: `marker_homography()` maps the marker's corners to a 10×10 cm square, and every measurement projects pixels through it. `measure_rectangle()` (Milestone 2) averages opposite sides. `measure_polygon()` / `measure_mask()` (Milestone 3) take any contour and use `minAreaRect` in cm for length/width and `contourArea` for area.
- `segment.py`: lazy-imports ultralytics, and `predict_masks()` returns full-resolution `(mask, confidence)` per pothole.

## Measurement principles

- The marker and the pothole rim must lie on the same physical plane, or the homography is invalid.
- Ruler values are independent ground truth used only for error percentages. They never feed into the predictions.
- Outputs are 2D only: no depth, volume or severity.
## Datasets

The raw downloads live outside the repo in `~/Documents/fyp datasets/`.
- **SHREC 2022 = Pothole Mix** (CC BY-NC 3.0). Masks are colour coded: in this release **potholes are red (RGB 255,0,0) and cracks green**, contrary to the paper's caption. Of 3,340 training images only 709 contain potholes. The crack-only sources (crack500, edmcrack600, gaps384) are kept as background negatives. Converted file names are prefixed `<source>__` because sources reuse names like `001.jpg`.
- **RDD2022** has bounding boxes only (Pascal VOC, classes D00/D10/D20/D40), so it's used to test detection on unseen countries, not for mask training. Only the train splits have annotations.
- Image filenames encode the camera angle (`_45`, `_60`, `_90`), so error can be compared across perspective.
