# Pothole FYP starter — Milestone 1

A simple, deliberately artificial test of OpenCV ArUco marker detection and planar, pixel-to-centimeter measurement. No pothole images, training, or GPU required. The BLUE ellipse is a mock segmentation mask (not DeepLab output). The nominal marker size is 10 cm in the synthetic coordinate system; this is NOT a real-world calibration.

## On macOS Terminal

```bash
cd ~/Downloads/pothole_fyp_starter   # change if you unzipped elsewhere
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Open `outputs/test_image.png`, `outputs/mock_mask.png`, and `outputs/result.png` in Finder.

### VS Code
Open this folder. Command Palette -> **Python: Select Interpreter** -> `.venv/bin/python`.

### Things to know
- Detects ArUco marker 7 and uses its apparent 200-pixel side as a 10-cm reference.
- Generates a blue ellipse and segments it using color thresholding instead of DeepLab.
- Dimensions are derived from its axis-aligned bounding box, not automatically from every pothole's principal axes.
- This simple scale is valid only for a plane with effectively uniform image scale and no material perspective effects. Real road images may require camera calibration and homography, a marker on the same road plane, and ground-truth validation.
- The algorithm measures visible 2D length/width/area, not depth, volume or severity.

## Milestone 3 — YOLO-seg pothole model

A trained segmentation model replaces the mock ellipse (Milestone 1) and manual corner clicks (Milestone 2). Each pothole mask is projected onto the ArUco marker's cm plane (marker ID 0, 10 cm, lying on the road beside the pothole).

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
# ultralytics pulls in opencv-python, which hides cv2.aruco from opencv-contrib-python:
python -m pip uninstall -y opencv-python
python -m pip install --force-reinstall --no-deps "opencv-contrib-python>=4.8,<4.12"

pytest                                   # measurement maths on synthetic images
python scripts/prepare_potholemix.py     # SHREC 2022 Pothole Mix -> YOLO-seg dataset
python scripts/prepare_rdd2022.py        # RDD2022 pothole (D40) boxes -> extra test set
python scripts/train.py --epochs 3       # quick pipeline check on the Mac GPU (mps)
python scripts/train.py                  # full training -> weights/pothole_seg.pt
python scripts/evaluate.py               # mask mAP on the test split
python measure_pothole.py                # pothole_images/ -> pothole_results/
```

Data: both scripts read the downloads from `~/Documents/fyp datasets/` by default (use `--root` / `--zip` to point elsewhere).
- [SHREC 2022 Pothole Mix](https://data.mendeley.com/datasets/kfth5g2xk3/2): pixel masks, unzipped `pothole-mix-v1.0-20220526.zip`. Potholes are red in the masks and cracks green.
- [RDD2022](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547): bounding boxes only, so it can't train segmentation. It's used to test pothole detection on roads from six countries. The zip doesn't need unpacking.

To check accuracy, write a CSV with columns `image,length_cm,width_cm` from ruler measurements and run `python measure_pothole.py --ground-truth truth.csv`.

For help with installing Python or the right compatible macOS wheel see https://docs.opencv.org/doc/doxygen/html/db/dd1/tutorial_py_pip_install.html
