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

For help with installing Python or the right compatible macOS wheel see https://docs.opencv.org/doc/doxygen/html/db/dd1/tutorial_py_pip_install.html
