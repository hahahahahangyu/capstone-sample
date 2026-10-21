
from pathlib import Path
import cv2

# Folder paths
base = Path(__file__).resolve().parent
input_folder = base / "test_images"
output_folder = base / "aruco_results"
output_folder.mkdir(exist_ok=True)

# IMPORTANT: Change this to your actual marker dictionary
DICT_NAME = "DICT_4X4_50"

dictionary = cv2.aruco.getPredefinedDictionary(
    getattr(cv2.aruco, DICT_NAME)
)

detector = cv2.aruco.ArucoDetector(
    dictionary,
    cv2.aruco.DetectorParameters()
)

for image_path in sorted(input_folder.glob("*.jpeg")):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Cannot read {image_path.name}")
        continue

    corners, ids, rejected = detector.detectMarkers(image)

    if ids is not None:
        print(f"{image_path.name}: Found marker IDs {ids.flatten().tolist()}")
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
    else:
        print(f"{image_path.name}: No ArUco marker detected")

    output_path = output_folder / image_path.name
    cv2.imwrite(str(output_path), image)

print("Detection test finished.")
