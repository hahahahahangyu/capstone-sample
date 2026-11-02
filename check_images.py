
from pathlib import Path
import cv2

folder = Path(__file__).resolve().parent / "test_images"

for path in sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
    image = cv2.imread(str(path))

    if image is None:
        print(f"Failed to read: {path.name}")
    else:
        height, width = image.shape[:2]
        print(f"Loaded {path.name}: {width} x {height}") #in pixels
