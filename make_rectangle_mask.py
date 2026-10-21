
from pathlib import Path
import cv2
import numpy as np

base = Path(__file__).resolve().parent
image_path = base / "test_images" / "rec_90.jpeg"
output = base / "outputs"
output.mkdir(exist_ok=True)

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)

# Resize only for convenient display
h, w = image.shape[:2]
scale = min(1.0, 1000 / h)
display = cv2.resize(image, None, fx=scale, fy=scale)

points = []

def mouse_callback(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Corner {len(points)}: ({x}, {y})")

window = "Select 4 Box Corners"
cv2.namedWindow(window)
cv2.setMouseCallback(window, mouse_callback)

while True:
    preview = display.copy()

    for point in points:
        cv2.circle(preview, point, 5, (0, 0, 255), -1)

    if len(points) >= 2:
        cv2.polylines(
            preview, [np.array(points)],
            len(points) == 4, (0, 255, 0), 2
        )

    cv2.imshow(window, preview)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("r"):
        points.clear()
    elif key == ord("s") and len(points) == 4:
        break
    elif key == 27:
        cv2.destroyAllWindows()
        raise SystemExit("Cancelled")

cv2.destroyAllWindows()

# Convert displayed coordinates to original pixels
polygon = np.round(
    np.array(points) / scale
).astype(np.int32)

# White = box top surface, black = background
mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(mask, [polygon], 255)

# Create an outlined result for inspection
result = image.copy()
cv2.polylines(result, [polygon], True, (0, 255, 0), 8)

cv2.imwrite(str(output / "rectangle_mask.png"), mask)
cv2.imwrite(str(output / "rectangle_outline.png"), result)

print("Mask and outline saved successfully.")
print("Masked area in pixels:", cv2.countNonZero(mask))
