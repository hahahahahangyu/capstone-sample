"""Convert SHREC 2022 Pothole Mix into an Ultralytics YOLO-seg dataset.

Expected layout (as released, pothole-mix-v1.0-20220526.zip):
    <root>/{training,validation,testing}/<source>/{images,masks}/
Images and masks are paired by file stem. Masks are colour coded; in this
release potholes are RED (RGB 255,0,0) and cracks GREEN, despite the SHREC
paper's figure caption. Crack-only images are kept as pothole-free
background examples so the model learns that cracks are not potholes.

The official training/validation/testing split is kept, so results are
comparable with the SHREC 2022 track.

Outputs:
    data/processed/potholemix/{images,labels}/{train,val,test}/<source>__<name>
    data/manifests/potholemix_{train,val,test}.csv  (image, source, n_potholes)
    data/pothole_seg.yaml
"""

from pathlib import Path
import argparse
import csv
import shutil
import sys

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from pothole.measure import mask_contours  # noqa: E402

DEFAULT_ROOT = (Path.home() / "Documents" / "fyp datasets" / "SHREC2022"
                / "Pothole-Mix" / "pothole-mix")
SPLITS = {"training": "train", "validation": "val", "testing": "test"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=BASE / "data/processed/potholemix")
    parser.add_argument("--mask-color", type=str, default="255,0,0",
                        help="R,G,B of the pothole class in the masks")
    parser.add_argument("--min-area", type=int, default=100,
                        help="drop mask blobs smaller than this many pixels")
    parser.add_argument("--max-size", type=int, default=1280,
                        help="downscale images so the long side is at most this; "
                             "decoding 2560 px originals every epoch made training ~4x slower")
    return parser.parse_args()


def save_image(src, dst, max_size):
    image = cv2.imread(str(src))
    scale = max_size / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    # Labels are normalised, so they stay valid after resizing.
    cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 95])


def find_pairs(source_dir):
    masks = {p.stem: p for p in (source_dir / "masks").iterdir()
             if p.suffix.lower() in IMAGE_EXTS}
    return [(p, masks[p.stem]) for p in sorted((source_dir / "images").iterdir())
            if p.suffix.lower() in IMAGE_EXTS and p.stem in masks]


def load_binary_mask(path, mask_color):
    mask = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if mask is None:
        return None
    r, g, b = (int(v) for v in mask_color.split(","))
    # Small tolerance in case a mask was saved with lossy compression.
    target = np.array([b, g, r], dtype=np.int16)
    close = np.abs(mask.astype(np.int16) - target).max(axis=2) <= 40
    return close.astype(np.uint8) * 255


def yolo_label_lines(mask, min_area):
    """One 'class x1 y1 x2 y2 ...' line per blob, coordinates normalised."""
    height, width = mask.shape
    lines = []
    for contour in mask_contours(mask, min_area):
        # Simplify so label files stay small; 0.2% of perimeter is visually exact.
        eps = 0.002 * cv2.arcLength(contour, True)
        poly = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        coords = (poly / [width, height]).clip(0, 1).flatten()
        lines.append("0 " + " ".join(f"{v:.6f}" for v in coords))
    return lines


def main():
    args = parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(f"Pothole Mix folder not found: {args.root}")
    if args.out.exists():
        shutil.rmtree(args.out)
    manifests = BASE / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    for folder, split in SPLITS.items():
        image_out = args.out / "images" / split
        label_out = args.out / "labels" / split
        image_out.mkdir(parents=True)
        label_out.mkdir(parents=True)
        rows = []
        for source_dir in sorted(p for p in (args.root / folder).iterdir() if p.is_dir()):
            with_pothole = 0
            pairs = find_pairs(source_dir)
            for image_path, mask_path in pairs:
                mask = load_binary_mask(mask_path, args.mask_color)
                if mask is None:
                    print(f"Cannot read mask {mask_path}. Skipping.")
                    continue
                lines = yolo_label_lines(mask, args.min_area)
                # Sources reuse names like 001.jpg, so prefix with the source.
                name = f"{source_dir.name}__{image_path.stem}.jpg"
                save_image(image_path, image_out / name, args.max_size)
                (label_out / f"{Path(name).stem}.txt").write_text("\n".join(lines))
                rows.append([name, source_dir.name, len(lines)])
                with_pothole += bool(lines)
            print(f"{split:5s} {source_dir.name:28s} {len(pairs):5d} images, "
                  f"{with_pothole:5d} with potholes")
        with (manifests / f"potholemix_{split}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "source", "n_potholes"])
            writer.writerows(rows)
        total_pos = sum(r[2] > 0 for r in rows)
        print(f"{split:5s} TOTAL {len(rows)} images, {total_pos} with potholes\n")

    yaml_path = BASE / "data" / "pothole_seg.yaml"
    yaml_path.write_text(
        f"path: {args.out}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: pothole\n"
    )
    print(f"Dataset config written to {yaml_path}")


if __name__ == "__main__":
    main()
