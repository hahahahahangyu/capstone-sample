"""Extract RDD2022 pothole (D40) images and boxes into YOLO detection format.

Reads the official CRDDC release zip directly (outer zip -> per-country zip ->
Pascal VOC xml + jpg), so the 13 GB archive never needs unpacking. Only train
images have annotations; images with at least one D40 box are kept.

RDD2022 has boxes, not outlines, so this set is for testing whether the
segmentation model *finds* potholes on unseen roads, not for mask training.

Outputs:
    data/processed/rdd2022_d40/images/<Country>_xxxxxx.jpg
    data/processed/rdd2022_d40/labels/<Country>_xxxxxx.txt  (0 xc yc w h)
    data/manifests/rdd2022_d40.csv  (image, country, n_potholes, width, height)
"""

from pathlib import Path
import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
import zipfile

BASE = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = (Path.home() / "Documents" / "fyp datasets" / "RDD2022_fullDataset"
               / "RDD2022_released_through_CRDDC2022.zip")
POTHOLE = "D40"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out", type=Path, default=BASE / "data/processed/rdd2022_d40")
    parser.add_argument("--countries", nargs="*", default=None,
                        help="e.g. Japan India (default: all)")
    return parser.parse_args()


def pothole_boxes(xml_bytes):
    """Return (width, height, [(xmin, ymin, xmax, ymax), ...]) for D40 only."""
    root = ET.fromstring(xml_bytes)
    width = float(root.findtext("size/width"))
    height = float(root.findtext("size/height"))
    boxes = []
    for obj in root.iter("object"):
        if obj.findtext("name") != POTHOLE:
            continue
        box = obj.find("bndbox")
        boxes.append(tuple(float(box.findtext(k))
                           for k in ("xmin", "ymin", "xmax", "ymax")))
    return width, height, boxes


def yolo_line(box, width, height):
    xmin, ymin, xmax, ymax = box
    xmin, xmax = max(0.0, xmin), min(width, xmax)
    ymin, ymax = max(0.0, ymin), min(height, ymax)
    return (f"0 {(xmin + xmax) / 2 / width:.6f} {(ymin + ymax) / 2 / height:.6f} "
            f"{(xmax - xmin) / width:.6f} {(ymax - ymin) / height:.6f}")


def main():
    args = parse_args()
    if not args.zip.is_file():
        raise FileNotFoundError(f"RDD2022 zip not found: {args.zip}")
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True)
    (args.out / "labels").mkdir(parents=True)
    manifests = BASE / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    rows = []
    with zipfile.ZipFile(args.zip) as outer:
        country_zips = [n for n in outer.namelist() if n.endswith(".zip")]
        for member in sorted(country_zips):
            country = Path(member).stem
            if args.countries and country not in args.countries:
                continue
            with zipfile.ZipFile(outer.open(member)) as inner:
                names = set(inner.namelist())
                xmls = sorted(n for n in names if n.endswith(".xml"))
                kept = boxes_total = 0
                for xml_name in xmls:
                    width, height, boxes = pothole_boxes(inner.read(xml_name))
                    if not boxes:
                        continue
                    stem = Path(xml_name).stem
                    image_name = f"{country}/train/images/{stem}.jpg"
                    if image_name not in names:
                        print(f"  missing image for {xml_name}")
                        continue
                    (args.out / "images" / f"{stem}.jpg").write_bytes(
                        inner.read(image_name))
                    (args.out / "labels" / f"{stem}.txt").write_text(
                        "\n".join(yolo_line(b, width, height) for b in boxes))
                    rows.append([f"{stem}.jpg", country, len(boxes),
                                 int(width), int(height)])
                    kept += 1
                    boxes_total += len(boxes)
            print(f"{country}: {kept} pothole images ({boxes_total} boxes) "
                  f"out of {len(xmls)} annotated")

    with (manifests / "rdd2022_d40.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "country", "n_potholes", "width", "height"])
        writer.writerows(rows)
    print(f"\nTotal: {len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
