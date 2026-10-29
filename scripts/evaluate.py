"""Report box and mask mAP of the trained model on the held-out test split."""

from pathlib import Path
import argparse

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=BASE / "weights/pothole_seg.pt")
    parser.add_argument("--data", type=Path, default=BASE / "data/pothole_seg.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    metrics = YOLO(str(args.weights)).val(
        data=str(args.data), split=args.split, device=args.device,
        project=str(BASE / "runs"), name=f"eval_{args.split}", exist_ok=True,
    )
    print(f"\n{args.split} split")
    print(f"  Box  mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")
    print(f"  Mask mAP50: {metrics.seg.map50:.3f}   mAP50-95: {metrics.seg.map:.3f}")


if __name__ == "__main__":
    main()
