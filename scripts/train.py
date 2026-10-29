"""Train a YOLO-seg pothole model on data/pothole_seg.yaml.

Runs go to runs/<name>/; the best checkpoint is copied to weights/pothole_seg.pt,
which is what measure_pothole.py loads by default.
"""

from pathlib import Path
import argparse
import shutil

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE / "data/pothole_seg.yaml")
    parser.add_argument("--model", default="yolo11n-seg.pt",
                        help="pretrained start point, e.g. yolo11s-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps", help="mps, cpu or 0 (CUDA)")
    parser.add_argument("--name", default="pothole_seg")
    parser.add_argument("--patience", type=int, default=20,
                        help="stop early after this many epochs without improvement")
    parser.add_argument("--cache", default=False, choices=[False, "ram", "disk"],
                        help="cache decoded images (ram needs ~3 GB free)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f"{args.data} not found. Run scripts/prepare_potholemix.py first."
        )
    model = YOLO(args.model)
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, patience=args.patience,
                cache=args.cache,
                project=str(BASE / "runs"), name=args.name, exist_ok=True)

    best = BASE / "runs" / args.name / "weights" / "best.pt"
    weights_dir = BASE / "weights"
    weights_dir.mkdir(exist_ok=True)
    shutil.copy2(best, weights_dir / "pothole_seg.pt")
    print(f"Best weights copied to {weights_dir / 'pothole_seg.pt'}")


if __name__ == "__main__":
    main()
