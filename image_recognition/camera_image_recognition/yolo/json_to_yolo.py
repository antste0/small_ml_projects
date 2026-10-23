"""Convert LabelMe / X-AnyLabeling (JSON) annotations to YOLO format.

Examples:
    python json_to_yolo.py
    python json_to_yolo.py -i ./assets -o ./dataset --classes bottle flashlight pen
    python json_to_yolo.py --split train=0.8 val=0.2 --recursive --clean
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SHAPE_TYPES = ("rectangle", "polygon", "rotation", "circle")


def parse_split(items: list[str]) -> dict[str, float]:
    split = {}
    for item in items:
        name, _, ratio = item.partition("=")
        if not ratio:
            raise SystemExit(f"Invalid split: {item!r} (expected name=fraction)")
        split[name] = float(ratio)
    total = sum(split.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        print(f"Warning: split ratios sum to {total:.3f} - normalizing")
        split = {k: v / total for k, v in split.items()}
    return split


def shape_points(shape: dict) -> list[list[float]]:
    points = shape["points"]
    if shape.get("shape_type") == "circle":
        (cx, cy), (px, py) = points[:2]
        r = math.hypot(px - cx, py - cy)
        return [[cx - r, cy - r], [cx + r, cy + r]]
    return points


def convert_bbox(points, img_w, img_h):
    xs = [min(max(p[0], 0), img_w) for p in points]
    ys = [min(max(p[1], 0), img_h) for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    width    = (x_max - x_min) / img_w
    height   = (y_max - y_min) / img_h
    return x_center, y_center, width, height


def load_json(json_file: Path) -> dict:
    with open(json_file, encoding="utf-8") as f:
        return json.load(f)


def collect_classes(json_files: list[Path]) -> list[str]:
    labels = set()
    for json_file in json_files:
        labels.update(s["label"] for s in load_json(json_file).get("shapes", []))
    return sorted(labels)


def convert_json(data: dict, class_map: dict[str, int], name: str) -> list[str]:
    img_w, img_h = data["imageWidth"], data["imageHeight"]
    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"]
        shape_type = shape.get("shape_type", "polygon")
        if label not in class_map:
            print(f"Skipping unknown class: {label} ({name})")
            continue
        if shape_type not in SHAPE_TYPES:
            print(f"Skipping shape type: {shape_type} ({name})")
            continue
        x, y, w, h = convert_bbox(shape_points(shape), img_w, img_h)
        if w <= 0 or h <= 0:
            print(f"Skipping empty bbox: {label} ({name})")
            continue
        lines.append(f"{class_map[label]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return lines


def find_image(json_file: Path, data: dict) -> Path | None:
    # try the path stored in the JSON first, then the same stem with a common extension
    if data.get("imagePath"):
        candidate = json_file.parent / Path(data["imagePath"].replace("\\", "/"))
        if candidate.exists():
            return candidate
    return next(
        (json_file.with_suffix(ext) for ext in IMAGE_EXTS + tuple(e.upper() for e in IMAGE_EXTS)
         if json_file.with_suffix(ext).exists()),
        None,
    )


def assign_splits(files: list[Path], split: dict[str, float]) -> list[tuple[str, Path]]:
    names = list(split)
    n = len(files)
    result, start = [], 0
    for i, name in enumerate(names):
        # the last split takes the remainder so no files are lost to rounding
        end = n if i == len(names) - 1 else start + int(n * split[name])
        result += [(name, f) for f in files[start:end]]
        start = end
    return result


def write_yaml(yaml_path: Path, output_dir: Path, split: dict[str, float], classes: list[str]):
    lines = [f"path: {output_dir.as_posix()}"]
    lines += [f"{name}: images/{name}" for name in split]
    lines += ["", f"nc: {len(classes)}", f"names: {json.dumps(classes, ensure_ascii=False)}", ""]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Convert JSON annotations (LabelMe / X-AnyLabeling) to YOLO format")
    p.add_argument("-i", "--input", type=Path, default=Path("./assets"), help="directory with images and .json files")
    p.add_argument("-o", "--output", type=Path, default=Path("./dataset"), help="output dataset directory")
    p.add_argument("-c", "--classes", nargs="+", help="class list (default: detected from annotations, sorted)")
    p.add_argument("-s", "--split", nargs="+", default=["train=0.7", "val=0.2", "test=0.1"],
                   help="split as name=fraction (default: train=0.7 val=0.2 test=0.1)")
    p.add_argument("--seed", type=int, default=42, help="random seed for the split")
    p.add_argument("--yaml", type=Path, default=Path("./dataset.yaml"), help="where to write dataset.yaml")
    p.add_argument("--no-yaml", action="store_true", help="do not generate dataset.yaml")
    p.add_argument("-r", "--recursive", action="store_true", help="search for .json files recursively")
    p.add_argument("--clean", action="store_true", help="remove images/ and labels/ before converting")
    p.add_argument("--skip-missing", action="store_true", help="skip annotations without a matching image")
    return p.parse_args()


def main():
    args = parse_args()
    split = parse_split(args.split)

    pattern = "**/*.json" if args.recursive else "*.json"
    json_files = sorted(args.input.glob(pattern))
    if not json_files:
        raise SystemExit(f"No .json files found in {args.input}")

    classes = args.classes or collect_classes(json_files)
    class_map = {name: i for i, name in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")

    if args.clean:
        for sub in ("images", "labels"):
            shutil.rmtree(args.output / sub, ignore_errors=True)
    for name in split:
        (args.output / "images" / name).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / name).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    random.shuffle(json_files)

    counts = dict.fromkeys(split, 0)
    for name, json_file in assign_splits(json_files, split):
        data = load_json(json_file)
        img_file = find_image(json_file, data)
        if img_file is None:
            print(f"Missing image for: {json_file}")
            if args.skip_missing:
                continue

        # with -r, names can repeat across subdirectories, so prefix them with the relative path
        rel = json_file.relative_to(args.input).with_suffix("")
        stem = "_".join(rel.parts)

        lines = convert_json(data, class_map, json_file.name)
        (args.output / "labels" / name / f"{stem}.txt").write_text("\n".join(lines))
        if img_file:
            shutil.copy2(img_file, args.output / "images" / name / f"{stem}{img_file.suffix}")

        counts[name] += 1

    if not args.no_yaml:
        write_yaml(args.yaml, args.output, split, classes)
        print(f"Saved {args.yaml}")

    summary = " / ".join(f"{c} {name}" for name, c in counts.items())
    print(f"Conversion finished: {summary}")


if __name__ == "__main__":
    main()
