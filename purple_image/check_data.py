"""Validate image-caption pairs before starting a long training run."""
import argparse
import json
from pathlib import Path
from PIL import Image, UnidentifiedImageError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    root = Path(args.data)
    index = root / "captions.jsonl"
    if not index.is_file():
        raise SystemExit(f"ERROR: missing {index}")
    total = valid = bad = 0
    for line_no, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        try:
            item = json.loads(line)
            relative = Path(item["image"])
            caption = str(item["caption"]).strip()
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("image path must be relative and remain inside data/")
            if not caption:
                raise ValueError("caption is empty")
            path = root / relative
            with Image.open(path) as image:
                image.verify()
            valid += 1
        except (KeyError, ValueError, json.JSONDecodeError, OSError, UnidentifiedImageError) as exc:
            bad += 1
            print(f"BAD line {line_no}: {exc}")
    print(f"Total: {total} | valid: {valid} | invalid: {bad}")
    if valid < 2:
        raise SystemExit("ERROR: at least two valid pairs are required.")
    if bad:
        raise SystemExit("Fix invalid rows before training.")
    print("Dataset check passed.")


if __name__ == "__main__":
    main()
