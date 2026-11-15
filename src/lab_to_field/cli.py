"""Diagnose leaf photos from the command line.

    lab-to-field-predict leaf.jpg
    lab-to-field-predict test_images/ --precision int8 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lab_to_field.inference import NOT_SURE, PRECISIONS, LeafDiagnoser

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def collect(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        elif p.exists():
            files.append(p)
        else:
            sys.exit(f"error: {p} does not exist")
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab-to-field-predict", description="Diagnose crop disease from leaf photos.")
    ap.add_argument("paths", nargs="+", type=Path, help="image files or folders of images")
    ap.add_argument("--precision", choices=PRECISIONS, default="fp16", help="which exported model to use")
    ap.add_argument("--top", type=int, default=3, help="how many top guesses to show")
    ap.add_argument("--json", action="store_true", help="print one JSON object per photo")
    args = ap.parse_args(argv)

    files = collect(args.paths)
    if not files:
        sys.exit("error: no images found")
    model = LeafDiagnoser(args.precision)
    for f in files:
        d = model.diagnose(f)
        if args.json:
            print(json.dumps({"file": str(f), **d.to_dict(args.top)}))
            continue
        print(f"{f.name}\n  → {d.answer}" + (f" (best guess: {d.best_guess})" if d.answer == NOT_SURE else ""))
        for label, p in d.top(args.top):
            print(f"     {p:6.1%}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
