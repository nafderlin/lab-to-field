#!/usr/bin/env bash
# Build the static in-browser demo (default: build/space). This is what the Hugging Face Space serves.
#   Preview locally:  python -m http.server -d build/space 8000
#   Upload by hand:   hf upload <user>/<space> build/space . --repo-type space
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-build/space}"

rm -rf "$out" && mkdir -p "$out/model" "$out/samples"
cp web/index.html web/style.css web/app.js web/preprocess.js "$out/"
cp export/plant_fp16.onnx export/labels.json "$out/model/"
cp deploy/space/README.md "$out/README.md"           # Space card with the Hugging Face config header

# Sample field photos, renamed to URL-safe names, with the true label taken from the file name.
python3 - "$out/samples" <<'PY'
import json, shutil, sys
from pathlib import Path

out = Path(sys.argv[1])
classes = {c.lower(): c for c in json.loads(Path("export/labels.json").read_text())["classes"]}
photos = sorted(Path("app/samples").glob("*.jpg")) + sorted(Path("test_images").glob("*.jpg"))
samples = []
for i, p in enumerate(photos, 1):
    name = f"field-{i:02d}.jpg"
    shutil.copy(p, out / name)
    head = p.stem.split(" - ")[0].split(" (")[0].strip()
    samples.append({"src": f"samples/{name}", "name": p.stem, "label": classes.get(head.lower())})
(out / "samples.json").write_text(json.dumps(samples, indent=1))
print(f"{len(samples)} sample photos")
PY
echo "Static demo in $out ($(du -sh "$out" | cut -f1))"
