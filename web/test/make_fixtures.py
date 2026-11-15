"""Write what Python sees for every bundled photo, so web/test can check the JS port against it.

    PYTHONPATH=src python web/test/make_fixtures.py build/web-fixtures
"""
import json
import sys
from pathlib import Path

from PIL import Image

from lab_to_field.inference import LeafDiagnoser, preprocess

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build" / "web-fixtures")
out.mkdir(parents=True, exist_ok=True)

model = LeafDiagnoser("fp16")
cases = []
for i, f in enumerate(sorted([*(ROOT / "test_images").glob("*.jpg"), *(ROOT / "app" / "samples").glob("*.jpg")])):
    img = Image.open(f).convert("RGB")
    x, _ = preprocess(img)
    logits, _ = model.logits(img)
    d = model.diagnose(img)
    (out / f"{i}.rgb").write_bytes(img.tobytes())
    (out / f"{i}.tensor").write_bytes(x.tobytes())
    cases.append({"id": i, "file": f.relative_to(ROOT).as_posix(), "width": img.width, "height": img.height,
                  "logits": [float(v) for v in logits], "answer": d.answer, "confidence": d.confidence})
(out / "cases.json").write_text(json.dumps({"meta": json.loads((ROOT / "export" / "labels.json").read_text()),
                                            "cases": cases}))
print(f"wrote {len(cases)} cases to {out}")
