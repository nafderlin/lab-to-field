"""Tests for the ONNX inference path (no PyTorch needed).

Run:  PYTHONPATH=src pytest
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lab_to_field import cli
from lab_to_field.inference import NOT_SURE, Diagnosis, LeafDiagnoser, preprocess, softmax

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "golden_predictions.json").read_text())


@pytest.fixture(scope="module")
def fp16():
    return LeafDiagnoser("fp16")


# --- preprocessing -----------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(640, 480), (480, 640), (224, 224), (3000, 200)])
def test_preprocess_shape_and_dtype(size):
    x, crop = preprocess(Image.new("RGB", size, (120, 160, 80)))
    assert x.shape == (1, 3, 224, 224)
    assert x.dtype == np.float32
    assert crop.size == (224, 224)


def test_preprocess_handles_grayscale_and_alpha():
    for mode in ["L", "RGBA", "P"]:
        x, _ = preprocess(Image.new(mode, (300, 300)))
        assert x.shape == (1, 3, 224, 224)


def test_preprocess_normalises_with_imagenet_stats():
    # A pixel equal to the ImageNet mean must map to ~0 in every channel.
    mean_rgb = tuple(round(v * 255) for v in (0.485, 0.456, 0.406))
    x, _ = preprocess(Image.new("RGB", (256, 256), mean_rgb))
    assert np.abs(x).max() < 0.01


# --- calibration -------------------------------------------------------------------------------

def test_softmax_is_a_distribution():
    p = softmax(np.array([2.0, -1.0, 0.5, 10.0]))
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()


def test_temperature_changes_confidence_not_the_answer():
    z = np.random.default_rng(0).normal(size=28) * 3
    sharp, soft = softmax(z, 0.5), softmax(z, 2.0)
    assert sharp.argmax() == soft.argmax() == z.argmax()
    assert sharp.max() > softmax(z).max() > soft.max()


def test_softmax_is_numerically_stable():
    assert np.isfinite(softmax(np.array([1000.0, 1001.0, 999.0]))).all()


# --- abstention --------------------------------------------------------------------------------

def test_abstains_below_threshold():
    classes = ["a", "b", "c"]
    sure = Diagnosis(np.array([0.8, 0.1, 0.1]), classes, threshold=0.5)
    unsure = Diagnosis(np.array([0.4, 0.35, 0.25]), classes, threshold=0.5)
    assert (sure.answer, sure.is_sure) == ("a", True)
    assert (unsure.answer, unsure.best_guess, unsure.is_sure) == (NOT_SURE, "a", False)
    assert [c for c, _ in unsure.top(2)] == ["a", "b"]


# --- the exported models -----------------------------------------------------------------------

def test_metadata_is_consistent(fp16):
    assert len(fp16.classes) == 28 == len(set(fp16.classes))
    assert 0 < fp16.threshold < 1
    assert 0.5 < fp16.temperature < 2


def test_rejects_unknown_precision():
    with pytest.raises(ValueError):
        LeafDiagnoser("fp8")


# ONNX Runtime's INT8 kernels round differently on x86 and ARM CPUs: confidences move by up to ~2.3 points
# between an Apple Silicon Mac and the Linux CI runner (FP16: 0.03 points), but no answer changes.
TOLERANCE = {"fp16": 2e-3, "int8": 5e-2}


@pytest.mark.parametrize("precision", ["fp16", "int8"])
def test_predictions_match_golden_snapshot(precision):
    """Any change to preprocessing, calibration or the model files must be deliberate."""
    model = LeafDiagnoser(precision)
    for rel, expected in GOLDEN[precision].items():
        d = model.diagnose(ROOT / rel)
        assert d.best_guess == expected["best_guess"], rel
        assert d.answer == expected["answer"], rel
        assert d.confidence == pytest.approx(expected["confidence"], abs=TOLERANCE[precision]), rel


def test_fp16_matches_fp32(fp16):
    fp32 = LeafDiagnoser("fp32")
    for f in sorted((ROOT / "test_images").glob("*.jpg")):
        a, b = fp16.diagnose(f), fp32.diagnose(f)
        assert a.best_guess == b.best_guess, f.name
        assert np.abs(a.probs - b.probs).max() < 0.01, f.name


def test_uncalibrated_probabilities_keep_the_ranking(fp16):
    img = ROOT / "test_images" / "Tomato late blight - field photo 12.jpg"
    calibrated, raw = fp16.diagnose(img), fp16.diagnose(img, temperature=1.0)
    assert calibrated.best_guess == raw.best_guess


# --- command line ------------------------------------------------------------------------------

def test_cli_json_output(capsys):
    cli.main([str(ROOT / "test_images"), "--json", "--top", "2"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == len(list((ROOT / "test_images").glob("*.jpg")))
    assert all(len(r["top"]) == 2 and 0 <= r["confidence"] <= 1 for r in lines)


def test_cli_missing_file():
    with pytest.raises(SystemExit):
        cli.main(["does-not-exist.jpg"])
