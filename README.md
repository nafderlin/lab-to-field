<div align="center">

# 🌿 Lab to Field

**Crop disease detection that still works when you leave the lab.**

[![tests](https://github.com/nafderlin/lab-to-field/actions/workflows/tests.yml/badge.svg)](https://github.com/nafderlin/lab-to-field/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-2C5F2D.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-4F7CAC.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-D9822B.svg)
![ONNX 8.5 MB](https://img.shields.io/badge/ONNX_model-8.5_MB-97BC62.svg)
![Runs on CPU](https://img.shields.io/badge/runs_offline-CPU-7A8577.svg)

**Live demo:** coming soon · runs entirely in your browser, nothing is uploaded

</div>

---

Our plant-disease model scored **over 90%** on our test photos. Then we tried it on real leaves in a real field, and
it failed, while still sounding completely sure of itself.

<p align="center">
  <img src="results/figures/01_lab_vs_field.png" alt="Lab photos (one leaf, plain background) next to field photos (cluttered, uneven light)" width="85%">
</p>

The test photos were lab photos: one leaf, a plain background and perfect light, just like the training photos.
A high score there only shows the model is good at lab photos. **Lab to Field** is what we built while working out why
it failed and how to fix it:

| | |
|---|---|
| 📏 **Measure** | How big is the lab-to-field gap, on field photos the model never sees during development? |
| 🔍 **Explain** | Why does it happen? (Lab backgrounds give away the answer.) |
| 🔧 **Close** | Transfer learning, targeted augmentation and a small amount of real field data. |
| 🤚 **Abstain** | Teach the model to say *"not sure"* instead of guessing confidently. |

## 📊 Results at a glance

<table>
<tr>
<td align="center"><h3>8.9% → 64.4%</h3>field accuracy<br><sub>scratch CNN → final model</sub></td>
<td align="center"><h3>77.8%</h3>accuracy when it answers<br><sub>answers 71% of field photos, says "not sure" on the rest</sub></td>
<td align="center"><h3>0.69 → 0.09</h3>calibration error (ECE)<br><sub>how far confidence is from actual accuracy</sub></td>
<td align="center"><h3>8.5 MB</h3>FP16 ONNX<br><sub>same accuracy as FP32, runs offline on a CPU</sub></td>
</tr>
</table>

| Model | Trained on | Lab acc. | **Field acc.** (95% CI) | Field ECE ↓ |
|---|---|---:|---:|---:|
| CNN from scratch | lab | 95.8% | 8.9% (5.5–12.7) | 0.69 |
| MobileNetV3, fine-tuned | lab | 98.9% | 30.1% (24–36) | 0.52 |
| ResNet-50, fine-tuned | lab | 98.9% | 32.2% (27–39) | 0.41 |
| MobileNetV3 + background replacement | lab | 97.9% | 39.0% (33–45) | 0.43 |
| DINOv2 ViT-S, frozen | lab | 97.6% | 46.2% (40–53) | 0.33 |
| ⭐ **MobileNetV3 + field data + bg aug + label smoothing** | lab + field | 98.0% | **64.4% (58–70)** | **0.09** |

<sub>28 classes, so chance is 3.6%. Field test = PlantDoc test set (236 photos), used only to report the final numbers,
never to choose settings. INT8 shrinks the model to 4.7 MB but costs 3.8 points; it would need quantization-aware training.</sub>

## 💡 What we learned

> **1. The gap is huge, and the model doesn't know it.**
> A CNN trained from scratch gets 95.8% on lab photos but 8.9% on field photos, and its average confidence is still 78%.

> **2. Lab photos have a shortcut.**
> A classifier that sees **only the four corners** of each image (background, no leaf) gets the disease right 46% of
> the time on lab photos, against 3.6% chance. On field photos the same trick gets 6%. Models learn the background.

> **3. Pretraining matters most among the model choices.** Self-supervised **DINOv2** features transfer best (46%).

> **4. Targeted augmentation beats generic augmentation.** Pasting leaves onto new backgrounds adds about 13 points.
> RandAugment adds nothing.

> **5. Nothing beats real field photos.** About 2,000 PlantDoc photos take MobileNetV3 from 30% to 64%.

> **6. Label smoothing makes the model honest.** Calibration error drops from 0.44 to 0.16 with similar accuracy.

## ✅ How we kept the evaluation honest

- [x] PlantVillage split **by leaf**. The official split still had 37 leaves in both train and test; we removed them.
- [x] PlantDoc **train/test duplicates** found with perceptual hashing (15) and removed from training
- [x] Every setting chosen on **field validation** data; the field test set is only used for the final report
- [x] **95% bootstrap confidence intervals** on every field number
- [x] Calibration (ECE, reliability diagrams) and abstention measured on held-out data

## 🚀 Quick start

**In the browser.** The demo in [`web/`](web/) runs the 8.5 MB model on your device with
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). There's no server, so photos never leave the phone or laptop.
It re-implements Pillow's resize in JavaScript, so it gives the same answers as Python (checked in CI on every bundled photo).

```bash
scripts/build_space.sh && python -m http.server -d build/space 8000     # then open http://localhost:8000
```

**The full app** (explanation pages + diagnosis). The trained ONNX models are in `export/`, so you don't need PyTorch,
a GPU or any training:

```bash
pip install -r app/requirements.txt     # streamlit, onnxruntime, numpy, pillow
streamlit run app/streamlit_app.py
```

Open **Diagnose a leaf** and upload a photo, or drop your own photos into [`test_images/`](test_images/).

**From the command line or Python:**

```bash
PYTHONPATH=src python -m lab_to_field.cli test_images/ --json     # or `lab-to-field-predict` after `uv sync`
```

```python
from lab_to_field.inference import LeafDiagnoser

d = LeafDiagnoser("fp16").diagnose("leaf.jpg")
print(d.answer, f"{d.confidence:.0%}")   # "Tomato late blight 93%", or "Not sure" below the calibrated threshold
```

**With Docker:** `docker build -t lab-to-field . && docker run -p 8501:8501 lab-to-field`

**Reproduce everything from scratch** (NVIDIA GPU, about 1.5 h on an RTX 4050):

```bash
uv sync                                                                   # Python 3.12, PyTorch (CUDA 12.8)
uv run python scripts/download_plantvillage_subset.py                     # ~480 MB PlantVillage subset
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset data/raw/plantdoc
scripts/run_all_notebooks.sh
```

If a notebook's results already exist in `results/<name>.json`, re-running it loads them instead of retraining.

<details>
<summary><b>📁 Project structure</b></summary>

```
notebooks/
  01_problem_and_data       task, datasets, leakage checks, background-shortcut test
  02_cnn_from_scratch       baseline CNN, tensor shapes, parameter counting
  03_transfer_learning      linear probe vs fine-tune: MobileNetV3, EfficientNet, ResNet, DINOv2
  04_augmentation           controlled augmentation ablation, background replacement
  05_training_choices       optimizers, losses, activations, calibration, abstention, final model
  06_explain_and_export     Grad-CAM, ONNX export, FP16 / INT8
src/lab_to_field/           training code (config, data, models, train, metrics, calibrate, experiment)
                            + inference.py / cli.py: PyTorch-free ONNX inference used by the app and CLI
app/                        Streamlit app: overview, step-by-step explanation pages, leaf diagnosis
web/                        in-browser demo (ONNX Runtime Web); web/test checks it against Python
tests/                      inference tests, incl. a golden snapshot of model outputs (run: pytest)
deploy/space/               Hugging Face Space card; scripts/build_space.sh builds the static Space
export/                     trained ONNX models (FP32 / FP16 / INT8) + labels and calibration
results/                    metrics (JSON) and figures
test_images/                your own leaf photos to try in the app
scripts/                    data download, notebook runner
docs/                       presentation slides
```

</details>

## 🧭 Limitations and what's next

- PlantDoc is small (236 test photos), has some label noise, and has **no wheat, rice or cotton**.
- Grad-CAM shows the final model still sometimes looks at the background.
- **Next:** distil DINOv2 into MobileNetV3, quantization-aware training for INT8, leaf segmentation, expert-labelled
  photos from real farms, an Android app.

> ⚠️ This is a research project, not a substitute for an agronomist.

## 🤝 Contributing

Issues and PRs are welcome, especially **field photos of crops PlantDoc doesn't cover** (wheat, rice, cotton) and
results on other field datasets. If you try the model on your own leaves, open an issue and tell us what worked and what didn't.

## 📚 Data and credits

- **PlantVillage** (Hughes & Salathé; Mohanty et al. 2016). [Hugging Face `mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage), CC BY-SA 3.0. We use a leaf-grouped subset of 28 classes.
- **PlantDoc** (Singh et al. 2020). [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset), CC BY 4.0. The sample photos in `app/samples/` come from its test set.

Started as a group project for *Edge AI and Robotics: Data Center Vision* (Thapar Institute).

## 👥 Authors

**Abubakr** · **Mehul** · **Xiaodong Tan**

## 📄 License

Code: [MIT](LICENSE). Datasets and sample photos keep their own licenses (see above).
