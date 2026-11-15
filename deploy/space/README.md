---
title: Lab to Field
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8501
license: mit
short_description: Crop disease detection that works on real field photos
tags:
  - image-classification
  - plant-disease
  - onnx
  - streamlit
  - distribution-shift
---

# 🌿 Lab to Field

Plant-disease models score ~99% on lab photos and collapse on real field photos. This demo runs a MobileNetV3 trained
to hold up in the field (64.4% on PlantDoc field photos vs 30% for a lab-only model) from an 8.5 MB ONNX file on CPU,
and it says **"not sure"** instead of guessing when a photo is outside what it knows.

**Try it:** open *Diagnose a leaf*, pick a sample field photo or upload your own (one leaf, filling the frame, daylight).

Code, notebooks and the full write-up: **https://github.com/nafderlin/lab-to-field**

Not a substitute for an agronomist.
