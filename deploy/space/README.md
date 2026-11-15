---
title: Lab to Field
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
license: mit
short_description: Crop disease check that runs in your browser
tags:
  - image-classification
  - plant-disease
  - onnx
  - onnxruntime-web
  - distribution-shift
---

# 🌿 Lab to Field

Plant-disease models score ~99% on lab photos and collapse on real field photos. This demo runs a MobileNetV3 trained
to hold up in the field (64.4% on PlantDoc field photos vs 30% for a lab-only model). It runs **entirely in your
browser** with ONNX Runtime Web (8.5 MB FP16 model, nothing is uploaded), and it says **"not sure"** instead of
guessing when a photo is outside what it knows.

Code, notebooks and the full write-up: **https://github.com/nafderlin/lab-to-field**

Not a substitute for an agronomist.
