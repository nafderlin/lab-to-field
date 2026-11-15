#!/usr/bin/env bash
# Assemble the files the Hugging Face Space needs into one folder (default: build/space).
# Upload it with:  hf upload <user>/<space> build/space . --repo-type space
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-build/space}"

rm -rf "$out" && mkdir -p "$out/results"
cp -R app src export test_images .streamlit Dockerfile "$out/"
cp -R results/figures "$out/results/"
cp results/0*.json "$out/results/"
cp deploy/space/README.md "$out/README.md"           # Space card with the Hugging Face config header
find "$out" -name __pycache__ -type d -prune -exec rm -rf {} +
echo "Space files in $out ($(du -sh "$out" | cut -f1))"
