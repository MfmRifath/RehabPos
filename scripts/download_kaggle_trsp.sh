#!/usr/bin/env bash
# Toronto Rehab Stroke Pose (Kaggle derekdb/toronto-robot-stroke-posture-dataset).
# Requires the Kaggle CLI and YOUR credentials (~/.kaggle/kaggle.json). Check the dataset's
# terms on Kaggle first. NEVER run automatically (owner approval + credentials required).
set -euo pipefail
if [[ "${1:-}" != "--yes" ]]; then
  echo "usage: $0 --yes   (needs Kaggle credentials; owner approval required)" >&2
  exit 2
fi
command -v kaggle >/dev/null || { echo "kaggle CLI not found" >&2; exit 1; }
mkdir -p data/raw/trsp
kaggle datasets download -d derekdb/toronto-robot-stroke-posture-dataset -p data/raw/trsp --unzip
echo "done -> data/raw/trsp"
