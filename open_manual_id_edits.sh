#!/usr/bin/env bash
set -e

cd /home/nagas/research/mediapipewith3cams || exit 1

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export DISPLAY=":0"
export WAYLAND_DISPLAY="wayland-0"
export XDG_RUNTIME_DIR="/mnt/wslg/runtime-dir"
export QT_QPA_PLATFORM="wayland"

mkdir -p output/3dposeestimation/video

PYTHON_BIN="python"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

for i in 1 2 3 4 5 6; do
  csv="output/3dposeestimation/match1_${i}_smoothed.csv"
  out="output/3dposeestimation/match1_${i}_manual.csv"
  log="output/3dposeestimation/video/manual_id_edit_match1_${i}.log"
  nohup "${PYTHON_BIN}" code/manual_id_edit.py --input "${csv}" --output "${out}" --id_a 0 --id_b 1 > "${log}" 2>&1 &
done

echo "Launched manual ID editors for match1_1..match1_6"
