#!/bin/bash
# RTX quad: 4K -> 4 quarters, each one upscaled with RTX Video Super Resolution to 4K (8K mosaic) -> 4K,
# then a second RTX pass (HIGHBITRATE_ULTRA). Horizontal 3840x2160 input only.
# Usage: RTX_PYTHON=<python with torch + nvvfx> [RTX_TMP=<folder>] bash rtx_quad.sh <input_4k.mp4> <output.mp4>
set -e
P=${RTX_PYTHON:?set RTX_PYTHON to a python that has torch and nvvfx}
R="$(cd "$(dirname "$0")" && pwd)/rtx_upscale.py"
G=$1; C=$2; T=${RTX_TMP:-${TMPDIR:-/tmp}}/rtxq_$$; i=0
trap 'echo "rtx_quad: stopped at line $LINENO (exit $?). Temporary files are kept: ${T}_*" >&2' ERR
up() {  # up <in> <out> <w> <h> [quality]: one RTX pass; on failure show the last lines of its log
  if ! "$P" "$R" "$@" </dev/null >"${T}_log.txt" 2>&1 || [ ! -s "$2" ]; then
    tail -20 "${T}_log.txt" >&2; echo "rtx_quad: RTX upscale failed for $1" >&2; exit 1
  fi
  tail -1 "${T}_log.txt"
}
for xy in 0:0 1920:0 0:1080 1920:1080; do i=$((i+1))
  ffmpeg -nostdin -y -v error -i "$G" -vf crop=1920:1080:$xy -an -c:v libx264 -crf 10 "${T}_part$i.mp4"
  up "${T}_part$i.mp4" "${T}_big$i.mp4" 3840 2160; done
ffmpeg -nostdin -y -v error -i "${T}_big1.mp4" -i "${T}_big2.mp4" -i "${T}_big3.mp4" -i "${T}_big4.mp4" -i "$G" -filter_complex "[0][1]hstack[a];[2][3]hstack[b];[a][b]vstack,scale=3840:2160:flags=lanczos[v]" -map "[v]" -map 4:a? -c:v libx264 -crf 10 -c:a copy "${T}_joined.mp4"
up "${T}_joined.mp4" "$C" 3840 2160 HIGHBITRATE_ULTRA
rm -f "${T}"_*
