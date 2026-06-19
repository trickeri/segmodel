#!/usr/bin/env bash
# Build segmodel's Python venv for a backend.
#   ./build.sh          # cuda (BiRefNet via onnxruntime-gpu, default)
#   ./build.sh vulkan   # U2Net via ncnn-python
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${1:-cuda}"
[ "$BACKEND" = cuda ] || [ "$BACKEND" = vulkan ] || { echo "usage: $0 [cuda|vulkan]" >&2; exit 1; }

python3 -m venv "$HERE/venv"
PIP="$HERE/venv/bin/pip"
"$PIP" install -q --upgrade pip
"$PIP" install -q numpy opencv-python-headless
if [ "$BACKEND" = cuda ]; then
    "$PIP" install -q onnxruntime-gpu
else
    "$PIP" install -q ncnn
fi
echo "$BACKEND" > "$HERE/backend.txt"
echo "segmodel venv ready ($BACKEND)" >&2
