#!/usr/bin/env bash
# segmodel — system-wide salient-object segmentation daemon. Runs the Python
# server in its venv with one model warm on the GPU. Backend (cuda|vulkan) is
# read from backend.txt (written by build.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "$HERE/config.env" ] && . "$HERE/config.env"

export SG_HOST="${SG_HOST:-127.0.0.1}"
export SG_PORT="${SG_PORT:-48470}"
export SG_BACKEND="${SG_BACKEND:-$( [ -f "$HERE/backend.txt" ] && cat "$HERE/backend.txt" || echo cuda )}"

# Device placement (gpu|ram) — written by the model-manager dock, read on launch.
# "ram" forces the CPU ONNX Runtime provider (BiRefNet runs on CPU + system RAM).
DEVICE_FILE="${XDG_CACHE_HOME:-$HOME/.cache}/modelmanager/segmodel.device"
SG_DEVICE="$( [ -f "$DEVICE_FILE" ] && cat "$DEVICE_FILE" || echo gpu )"
[ "$SG_DEVICE" = ram ] && export SG_BACKEND=cpu

if [ "$SG_BACKEND" = vulkan ]; then
    DEFAULT_MODEL="$HERE/models/u2net"            # loads u2net.param/.bin
else
    DEFAULT_MODEL="$HERE/models/birefnet_lite.onnx"   # cuda + cpu both use the ONNX model
fi
export SG_MODEL="${SG_MODEL:-$DEFAULT_MODEL}"

[ -x "$HERE/venv/bin/python" ] || { echo "venv missing — run ./build.sh [$SG_BACKEND]" >&2; exit 1; }

echo "segmodel: ${SG_MODEL##*/} on $SG_HOST:$SG_PORT (backend=$SG_BACKEND)" >&2
exec "$HERE/venv/bin/python" "$HERE/src/server.py"
