#!/usr/bin/env bash
# Fetch segmodel models.
#   ./fetch-model.sh cuda     # BiRefNet_lite ONNX (default backend)
#   ./fetch-model.sh vulkan   # U2Net ONNX -> ncnn (.param/.bin) via pnnx
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/models"; mkdir -p "$DEST"
WHICH="${1:-cuda}"

fetch_cuda() {
    local url="https://huggingface.co/onnx-community/BiRefNet_lite/resolve/main/onnx/model.onnx"
    echo "fetching BiRefNet_lite ONNX (~214 MB) …" >&2
    curl -sL --retry 3 -o "$DEST/birefnet_lite.onnx" "$url"
}

fetch_vulkan() {
    # U2Net is conv-only, so it converts to ncnn cleanly (unlike BiRefNet's Swin).
    local onnx="$DEST/u2net.onnx"
    echo "fetching U2Net ONNX …" >&2
    curl -sL --retry 3 -o "$onnx" \
        "https://huggingface.co/tomjackson2023/rembg/resolve/main/u2net.onnx"
    echo "converting U2Net ONNX -> ncnn via pnnx …" >&2
    [ -x "$HERE/venv/bin/pnnx" ] || "$HERE/venv/bin/pip" install -q pnnx
    ( cd "$DEST" && "$HERE/venv/bin/pnnx" u2net.onnx 'inputshape=[1,3,320,320]' )
    # pnnx writes u2net.ncnn.param/.bin — normalise to the names the engine loads.
    mv -f "$DEST/u2net.ncnn.param" "$DEST/u2net.param"
    mv -f "$DEST/u2net.ncnn.bin"   "$DEST/u2net.bin"
}

case "$WHICH" in
    cuda)   fetch_cuda ;;
    vulkan) fetch_vulkan ;;
    all)    fetch_cuda; fetch_vulkan ;;
    *) echo "usage: $0 [cuda|vulkan|all]" >&2; exit 1 ;;
esac
echo "done -> $DEST" >&2
