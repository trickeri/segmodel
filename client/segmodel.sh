#!/usr/bin/env bash
# Minimal shell client for segmodel.
#   segmodel.sh <image> [out.png] [cutout|alpha]
#   segmodel.sh --status | --load | --unload | --park | --activate
set -euo pipefail
BASE="${SG_HTTP_URL:-http://${SG_HOST:-127.0.0.1}:${SG_PORT:-48470}}"

case "${1:-}" in
  --status)  curl -s "$BASE/status"; echo ;;
  --load|--unload|--park|--activate)
             curl -s -X POST -d '' "$BASE/${1#--}"; echo ;;
  "")        echo "usage: $0 <image> [out.png] [cutout|alpha] | --status|--load|--unload|--park|--activate" >&2; exit 1 ;;
  *)
    img="$1"; out="${2:-cutout.png}"; fmt="${3:-cutout}"
    curl -s --data-binary "@${img}" -H "Content-Type: image/png" "$BASE/matte?format=${fmt}" -o "$out"
    echo "wrote $out (format=$fmt)" ;;
esac
