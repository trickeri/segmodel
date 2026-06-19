"""segmodel — system-wide salient-object segmentation daemon.

One model warm on the GPU, exposed over local HTTP so any app (nulpaint for
Krita object-select, Kdenlive roto) can POST an image and get a cutout/alpha
back. Backend (cuda=BiRefNet | vulkan=U2Net) is chosen by env at launch.

Routes (same contract as mattemodel):
  POST /matte?format=cutout|alpha   multipart "image" OR raw image body -> PNG
  GET  /status                      JSON: model, backend, loaded, port
  POST /load /activate              ensure model resident; JSON status
  POST /unload /park                free the model; JSON status
"""
from __future__ import annotations
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engines import make_engine  # noqa: E402

HOST = os.environ.get("SG_HOST", "127.0.0.1")
PORT = int(os.environ.get("SG_PORT", "48470"))
BACKEND = os.environ.get("SG_BACKEND", "cuda")
MODEL = os.environ.get("SG_MODEL", "")

ENGINE = make_engine(BACKEND, MODEL)


def _extract_image(headers, body: bytes) -> bytes:
    """Return raw image bytes from a multipart 'image' field, or the raw body."""
    ctype = headers.get("Content-Type", "")
    if "multipart/form-data" in ctype and "boundary=" in ctype:
        boundary = ("--" + ctype.split("boundary=", 1)[1].strip()).encode()
        for part in body.split(boundary):
            if b"Content-Disposition" in part and b"name=\"image\"" in part:
                seg = part.split(b"\r\n\r\n", 1)
                if len(seg) == 2:
                    return seg[1].rstrip(b"\r\n")
        return b""
    return body  # raw image POST


def _status() -> dict:
    return {"service": "segmodel", "model": MODEL, "backend": BACKEND,
            "loaded": ENGINE.loaded(), "host": HOST, "port": PORT}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/status":
            self._json(_status())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path in ("/load", "/activate"):
            ENGINE.load(); return self._json(_status())
        if path in ("/unload", "/park"):
            ENGINE.unload(); return self._json(_status())
        if path != "/matte":
            return self._json({"error": "not found"}, 404)

        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n) if n else b""
        img_bytes = _extract_image(self.headers, body)
        if not img_bytes:
            return self._json({"error": "no image"}, 400)

        arr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return self._json({"error": "could not decode image"}, 400)

        fmt = "cutout"
        if "format=" in self.path:
            fmt = self.path.split("format=", 1)[1].split("&", 1)[0]

        try:
            fgr, alpha = ENGINE.matte(bgr)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        if fmt == "alpha":
            ok, png = cv2.imencode(".png", alpha)
        else:
            bgra = cv2.cvtColor(fgr, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha
            ok, png = cv2.imencode(".png", bgra)
        if not ok:
            return self._json({"error": "encode failed"}, 500)

        data = png.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    if not MODEL:
        print("segmodel: no model (set SG_MODEL)", file=sys.stderr); sys.exit(1)
    ENGINE.load()
    print(f"segmodel: {MODEL} on {HOST}:{PORT}  [backend={BACKEND}]", file=sys.stderr)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
