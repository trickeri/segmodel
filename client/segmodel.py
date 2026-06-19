"""Reference client for the segmodel daemon (stdlib only).

    from segmodel import matte
    rgba_png = matte(open("frame.png", "rb").read())            # RGBA cutout
    alpha_png = matte(open("frame.png", "rb").read(), "alpha")   # grayscale mask

Plus control helpers (status/load/unload/park/activate). Posts the raw image as
the request body (segmodel also accepts multipart "image").
"""
import os
import json
import urllib.request

HOST = os.environ.get("SG_HOST", "127.0.0.1")
PORT = os.environ.get("SG_PORT", "48470")
BASE = os.environ.get("SG_HTTP_URL", f"http://{HOST}:{PORT}")


def matte(image_bytes: bytes, fmt: str = "cutout") -> bytes:
    req = urllib.request.Request(
        f"{BASE}/matte?format={fmt}", data=image_bytes,
        headers={"Content-Type": "image/png"}, method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _post(path: str) -> dict:
    req = urllib.request.Request(f"{BASE}/{path}", data=b"", method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def status() -> dict:
    with urllib.request.urlopen(f"{BASE}/status") as resp:
        return json.loads(resp.read().decode())


def load():     return _post("load")
def activate(): return _post("activate")
def unload():   return _post("unload")
def park():     return _post("park")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(json.dumps(status(), indent=2))
    else:
        fmt = sys.argv[2] if len(sys.argv) > 2 else "cutout"
        out = sys.argv[3] if len(sys.argv) > 3 else "cutout.png"
        with open(sys.argv[1], "rb") as f:
            png = matte(f.read(), fmt)
        with open(out, "wb") as f:
            f.write(png)
        print(f"wrote {out} ({len(png)} bytes, format={fmt})")
