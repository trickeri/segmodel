# segmodel

System-wide **salient-object segmentation daemon**. One model loaded **once** on
the GPU and exposed over local HTTP, so any program can ask for a cutout of the
**main subject** — arbitrary objects (a car, a product, a prop), not just people.

Where **mattemodel** (RVM) mattes *humans*, segmodel handles **arbitrary salient
subjects**. Same service shape as the other `*model` daemons: systemd user
service, loopback HTTP, model-manager control API, **build-time backend choice**.

## Backends (pick at build time)

| `./build.sh <backend>` | Model | Runs on | Notes |
|---|---|---|---|
| **`cuda`** *(default)* | **BiRefNet** (ONNX Runtime) | NVIDIA | best quality; BiRefNet is a Swin transformer that only ORT runs well |
| **`vulkan`** | **U2Net** (ncnn) | any GPU | portable fallback; lighter, softer than BiRefNet |

> BiRefNet can't run on ncnn-Vulkan (its transformer ops aren't supported), which
> is exactly why backend lives in config: quality path = BiRefNet@cuda, portable
> path = U2Net@vulkan. Same HTTP/control API either way.

## Architecture

```
        ┌──────────── segmodel (python) ─────────────┐
        │  BiRefNet (cuda) | U2Net (vulkan)           │
        │  HTTP on 127.0.0.1:48470                    │
        │    POST /matte?format=cutout|alpha → PNG    │
        │    GET  /status                             │
        │    POST /load /activate /unload /park       │
        └───────────────────┬─────────────────────────┘
           nulpaint / Kdenlive clients POST images, get cutouts
```

## Install

```bash
cd ~/programming && git clone https://github.com/trickeri/segmodel.git
cd segmodel
./build.sh                 # cuda (default); or ./build.sh vulkan
./fetch-model.sh cuda      # or: vulkan
ln -s "$PWD/segmodel.service" ~/.config/systemd/user/segmodel.service
systemctl --user daemon-reload && systemctl --user enable --now segmodel.service
```

**Prerequisites** — `cuda`: an NVIDIA GPU + CUDA (onnxruntime-gpu, installed into
the venv). `vulkan`: a Vulkan GPU (ncnn-python). Both: Python 3, OpenCV/numpy
(installed into the venv by `build.sh`).

## Use

```bash
client/segmodel.sh frame.png cutout.png cutout
client/segmodel.sh --status
client/segmodel.sh --park            # free GPU; --activate to reload
python client/segmodel.py frame.png alpha mask.png
curl --data-binary @frame.png -H 'Content-Type: image/png' \
     'http://127.0.0.1:48470/matte?format=cutout' -o cutout.png
```

## License

MIT — see [LICENSE](LICENSE).
