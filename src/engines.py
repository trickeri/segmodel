"""segmodel inference engines — salient-object segmentation, pluggable backend.

- BiRefNetOrt  (BACKEND=cuda)   — BiRefNet via ONNX Runtime + CUDA. Default,
  best quality, arbitrary subjects. Validated.
- U2NetNcnn    (BACKEND=vulkan) — U2Net via ncnn-Vulkan. GPU-agnostic fallback.
  Implemented; needs a converted U2Net ncnn model (see fetch-model.sh vulkan).

Both expose the same API the server uses:
    load() / unload() / loaded() -> bool
    matte(bgr_uint8_HxWx3) -> (fgr_bgr_uint8, alpha_uint8)   # fgr is the source
    .backend  (class attr)
"""
from __future__ import annotations
import threading
import numpy as np
import cv2

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class BiRefNetOrt:
    backend = "cuda"

    def __init__(self, model_path: str, size: int = 1024, providers=None, backend=None):
        self.model_path = model_path
        self.size = size
        # Device placement: GPU (CUDA, falling back to CPU) by default, or CPU-only
        # when the model-manager dock moves this model to RAM (backend="cpu").
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if backend:
            self.backend = backend
        self._sess = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        with self._lock:
            if self._sess is not None:
                return True
            import onnxruntime as ort
            self._sess = ort.InferenceSession(
                self.model_path,
                providers=self.providers,
            )
            return True

    def unload(self):
        with self._lock:
            self._sess = None

    def loaded(self) -> bool:
        return self._sess is not None

    def matte(self, bgr: np.ndarray):
        if not self.loaded():
            self.load()
        H, W = bgr.shape[:2]
        x = cv2.resize(bgr, (self.size, self.size))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        with self._lock:
            name = self._sess.get_inputs()[0].name
            out = self._sess.run(None, {name: x})[0]
        m = out[0, 0] if out.ndim == 4 else np.squeeze(out)
        if m.min() < 0 or m.max() > 1:
            m = 1.0 / (1.0 + np.exp(-m))
        alpha = cv2.resize((m * 255).astype(np.uint8), (W, H))
        return bgr, alpha   # BiRefNet is a mask model; foreground = source pixels


class U2NetNcnn:
    backend = "vulkan"

    def __init__(self, model_prefix: str, size: int = 320):
        self.model_prefix = model_prefix   # loads <prefix>.param/.bin
        self.size = size
        self._net = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        with self._lock:
            if self._net is not None:
                return True
            import ncnn
            net = ncnn.Net()
            net.opt.use_vulkan_compute = True
            if net.load_param(self.model_prefix + ".param") != 0:
                return False
            if net.load_model(self.model_prefix + ".bin") != 0:
                return False
            self._net = net
            return True

    def unload(self):
        with self._lock:
            self._net = None

    def loaded(self) -> bool:
        return self._net is not None

    def matte(self, bgr: np.ndarray):
        if not self.loaded() and not self.load():
            raise RuntimeError(
                "U2Net ncnn model not loaded — run ./fetch-model.sh vulkan to "
                "provision it")
        import ncnn
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mat_in = ncnn.Mat.from_pixels_resize(
            rgb, ncnn.Mat.PixelType.PIXEL_RGB, W, H, self.size, self.size)
        # U2Net: scale to [0,1] then ImageNet-normalise.
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        norm = [1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)]
        mat_in.substract_mean_normalize(mean, norm)
        with self._lock:
            ex = self._net.create_extractor()
            ex.input("in0", mat_in)           # pnnx-converted U2Net input blob
            _, out = ex.extract("out0")        # d0 — the fused saliency map
        m = np.array(out)[0]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        alpha = cv2.resize((m * 255).astype(np.uint8), (W, H))
        return bgr, alpha


def make_engine(backend: str, model_path: str):
    if backend == "cuda":
        return BiRefNetOrt(model_path)
    if backend == "cpu":
        # CPU + system-RAM placement (model-manager "RAM" mode): same BiRefNet ONNX,
        # CPU execution provider only.
        return BiRefNetOrt(model_path, providers=["CPUExecutionProvider"], backend="cpu")
    if backend == "vulkan":
        return U2NetNcnn(model_path)
    raise ValueError(f"unknown backend: {backend!r} (cuda|cpu|vulkan)")
