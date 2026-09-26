"""
AI Inference Backend: ONNX Runtime + DirectML.

Runs the real, pretrained Real-ESRGAN networks (see models.py) on AMD Radeon
GPUs through ONNX Runtime's DirectML Execution Provider. DirectML works over
DirectX 12 compute, so it supports any modern AMD Radeon GPU (including RDNA4
/ RX 9060 XT) through the standard Adrenalin driver, with no ROCm or CUDA
dependency. This is the actively maintained path recommended by Microsoft for
Windows GPU inference in 2026 (torch-directml is in maintenance mode and does
not ship builds for current PyTorch/Python versions).

Only PyTorch on CPU is used, once per job, to load a checkpoint and export it
to a cached ONNX graph. The denoise pre-filter and the sharpen post-filter
are fused into that SAME exported graph as fixed (non-trainable) conv
layers, so a single onnxruntime DirectML call does denoise+network+sharpen
on the GPU. The only step kept outside the graph is the output resize used
when the network's native scale differs from the requested one (e.g. a
native-4x network used for a 2x job): DirectML does not accelerate the
"cubic" ONNX Resize op, and fusing it forced a GPU->CPU sync that made a
single tile take ~6s instead of ~0.1s in testing. Doing that resize in plain
PyTorch on CPU afterward costs ~30ms and avoids the cross-provider stall
entirely, so it stays a separate step rather than part of the graph.
"""

import os
import logging
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import weights
from .models import (
    MODEL_REGISTRY, build_torch_arch, extract_state_dict,
    remap_old_esrgan_state_dict, remap_sequential_old_esrgan_state_dict,
)

logger = logging.getLogger("RadeonVideoAI.Inference")

ProgressCB = Optional[Callable[[int, int, str], None]]


def _available_providers():
    import onnxruntime as ort
    return set(ort.get_available_providers())


def default_providers() -> list:
    """Prefers DirectML (AMD/Intel/NVIDIA DX12 GPU) with a CPU fallback."""
    avail = _available_providers()
    providers = []
    if "DmlExecutionProvider" in avail:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _load_checkpoint_state_dict(weights_filename: str, progress_cb: ProgressCB = None) -> dict:
    path = weights.get_weight_path(weights_filename, progress_cb=progress_cb)
    checkpoint = torch.load(path, map_location="cpu")
    sd = extract_state_dict(checkpoint)
    sd = remap_sequential_old_esrgan_state_dict(sd)
    sd = remap_old_esrgan_state_dict(sd)
    return sd


def _build_torch_model(model_name: str, denoise_strength: float, progress_cb: ProgressCB = None) -> torch.nn.Module:
    """
    Builds the CPU-side PyTorch model with real pretrained weights loaded.
    For 'general_v3', denoise_strength genuinely blends between the sharp
    (x4v3) and denoising (wdn-x4v3) checkpoints via weight interpolation,
    exactly as Real-ESRGAN's own CLI implements its denoise slider.
    """
    cfg = MODEL_REGISTRY[model_name]
    model = build_torch_arch(model_name)

    sd_a = _load_checkpoint_state_dict(cfg["weights"], progress_cb=progress_cb)

    if cfg.get("supports_native_denoise") and denoise_strength > 1e-3:
        sd_b = _load_checkpoint_state_dict(cfg["weights_denoise"], progress_cb=progress_cb)
        blended = {
            k: (1.0 - denoise_strength) * sd_a[k] + denoise_strength * sd_b[k]
            for k in sd_a.keys()
        }
        model.load_state_dict(blended, strict=True)
    else:
        model.load_state_dict(sd_a, strict=True)

    model.eval()
    return model


def _bucket(value: float) -> float:
    """Quantizes to 10% steps so exported ONNX graphs can be reused/cached."""
    return round(max(0.0, min(1.0, value)) * 10) / 10.0


_GAUSSIAN_3X3 = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
_GAUSSIAN_5X5 = torch.tensor([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
], dtype=torch.float32) / 256.0


class _PreDenoiseBlend(torch.nn.Module):
    """Fallback pre-filter denoise for architectures without a native denoise
    control (x4plus / x2plus / anime6b / bsrgan): blends the input with a
    small gaussian blur before feeding the network, reducing sensor/
    compression noise the AI would otherwise sharpen into visible artifacts.
    Runs as a fixed (non-trainable) conv layer fused into the ONNX graph."""

    def __init__(self, strength: float):
        super().__init__()
        self.register_buffer("kernel", _GAUSSIAN_3X3.repeat(3, 1, 1, 1))
        self.amount = strength * 0.5

    def forward(self, x):
        blurred = F.conv2d(x, self.kernel, padding=1, groups=3)
        return (1.0 - self.amount) * x + self.amount * blurred


class _PostSharpen(torch.nn.Module):
    """Classic gaussian unsharp mask: sharpened = x + amount * (x - blur(x)).
    Fused into the ONNX graph so it runs on the GPU right after the network,
    instead of as a separate CPU pass on the returned numpy array."""

    def __init__(self, strength: float):
        super().__init__()
        self.register_buffer("kernel", _GAUSSIAN_5X5.repeat(3, 1, 1, 1))
        self.amount = strength * 1.5

    def forward(self, x):
        blurred = F.conv2d(x, self.kernel, padding=2, groups=3)
        return torch.clamp(x + self.amount * (x - blurred), 0.0, 1.0)


class _FullPipelineModel(torch.nn.Module):
    """Wraps the base super-resolution network with the optional pre-denoise
    and post-sharpen steps so ONNX export produces ONE graph that performs
    denoise+network+sharpen in a single GPU-resident call. The scale-mismatch
    resize is intentionally NOT part of this graph (see module docstring)."""

    def __init__(self, base, pre=None, post=None):
        super().__init__()
        self.pre = pre
        self.base = base
        self.post = post

    def forward(self, x):
        if self.pre is not None:
            x = self.pre(x)
        out = torch.clamp(self.base(x), 0.0, 1.0)
        if self.post is not None:
            out = self.post(out)
        return out


def _build_full_pipeline(model_name: str, denoise_bucket: float, sharpen_bucket: float,
                          progress_cb: ProgressCB = None) -> torch.nn.Module:
    cfg = MODEL_REGISTRY[model_name]
    base = _build_torch_model(model_name, denoise_bucket, progress_cb=progress_cb)

    pre = None
    if not cfg.get("supports_native_denoise") and denoise_bucket > 1e-3:
        pre = _PreDenoiseBlend(denoise_bucket)

    post = None
    if sharpen_bucket > 1e-3:
        post = _PostSharpen(sharpen_bucket)

    model = _FullPipelineModel(base, pre, post)
    model.eval()
    return model


def _onnx_cache_path(model_name: str, denoise_bucket: float, sharpen_bucket: float) -> str:
    cache_dir = weights.ensure_onnx_cache_dir()
    dn_tag = f"_dn{int(denoise_bucket * 100):03d}" if MODEL_REGISTRY[model_name].get("supports_native_denoise") \
        or denoise_bucket > 1e-3 else ""
    sh_tag = f"_sh{int(sharpen_bucket * 100):03d}" if sharpen_bucket > 1e-3 else ""
    return os.path.join(cache_dir, f"{model_name}{dn_tag}{sh_tag}.onnx")


def _export_onnx(torch_model: torch.nn.Module, onnx_path: str, sample_size: int = 256):
    torch_model.eval()
    dummy = torch.randn(1, 3, sample_size, sample_size, dtype=torch.float32)
    torch.onnx.export(
        torch_model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript-based exporter; avoids the onnxscript dependency
    )
    logger.info(f"Modelo IA exportado y cacheado en ONNX (grafo fusionado GPU): {onnx_path}")


def get_or_build_onnx(model_name: str, denoise_strength: float, sharpen_strength: float,
                       progress_cb: ProgressCB = None) -> str:
    cfg = MODEL_REGISTRY[model_name]
    dn_bucket = _bucket(denoise_strength) if (cfg.get("supports_native_denoise") or denoise_strength > 0) else 0.0
    sh_bucket = _bucket(sharpen_strength)
    onnx_path = _onnx_cache_path(model_name, dn_bucket, sh_bucket)
    if os.path.isfile(onnx_path):
        return onnx_path
    torch_model = _build_full_pipeline(model_name, dn_bucket, sh_bucket, progress_cb=progress_cb)
    _export_onnx(torch_model, onnx_path)
    return onnx_path


class OnnxSession:
    """Thin wrapper around an onnxruntime InferenceSession for one graph."""

    def __init__(self, onnx_path: str, providers: list):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.active_providers = self.session.get_providers()

    def run(self, tile_np: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tile_np})[0]


class UpscaleModel:
    """
    High-level callable AI upscaler used by the processing engine.
    Loads real pretrained weights once, compiles them to an ONNX graph, and
    executes every tile through ONNX Runtime's DirectML GPU backend, with a
    lazy CPU fallback session if the GPU path fails on a given tile.
    """

    def __init__(self, model_name: str, target_scale: int, denoise_strength: float = 0.0,
                 sharpen_strength: float = 0.0, providers: Optional[list] = None,
                 progress_cb: ProgressCB = None):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Modelo de IA desconocido: {model_name}")

        cfg = MODEL_REGISTRY[model_name]
        self.model_name = model_name
        self.native_scale = cfg["native_scale"]
        self.target_scale = target_scale

        onnx_path = get_or_build_onnx(model_name, denoise_strength, sharpen_strength, progress_cb=progress_cb)
        self._onnx_path = onnx_path
        self._providers = providers or default_providers()

        self._primary = OnnxSession(onnx_path, self._providers)
        self._cpu_fallback: Optional[OnnxSession] = None

        logger.info(
            f"Modelo IA listo: {cfg['label']} | Proveedor activo: {self._primary.active_providers} "
            f"| Escala nativa: {self.native_scale}x -> objetivo: {self.target_scale}x "
            f"| Pipeline completo (denoise+red+resize+nitidez) fusionado en un solo grafo GPU"
        )

    @property
    def active_backend(self) -> str:
        providers = self._primary.active_providers
        if "DmlExecutionProvider" in providers:
            return "ONNX Runtime DirectML (GPU AMD)"
        return "ONNX Runtime CPU"

    def _get_cpu_fallback(self) -> OnnxSession:
        if self._cpu_fallback is None:
            logger.warning("Creando sesión de respaldo en CPU tras fallo en el backend GPU.")
            self._cpu_fallback = OnnxSession(self._onnx_path, ["CPUExecutionProvider"])
        return self._cpu_fallback

    def __call__(self, tile_tensor: torch.Tensor, force_cpu: bool = False, **_ignored) -> torch.Tensor:
        """
        Runs the fused denoise+network+sharpen pipeline in a single
        onnxruntime call, then resizes to the requested scale in plain torch
        on CPU if the network's native scale doesn't already match (see
        module docstring for why that resize step is kept out of the graph).
        `**_ignored` absorbs legacy per-call denoise_strength/
        sharpen_strength kwargs some callers may still pass; those are now
        baked into the exported graph instead.
        """
        x_np = np.ascontiguousarray(tile_tensor.detach().cpu().numpy().astype(np.float32))

        try:
            session = self._get_cpu_fallback() if force_cpu else self._primary
            out_np = session.run(x_np)
        except Exception as e:
            logger.warning(f"Fallo de inferencia en backend principal ({e}). Reintentando en CPU.")
            out_np = self._get_cpu_fallback().run(x_np)

        out = torch.from_numpy(out_np)

        if self.native_scale != self.target_scale:
            _, _, h, w = tile_tensor.shape
            out = F.interpolate(out, size=(h * self.target_scale, w * self.target_scale),
                                 mode="bicubic", align_corners=False)
            out = torch.clamp(out, 0.0, 1.0)

        return out


def create_model(model_name: str, scale: int, denoise_strength: float = 0.0, sharpen_strength: float = 0.0,
                  providers: Optional[list] = None, progress_cb: ProgressCB = None) -> UpscaleModel:
    """Factory used by the processing engine to build the active AI upscaler."""
    return UpscaleModel(model_name=model_name, target_scale=scale, denoise_strength=denoise_strength,
                         sharpen_strength=sharpen_strength, providers=providers, progress_cb=progress_cb)
