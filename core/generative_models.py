"""
Real Generative AI Recreation (Stable Diffusion img2img via ONNX Runtime DirectML).

Deliberately separate from core/inference.py (the Real-ESRGAN restoration
pipeline). This module hallucinates plausible new detail guided by the
original frame, rather than faithfully restoring what's already there — a
different technology with different trade-offs (much slower, potential
frame-to-frame flicker, no resolution change), which is exactly why the app
keeps it as an independent tab/engine instead of a mode of the upscaler.

Model: stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML Open RAIL-M
license — permissive for personal/most commercial use, with acceptable-use
restrictions; see https://huggingface.co/spaces/CompVis/stable-diffusion-license).
Runs through optimum's ORTPipelineForImage2Image, which exports the
text encoder / UNet / VAE to ONNX and executes them via onnxruntime with the
DirectML execution provider — confirmed working on AMD Radeon (RX 9060 XT)
at roughly 0.14-0.16s per diffusion step for a 512x512 tile.

NOTE on the `optimum` onnxruntime-directml detection patch below: optimum's
availability check only recognizes a fixed allowlist of onnxruntime package
names (onnxruntime-gpu, onnxruntime-rocm, ...) and is missing
"onnxruntime-directml" — a real upstream gap, not a bug in this app. Without
the patch, `optimum.onnxruntime` raises "onnx and onnxruntime packages are
necessary..." even though the import and DmlExecutionProvider work fine.
"""

import os
import logging
from typing import Optional, Callable

import numpy as np
import torch
from PIL import Image

from . import weights as weights_mod

logger = logging.getLogger("RadeonVideoAI.Generative")

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
NATIVE_TILE_SIZE = 512  # SD 1.5's native resolution; VAE requires multiples of 8
DEFAULT_PROMPT = "high quality, detailed, sharp focus, photorealistic"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed"

ProgressCB = Optional[Callable[[str], None]]


def _patch_optimum_onnxruntime_detection():
    """See module docstring: optimum's onnxruntime-distribution allowlist is
    missing 'onnxruntime-directml'. Patch the cached availability flag."""
    import optimum.utils.import_utils as _oiu
    _oiu._onnxruntime_available = True


def _onnx_cache_dir() -> str:
    cache_dir = os.path.join(weights_mod.ensure_onnx_cache_dir(), "sd15_img2img")
    return cache_dir


def _pad_to_multiple_of_8(img: Image.Image):
    """The VAE downsamples by 8x; pad any tile whose dimensions aren't a
    multiple of 8 (only possible for the single-tile case on very small
    videos) using edge replication, and remember the crop to undo it."""
    w, h = img.size
    pad_w = (8 - w % 8) % 8
    pad_h = (8 - h % 8) % 8
    if pad_w == 0 and pad_h == 0:
        return img, (0, 0, w, h)
    arr = np.array(img)
    arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return Image.fromarray(arr), (0, 0, w, h)


class GenerativeUpscaleModel:
    """
    Callable generative recreation model: takes a tile tensor, returns a
    tile tensor of the SAME size (no scale factor — this is not an
    upscaler). Mirrors the calling convention of core.inference.UpscaleModel
    ((1,3,H,W) float32 in [0,1] in, same shape out) so the engine loop looks
    the same, without sharing any code with the restoration pipeline.
    """

    def __init__(self, steps: int = 15, strength: float = 0.4,
                 prompt: str = DEFAULT_PROMPT, negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
                 providers: Optional[list] = None, progress_cb: ProgressCB = None):
        _patch_optimum_onnxruntime_detection()
        from optimum.onnxruntime import ORTPipelineForImage2Image

        self.steps = steps
        self.strength = strength
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        provider = "DmlExecutionProvider" if (providers and "DmlExecutionProvider" in providers) else "CPUExecutionProvider"

        cache_dir = _onnx_cache_dir()
        if os.path.isdir(cache_dir):
            if progress_cb:
                progress_cb("Cargando modelo generativo (ya exportado a ONNX)...")
            self.pipe = ORTPipelineForImage2Image.from_pretrained(cache_dir, provider=provider)
        else:
            if progress_cb:
                progress_cb("Preparando modelo generativo (descarga + exportación ONNX única, ~3-5 min)...")
            self.pipe = ORTPipelineForImage2Image.from_pretrained(
                MODEL_ID, export=True, provider=provider, safety_checker=None
            )
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            self.pipe.save_pretrained(cache_dir)
            if progress_cb:
                progress_cb("Modelo generativo exportado y cacheado.")

        self._active_providers = self.pipe.unet.session.get_providers()
        logger.info(f"Modelo generativo listo. Proveedor activo: {self._active_providers}")

    @property
    def active_backend(self) -> str:
        if "DmlExecutionProvider" in self._active_providers:
            return "ONNX Runtime DirectML (GPU AMD) — Generativo"
        return "ONNX Runtime CPU — Generativo"

    def __call__(self, tile_tensor: torch.Tensor, force_cpu: bool = False, **_ignored) -> torch.Tensor:
        arr = (tile_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        img = Image.fromarray(arr)
        padded_img, (cx0, cy0, cw, ch) = _pad_to_multiple_of_8(img)

        result = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=padded_img,
            strength=self.strength,
            num_inference_steps=self.steps,
            guidance_scale=7.0,
        ).images[0]

        out_arr = np.array(result)[cy0:cy0 + ch, cx0:cx0 + cw, :].astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_arr).permute(2, 0, 1).unsqueeze(0)
        return torch.clamp(out_tensor, 0.0, 1.0)


def create_generative_model(steps: int = 15, strength: float = 0.4,
                             providers: Optional[list] = None,
                             progress_cb: ProgressCB = None) -> GenerativeUpscaleModel:
    return GenerativeUpscaleModel(steps=steps, strength=strength, providers=providers, progress_cb=progress_cb)
