"""
Real AI Frame Interpolation (RIFE 4.9 via ONNX Runtime DirectML).

Deliberately separate from core/inference.py (restoration) and
core/generative_models.py (diffusion recreation): this model increases
frame rate by synthesizing new in-between frames from real optical-flow
estimation, a third distinct technology with its own tab/engine so a change
to either of the other two can never affect it.

Model: RIFE 4.9 (hzwer/Practical-RIFE, MIT license), pre-exported by the
community to a single self-contained ONNX graph (github.com/hzwer/
Practical-RIFE; ONNX export via huggingface.co/edgetools/rife). Confirmed
running on DirectML on AMD Radeon at ~46ms per interpolated frame at 720p.

Backend choice — DirectML, not Vulkan: real Flowframes/rife-ncnn-vulkan runs
RIFE through ncnn+Vulkan, a completely separate native inference stack from
everything else in this app (PyTorch/ONNX Runtime). This ONNX export lets
frame interpolation reuse the exact same GPU path already validated for the
other two features — DirectML, which is already vendor-agnostic (NVIDIA/
AMD/Intel) per core/amd_backend.py — with no new native dependency and no
separate Vulkan SDK/binary to bundle and maintain.

The model needs height/width padded to a multiple of 32; callers get back
a frame cropped to the original size, so padding is entirely invisible to
core/interpolation_engine.py.
"""

import logging
from typing import Optional, Callable, List

import numpy as np
import onnxruntime as ort

from . import weights as weights_mod

logger = logging.getLogger("RadeonVideoAI.Interpolation")

RIFE_PAD_MULTIPLE = 32
ProgressCB = Optional[Callable[[str], None]]


def _pad_to_multiple(h: int, w: int, multiple: int = RIFE_PAD_MULTIPLE):
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return pad_h, pad_w


class RifeInterpolationModel:
    """
    Callable frame-interpolation model: given two consecutive RGB frames
    (H, W, 3) uint8 and a list of timesteps in (0, 1), returns one
    synthesized in-between frame per timestep, each (H, W, 3) uint8 at the
    SAME resolution as the inputs.
    """

    def __init__(self, providers: Optional[list] = None, progress_cb: ProgressCB = None):
        if progress_cb:
            progress_cb("Preparando modelo de interpolación (RIFE 4.9, descarga única ~20 MB)...")

        def _dl_progress(downloaded, total, filename):
            if progress_cb and total:
                pct = int(downloaded / total * 100)
                progress_cb(f"Descargando modelo de interpolación: {pct}%")

        model_path = weights_mod.get_weight_path("rife49.onnx", progress_cb=_dl_progress)

        chosen_providers = list(providers) if providers else ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in chosen_providers:
            chosen_providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=chosen_providers)
        self._active_providers = self.session.get_providers()
        logger.info(f"Modelo RIFE listo. Proveedor activo: {self._active_providers}")

        if progress_cb:
            progress_cb(f"Modelo de interpolación listo — backend: {self.active_backend}")

    @property
    def active_backend(self) -> str:
        if "DmlExecutionProvider" in self._active_providers:
            return "ONNX Runtime DirectML (GPU) — RIFE 4.9"
        return "ONNX Runtime CPU — RIFE 4.9"

    def interpolate(self, frame_a: np.ndarray, frame_b: np.ndarray, timesteps: List[float]) -> List[np.ndarray]:
        """frame_a/frame_b: (H, W, 3) uint8 RGB. Returns len(timesteps) frames, same (H, W, 3) uint8."""
        h, w = frame_a.shape[:2]
        pad_h, pad_w = _pad_to_multiple(h, w)

        def to_tensor(frame: np.ndarray) -> np.ndarray:
            t = frame.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3,H,W)
            if pad_h or pad_w:
                t = np.pad(t, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            return t[np.newaxis, ...]  # (1,3,H',W')

        img0 = to_tensor(frame_a)
        img1 = to_tensor(frame_b)

        outputs = []
        for t in timesteps:
            ts = np.array([t], dtype=np.float32)
            result = self.session.run(None, {"img0": img0, "img1": img1, "timestep": ts})[0]
            frame = result[0].transpose(1, 2, 0)  # (H',W',3)
            frame = frame[:h, :w, :]  # crop off padding
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            outputs.append(frame)
        return outputs


def create_interpolation_model(providers: Optional[list] = None, progress_cb: ProgressCB = None) -> RifeInterpolationModel:
    return RifeInterpolationModel(providers=providers, progress_cb=progress_cb)


def timesteps_for_multiplier(multiplier: int) -> List[float]:
    """x2 inserts 1 frame at the midpoint; x4 inserts 3 frames at even quarters."""
    if multiplier == 4:
        return [0.25, 0.5, 0.75]
    return [0.5]
