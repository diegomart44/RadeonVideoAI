"""
Real AI Frame Interpolation — rife-ncnn-vulkan (same technology Flowframes uses).

Deliberately separate from core/inference.py (restoration) and
core/generative_models.py (diffusion recreation): this model increases
frame rate by synthesizing new in-between frames from real optical-flow
estimation, a third distinct technology with its own tab/engine so a change
to either of the other two can never affect it.

Backend: rife-ncnn-vulkan (github.com/nihui/rife-ncnn-vulkan, MIT license),
the exact same ncnn+Vulkan native binary Flowframes uses internally — chosen
over the ONNX Runtime DirectML path tried first because, measured head to
head on this machine, it was consistently faster (native FP16 "just works"
via ncnn, where converting the ONNX RIFE graph to FP16 ran into repeated
internal type-compatibility errors and, once finally loading, took minutes
to compile on DirectML instead of seconds — not something worth shipping).

Only the rife-v4.6 model + the exe + its one runtime DLL are bundled, not
the full ~430MB official release (which ships every RIFE version back to
1.2) — a slim ~12MB zip redistributed from the official release (MIT allows
this; LICENSE-rife-ncnn-vulkan is included in the zip) is downloaded once
and cached under models/weights/rife_vulkan/.

The binary only exposes a folder-batch interface (-i indir -o outdir),
always doubling frame count (N frames -> 2N: N originals + N-1 midpoints,
plus one harmless duplicate of the final frame baked into how the tool
closes out a batch — see core/interpolation_engine.py for how that
duplicate, and the one at a resumed segment's shared boundary frame, are
trimmed). x4 is two chained 2x passes, exactly like Flowframes does it
(confirmed against a real Flowframes run's own log message on this
project).
"""

import os
import logging
import zipfile
import subprocess
from typing import Optional, Callable

import requests

logger = logging.getLogger("RadeonVideoAI.Interpolation")

_NO_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIFE_VULKAN_DIR = os.path.join(BASE_DIR, "models", "weights", "rife_vulkan")
RIFE_VULKAN_ZIP_URL = (
    "https://github.com/diegomart44/RadeonVideoAI/releases/download/"
    "rife-ncnn-vulkan-v4.6/rife-ncnn-vulkan-slim-v4.6.zip"
)

ProgressCB = Optional[Callable[[str], None]]


def _exe_path() -> str:
    return os.path.join(RIFE_VULKAN_DIR, "rife-ncnn-vulkan.exe")


def _model_path() -> str:
    return os.path.join(RIFE_VULKAN_DIR, "rife-v4.6")


def ensure_rife_vulkan(progress_cb: ProgressCB = None) -> str:
    """Downloads and caches the rife-ncnn-vulkan binary + rife-v4.6 model on
    first use. Returns the executable path; the model directory is a fixed
    sibling (`rife-v4.6/`) found via `_model_path()`."""
    exe_path = _exe_path()
    model_dir = _model_path()
    if os.path.isfile(exe_path) and os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "flownet.param")):
        return exe_path

    if progress_cb:
        progress_cb("Preparando motor de interpolación (rife-ncnn-vulkan, descarga única ~12 MB)...")

    os.makedirs(RIFE_VULKAN_DIR, exist_ok=True)
    zip_path = os.path.join(RIFE_VULKAN_DIR, "_download.zip.part")
    try:
        with requests.get(RIFE_VULKAN_ZIP_URL, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total:
                        pct = int(downloaded / total * 100)
                        progress_cb(f"Descargando motor de interpolación: {pct}%")

        if progress_cb:
            progress_cb("Extrayendo motor de interpolación...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RIFE_VULKAN_DIR)
    finally:
        if os.path.isfile(zip_path):
            os.remove(zip_path)

    if not os.path.isfile(exe_path):
        raise RuntimeError(f"La descarga del motor de interpolación no contenía {exe_path}")

    logger.info(f"rife-ncnn-vulkan listo en {RIFE_VULKAN_DIR}")
    return exe_path


class RifeVulkanModel:
    """Thin wrapper around the rife-ncnn-vulkan executable: runs one 2x
    batch pass over a folder of PNG frames. Vulkan GPU selection is left on
    "auto" (no -g flag) — confirmed in testing to correctly pick the
    system's discrete GPU on its own, and Vulkan device indices don't
    necessarily match DirectML's, so there's nothing reliable to map from
    core/amd_backend.py's own device detection anyway."""

    def __init__(self, progress_cb: ProgressCB = None):
        self.exe_path = ensure_rife_vulkan(progress_cb)
        self.model_dir = _model_path()
        if progress_cb:
            progress_cb(f"Motor de interpolación listo — backend: {self.active_backend}")

    @property
    def active_backend(self) -> str:
        return "ncnn-vulkan (GPU) — RIFE v4.6 (misma tecnología que Flowframes)"

    def run_2x_pass(self, input_dir: str, output_dir: str):
        """Doubles the frame count of the images in input_dir, writing
        results to output_dir (must already exist). Raises on failure."""
        os.makedirs(output_dir, exist_ok=True)
        cmd = [self.exe_path, "-i", input_dir, "-o", output_dir, "-m", self.model_dir]
        res = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW_FLAGS)
        if res.returncode != 0:
            raise RuntimeError(f"rife-ncnn-vulkan falló: {res.stderr.strip() or res.stdout.strip()}")


def create_interpolation_model(progress_cb: ProgressCB = None, **_ignored) -> RifeVulkanModel:
    return RifeVulkanModel(progress_cb=progress_cb)
