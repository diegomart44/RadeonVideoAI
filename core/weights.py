"""
AI Model Weights Manager.
Downloads and caches official pretrained Real-ESRGAN checkpoints
(BSD-3-Clause / Apache-2.0 licensed, released by the Real-ESRGAN authors)
required to perform genuine AI super-resolution, denoising and detail
reconstruction. Models are fetched once and cached under models/weights/.
"""

import os
import logging
import requests
from typing import Callable, Optional

logger = logging.getLogger("RadeonVideoAI.Weights")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "weights")
ONNX_CACHE_DIR = os.path.join(WEIGHTS_DIR, "onnx_cache")

# Official release URLs published by xinntao/Real-ESRGAN.
MODEL_URLS = {
    "RealESRGAN_x4plus.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "realesr-general-x4v3.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "realesr-general-wdn-x4v3.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "BSRGAN.pth":
        "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth",
    "4x-UltraSharp.pth":
        "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
}

ProgressCB = Optional[Callable[[int, int, str], None]]


def get_weight_path(filename: str, progress_cb: ProgressCB = None) -> str:
    """
    Returns the local path to a pretrained checkpoint, downloading it from the
    official Real-ESRGAN GitHub release the first time it is needed.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dest = os.path.join(WEIGHTS_DIR, filename)

    if os.path.isfile(dest) and os.path.getsize(dest) > 1024 * 100:
        return dest

    url = MODEL_URLS.get(filename)
    if not url:
        raise ValueError(f"No hay URL de descarga registrada para el modelo: {filename}")

    logger.info(f"Descargando pesos de IA (una sola vez): {filename} ...")
    tmp_dest = dest + ".part"
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp_dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total, filename)
        os.replace(tmp_dest, dest)
    except Exception:
        if os.path.isfile(tmp_dest):
            try:
                os.remove(tmp_dest)
            except OSError:
                pass
        raise

    logger.info(f"Descarga completada: {filename} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return dest


def ensure_onnx_cache_dir() -> str:
    os.makedirs(ONNX_CACHE_DIR, exist_ok=True)
    return ONNX_CACHE_DIR
