"""
Generative Video Recreation Engine.

Deliberately separate from core/engine.py (Real-ESRGAN restoration): this
engine does NOT rescale the video (output resolution == input resolution)
and does NOT share the model registry or ONNX graphs used by the
restoration pipeline. It exists so the two features can evolve and fail
independently, as requested — a bug or change in one must never affect the
other. It reuses the same battle-tested tiling/blending (MemoryManager) and
FFmpeg I/O (media_pipeline) primitives, since those are format/tiling
concerns unrelated to which model fills in each tile.
"""

import time
import logging
import threading
import numpy as np
import torch
from typing import Callable, Optional, Dict, Any

from .amd_backend import amd_hardware
from .memory_manager import MemoryManager
from .media_pipeline import FFmpegLocator, VideoMetadataReader, FFmpegFrameReader, FFmpegFrameWriter
from .generative_models import create_generative_model, NATIVE_TILE_SIZE

logger = logging.getLogger("RadeonVideoAI.GenerativeEngine")


class GenerativeVideoEngine:
    """
    Generative recreation pipeline: takes a video, regenerates every frame
    at the SAME resolution through a Stable Diffusion img2img model guided
    by the original frame. No scale factor — this is not an upscaler.
    """

    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self._stop_requested = False
        self._pause_event = threading.Event()
        self._pause_event.set()

    def cancel(self):
        self._stop_requested = True
        self._pause_event.set()

    def pause(self):
        self.is_paused = True
        self._pause_event.clear()

    def resume(self):
        self.is_paused = False
        self._pause_event.set()

    def process_video(
        self,
        input_path: str,
        output_path: str,
        steps: int = 15,
        strength: float = 0.4,
        overlap: int = 48,
        encoder: str = "hevc_amf",
        bitrate_mbps: int = 25,
        vram_safety: float = 0.90,
        progress_callback: Optional[Callable[[int, int, float, str, Dict[str, Any]], None]] = None,
        preview_callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        self.is_running = True
        self._stop_requested = False
        self.is_paused = False
        self._pause_event.set()

        reader: Optional[FFmpegFrameReader] = None
        writer: Optional[FFmpegFrameWriter] = None

        try:
            meta = VideoMetadataReader.probe(input_path)
            orig_w, orig_h = meta["width"], meta["height"]
            fps = meta["fps"]
            total_frames = meta["total_frames"]
            has_audio = meta["has_audio"]

            logger.info(f"Generando (recreación IA): {orig_w}x{orig_h} @ {fps:.2f}fps ({total_frames} fotogramas)")

            mem_mgr = MemoryManager(
                vram_total_mb=amd_hardware.vram_total_mb,
                max_vram_ratio=vram_safety,
                default_tile_size=NATIVE_TILE_SIZE,
                overlap=overlap
            )

            if status_callback:
                status_callback("Preparando modelo generativo (puede descargar/exportar la primera vez, varios minutos)...")

            def _status(msg):
                if status_callback:
                    status_callback(msg)

            model = create_generative_model(
                steps=steps,
                strength=strength,
                providers=amd_hardware.providers,
                progress_cb=_status
            )

            if status_callback:
                status_callback(f"Modelo generativo listo — backend: {model.active_backend}")

            if "amf" in encoder and not FFmpegLocator.check_amf_support():
                fallback_enc = "libx265" if "hevc" in encoder else "libx264"
                logger.warning(f"AMD AMF no detectado. Conmutando a {fallback_enc}.")
                encoder = fallback_enc

            reader = FFmpegFrameReader(input_path, orig_w, orig_h)
            writer = FFmpegFrameWriter(
                output_path=output_path,
                input_source_for_audio=input_path,
                width=orig_w,
                height=orig_h,
                fps=fps,
                has_audio=has_audio,
                encoder=encoder,
                bitrate_mbps=bitrate_mbps
            )

            frame_idx = 0
            start_time = time.time()
            last_fps_calc_time = start_time
            last_fps_frame_idx = 0
            current_fps = 0.0

            while not self._stop_requested:
                self._pause_event.wait()
                if self._stop_requested:
                    break

                frame = reader.read_frame()
                if frame is None:
                    break

                frame_idx += 1
                mem_mgr.check_vram_safety()

                t_frame = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                tiles = mem_mgr.split_into_tiles(t_frame, tile_size=NATIVE_TILE_SIZE)

                generated_tiles = []
                with torch.no_grad():
                    for tile, y0, y1, x0, x1, flags in tiles:
                        tile_batch = tile.unsqueeze(0)
                        try:
                            gen_tile = model(tile_batch)
                        except Exception as e:
                            logger.warning(f"Fallo de generación en un parche ({e}). Reintentando en CPU (será muy lento).")
                            gen_tile = model(tile_batch, force_cpu=True)
                        generated_tiles.append((gen_tile.squeeze(0), y0, y1, x0, x1, flags))

                    out_tensor = mem_mgr.blend_tiles(generated_tiles, orig_h, orig_w, scale=1)

                out_np = (out_tensor.permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                writer.write_frame(out_np)

                if preview_callback:
                    preview_callback(frame, out_np)

                now = time.time()
                time_delta = now - last_fps_calc_time
                if time_delta >= 1.0:
                    current_fps = (frame_idx - last_fps_frame_idx) / time_delta
                    last_fps_calc_time = now
                    last_fps_frame_idx = frame_idx

                eta_str = "--:--:--"
                if total_frames > 0 and current_fps > 0:
                    remaining_secs = int((total_frames - frame_idx) / current_fps)
                    m, s = divmod(remaining_secs, 60)
                    h, m = divmod(m, 60)
                    eta_str = f"{h:02d}:{m:02d}:{s:02d}"

                if progress_callback:
                    telemetry = {
                        "vram_used_pct": int(mem_mgr.get_current_memory_usage_ratio() * 100),
                        "tile_size": mem_mgr.current_tile_size,
                        "device": amd_hardware.backend_name
                    }
                    progress_callback(frame_idx, total_frames, current_fps, eta_str, telemetry)

            logger.info(f"Generación finalizada. Fotogramas completados: {frame_idx}")
            return not self._stop_requested

        except Exception as e:
            logger.error(f"Error durante la generación de video: {e}", exc_info=True)
            raise e
        finally:
            if reader:
                reader.close()
            if writer:
                writer.close()
            self.is_running = False
