"""
RadeonVideoAI Core Video Processing Engine.
Orchestrates FFmpeg binary streaming, AMD DirectML/ROCm/CPU acceleration,
dynamic VRAM tiling with Cosine Feathering, and live preview dispatch.
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
from .inference import create_model

logger = logging.getLogger("RadeonVideoAI.Engine")

class VideoProcessingEngine:
    """
    Main processing pipeline executing the AI upscaling/restoration loop.
    Supports asynchronous execution, pause, resume, and cancellation.
    """

    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self._stop_requested = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Unpaused initially

    def cancel(self):
        """Signals the processing loop to gracefully cancel and clean up."""
        self._stop_requested = True
        self._pause_event.set()  # Unblock if paused

    def pause(self):
        """Pauses the processing loop."""
        self.is_paused = True
        self._pause_event.clear()

    def resume(self):
        """Resumes the processing loop."""
        self.is_paused = False
        self._pause_event.set()

    def process_video(
        self,
        input_path: str,
        output_path: str,
        scale: int = 2,
        model_name: str = "general_v3",
        tile_size: int = 512,
        overlap: int = 48,
        denoise: float = 0.0,
        sharpen: float = 0.0,
        encoder: str = "hevc_amf",
        bitrate_mbps: int = 25,
        vram_safety: float = 0.90,
        progress_callback: Optional[Callable[[int, int, float, str, Dict[str, Any]], None]] = None,
        preview_callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Executes end-to-end video enhancement.
        Streams frames through binary pipes, applies tiled inference, and muxes audio intact.
        """
        self.is_running = True
        self._stop_requested = False
        self.is_paused = False
        self._pause_event.set()

        reader: Optional[FFmpegFrameReader] = None
        writer: Optional[FFmpegFrameWriter] = None

        try:
            # 1. Inspect Video Metadata
            meta = VideoMetadataReader.probe(input_path)
            orig_w, orig_h = meta["width"], meta["height"]
            fps = meta["fps"]
            total_frames = meta["total_frames"]
            has_audio = meta["has_audio"]
            target_w = orig_w * scale
            target_h = orig_h * scale

            logger.info(f"Procesando: {orig_w}x{orig_h} -> {target_w}x{target_h} @ {fps:.2f}fps ({total_frames} frames)")

            # 2. Setup Memory Manager
            mem_mgr = MemoryManager(
                vram_total_mb=amd_hardware.vram_total_mb,
                max_vram_ratio=vram_safety,
                default_tile_size=tile_size,
                overlap=overlap
            )

            # 3. Setup AI Model (real pretrained Real-ESRGAN weights, ONNX Runtime DirectML)
            if status_callback:
                status_callback("Preparando modelo de IA (puede descargar pesos oficiales la primera vez)...")

            def _weights_progress(downloaded, total, filename):
                if status_callback and total:
                    pct = int(downloaded / total * 100)
                    status_callback(f"Descargando pesos IA {filename}: {pct}%")

            model = create_model(
                model_name=model_name,
                scale=scale,
                denoise_strength=denoise,
                sharpen_strength=sharpen,
                providers=amd_hardware.providers,
                progress_cb=_weights_progress
            )

            if status_callback:
                status_callback(f"Modelo IA listo — backend: {model.active_backend}")

            # 4. Fallback encoder check
            if "amf" in encoder and not FFmpegLocator.check_amf_support():
                fallback_enc = "libx265" if "hevc" in encoder else "libx264"
                logger.warning(f"AMD AMF no detectado en este sistema. Conmutando a {fallback_enc}.")
                encoder = fallback_enc

            # 5. Open Binary In-Memory Pipes
            reader = FFmpegFrameReader(input_path, orig_w, orig_h)
            writer = FFmpegFrameWriter(
                output_path=output_path,
                input_source_for_audio=input_path,
                width=target_w,
                height=target_h,
                fps=fps,
                has_audio=has_audio,
                encoder=encoder,
                bitrate_mbps=bitrate_mbps
            )

            # 6. Processing Loop
            frame_idx = 0
            start_time = time.time()
            last_fps_calc_time = start_time
            last_fps_frame_idx = 0
            current_fps = 0.0

            # Whole-frame fast path: tiling exists to keep arbitrarily large
            # frames within VRAM, but it has a real cost — 6 overlapping
            # 768px tiles for a 1080p-ish frame means ~6x the per-call
            # overhead plus redundant compute in every overlap margin.
            # Measured on this project's own dev hardware (RX 9060 XT,
            # 16GB): processing a 1916x812 frame whole took 541ms (1.85
            # fps); tiled at 768px/48px overlap took 1334ms (0.75 fps) for
            # the *same* frame — matching Topaz Video AI's own measured
            # throughput on identical hardware/resolution, which processes
            # frames whole via a native ffmpeg filter with no tiling at
            # all. So: try the whole frame once per job; if it fits in
            # VRAM, every frame after that skips tiling entirely. If it
            # doesn't fit (lower-VRAM cards, very high resolutions), the
            # exception is caught and the job transparently falls back to
            # the existing tiled path for its whole duration — tiling
            # never stops being correct, this only skips it when it's
            # provably unnecessary overhead on *this* machine.
            whole_frame_mode: Optional[bool] = None

            while not self._stop_requested:
                # Handle pause
                self._pause_event.wait()
                if self._stop_requested:
                    break

                frame = reader.read_frame()
                if frame is None:
                    # End of stream
                    break

                frame_idx += 1

                # Check dynamic VRAM threshold and degrade tile size if approaching limit
                mem_mgr.check_vram_safety()

                # Prepare tensor for inference: (H, W, 3) uint8 -> (1, 3, H, W) float32 in [0, 1]
                t_frame = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                if whole_frame_mode is not False:
                    try:
                        with torch.no_grad():
                            out_tensor = model(t_frame).squeeze(0)
                        if whole_frame_mode is None:
                            whole_frame_mode = True
                            logger.info("El fotograma completo entra en VRAM: procesando sin dividir en parches (más rápido).")
                            if status_callback:
                                status_callback("GPU con VRAM suficiente — procesando fotograma completo sin parches.")
                    except Exception as e:
                        whole_frame_mode = False
                        logger.warning(f"El fotograma completo no entra en VRAM de una vez ({e}); usando modo por parches para todo el video.")
                        out_tensor = None

                if whole_frame_mode is False:
                    # Split into tiles with Cosine Feathering margins
                    tiles = mem_mgr.split_into_tiles(t_frame)

                    upscaled_tiles = []
                    with torch.no_grad():
                        for tile, y0, y1, x0, x1, flags in tiles:
                            tile_batch = tile.unsqueeze(0)

                            try:
                                up_tile = model(tile_batch)
                            except Exception as e:
                                logger.warning(f"Fallo de inferencia GPU en un parche ({e}). Degradando y reintentando en CPU.")
                                mem_mgr.degrade_tile_size()
                                up_tile = model(tile_batch, force_cpu=True)

                            # Squeeze batch dimension
                            upscaled_tiles.append((up_tile.squeeze(0), y0, y1, x0, x1, flags))

                        # Blend all tiles seamlessly with Cosine Feathering
                        out_tensor = mem_mgr.blend_tiles(upscaled_tiles, target_h, target_w, scale=scale)

                # Convert back to (target_H, target_W, 3) uint8 numpy
                out_np = (out_tensor.permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)

                # Write to FFmpeg stdin pipe
                writer.write_frame(out_np)

                # Dispatch live preview frame (throttling handled by preview receiver/worker)
                if preview_callback:
                    preview_callback(frame, out_np)

                # Calculate speed and ETA
                now = time.time()
                time_delta = now - last_fps_calc_time
                if time_delta >= 1.0:
                    current_fps = (frame_idx - last_fps_frame_idx) / time_delta
                    last_fps_calc_time = now
                    last_fps_frame_idx = frame_idx

                eta_str = "--:--:--"
                if total_frames > 0 and current_fps > 0:
                    remaining_frames = total_frames - frame_idx
                    remaining_secs = int(remaining_frames / current_fps)
                    m, s = divmod(remaining_secs, 60)
                    h, m = divmod(m, 60)
                    eta_str = f"{h:02d}:{m:02d}:{s:02d}"

                if progress_callback:
                    vram_telemetry = {
                        "vram_used_pct": int(mem_mgr.get_current_memory_usage_ratio() * 100),
                        "tile_size": 0 if whole_frame_mode else mem_mgr.current_tile_size,
                        "device": amd_hardware.backend_name
                    }
                    progress_callback(frame_idx, total_frames, current_fps, eta_str, vram_telemetry)

            logger.info(f"Procesamiento finalizado. Fotogramas completados: {frame_idx}")
            return not self._stop_requested

        except Exception as e:
            logger.error(f"Error durante el procesamiento de video: {e}", exc_info=True)
            raise e
        finally:
            if reader:
                reader.close()
            if writer:
                writer.close()
            self.is_running = False
