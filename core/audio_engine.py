"""
Audio Restoration Engine.
Extracts a video/audio file's audio track, restores it with AI (noise,
reverberation, low resolution, clipping — see core/audio_models.py), and
remuxes the result back against the original video stream untouched
(-c:v copy) or, for a pure-audio input, saves the restored track directly.

Deliberately its own standalone engine (not a step bolted onto
Reescalar/Generar/Interpolar): different input modality (a 1D audio
signal, not per-frame video), a different AI model, and a different
backend (CPU-only PyTorch, not ONNX Runtime DirectML). Sharing code with
the frame-based engines would only create a false coupling between
unrelated pipelines.
"""

import os
import sys
import math
import time
import logging
import threading
import tempfile
import shutil
import subprocess
import numpy as np
import soundfile as sf
from typing import Callable, Optional, Dict, Any

from .media_pipeline import FFmpegLocator

logger = logging.getLogger("RadeonVideoAI.AudioEngine")

_NO_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

SAMPLE_RATE = 44100
CHUNK_SECONDS = 30  # matches VoiceFixer's own internal segment size


class AudioRestorationEngine:
    """
    Mirrors VideoProcessingEngine's pause/resume/cancel pattern (a
    threading.Event checked between units of work), but the "unit of work"
    here is a ~30s audio chunk instead of a video frame — restoring a
    chunk takes tens of seconds on CPU, so pause/cancel are only as
    responsive as that chunk boundary, not frame-accurate.
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

    def process_audio(
        self,
        input_path: str,
        output_path: str,
        mode: int = 0,
        audio_bitrate_kbps: int = 192,
        progress_callback: Optional[Callable[[int, int, float, str, Dict[str, Any]], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        self.is_running = True
        self._stop_requested = False
        self.is_paused = False
        self._pause_event.set()

        work_dir = tempfile.mkdtemp(prefix="radeonvideoai_audio_")
        extracted_wav = os.path.join(work_dir, "extracted.wav")
        restored_wav = os.path.join(work_dir, "restored.wav")

        try:
            if not self._has_audio_stream(input_path):
                raise RuntimeError("El archivo de entrada no tiene una pista de audio para restaurar.")
            has_video = self._has_video_stream(input_path)

            if status_callback:
                status_callback("Extrayendo la pista de audio original...")
            # Forcing mono + 44.1kHz here means every downstream step (model,
            # chunking math) can assume that shape instead of re-checking it.
            self._extract_audio(input_path, extracted_wav)

            if status_callback:
                status_callback("Preparando modelo de IA de restauración de audio (puede descargar ~600MB la primera vez)...")

            def _dl_progress(downloaded, total, filename):
                if status_callback and total:
                    pct = int(downloaded / total * 100)
                    status_callback(f"Descargando modelo de audio {filename}: {pct}%")

            from .audio_models import AudioRestorationModel
            model = AudioRestorationModel(progress_cb=_dl_progress)

            if status_callback:
                status_callback(f"Modelo de audio listo — backend: {model.active_backend}")

            wav, _sr = sf.read(extracted_wav, dtype="float32", always_2d=False)
            total_samples = wav.shape[0]
            chunk_len = SAMPLE_RATE * CHUNK_SECONDS
            total_chunks = max(1, math.ceil(total_samples / chunk_len))

            writer = sf.SoundFile(
                restored_wav, mode="w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16"
            )

            start_time = time.time()
            last_calc_time = start_time
            last_chunk_idx = 0
            current_rate = 0.0
            chunk_idx = 0

            try:
                while not self._stop_requested and chunk_idx < total_chunks:
                    self._pause_event.wait()
                    if self._stop_requested:
                        break

                    seg = wav[chunk_idx * chunk_len: (chunk_idx + 1) * chunk_len]
                    restored_seg = model.restore_chunk(seg, mode=mode)
                    # restore_inmem() returns a (1, N) channel-first array,
                    # not a flat (N,) one; a mono SoundFile.write() needs 1D.
                    restored_seg = np.asarray(restored_seg, dtype=np.float32).reshape(-1)
                    writer.write(restored_seg)

                    chunk_idx += 1

                    now = time.time()
                    time_delta = now - last_calc_time
                    if time_delta >= 1.0:
                        current_rate = (chunk_idx - last_chunk_idx) / time_delta
                        last_calc_time = now
                        last_chunk_idx = chunk_idx

                    eta_str = "--:--:--"
                    if current_rate > 0:
                        remaining_secs = int((total_chunks - chunk_idx) / current_rate)
                        m, s = divmod(remaining_secs, 60)
                        h, m = divmod(m, 60)
                        eta_str = f"{h:02d}:{m:02d}:{s:02d}"

                    if progress_callback:
                        telemetry = {"tile_size": 0, "device": model.active_backend}
                        progress_callback(chunk_idx, total_chunks, current_rate, eta_str, telemetry)
            finally:
                writer.close()

            if self._stop_requested:
                logger.info("Restauración de audio cancelada por el usuario.")
                return False

            if status_callback:
                status_callback("Combinando el audio restaurado con el original...")
            self._mux_output(input_path, restored_wav, output_path, has_video, audio_bitrate_kbps)

            logger.info("Restauración de audio finalizada.")
            return True

        except Exception as e:
            logger.error(f"Error durante la restauración de audio: {e}", exc_info=True)
            raise e
        finally:
            self.is_running = False
            shutil.rmtree(work_dir, ignore_errors=True)

    def _has_video_stream(self, path: str) -> bool:
        ffprobe = FFmpegLocator.get_ffprobe_path()
        try:
            res = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v",
                 "-show_entries", "stream=index", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW_FLAGS
            )
            return bool(res.stdout.strip())
        except Exception:
            return False

    def _has_audio_stream(self, path: str) -> bool:
        ffprobe = FFmpegLocator.get_ffprobe_path()
        try:
            res = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=index", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW_FLAGS
            )
            return bool(res.stdout.strip())
        except Exception:
            return False

    def _extract_audio(self, input_path: str, out_wav: str):
        ffmpeg = FFmpegLocator.get_ffmpeg_path()
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-i", input_path,
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le",
            out_wav
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, creationflags=_NO_WINDOW_FLAGS)
        if res.returncode != 0 or not os.path.isfile(out_wav):
            raise RuntimeError(f"No se pudo extraer el audio del archivo de entrada: {res.stderr}")

    def _mux_output(self, input_path: str, restored_wav: str, output_path: str, has_video: bool, audio_bitrate_kbps: int):
        ffmpeg = FFmpegLocator.get_ffmpeg_path()
        if has_video:
            cmd = [
                ffmpeg, "-y", "-v", "error",
                "-i", input_path, "-i", restored_wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", f"{audio_bitrate_kbps}k",
                "-shortest", output_path
            ]
        else:
            # Pure-audio input: let ffmpeg pick the encoder from
            # output_path's extension (e.g. .wav stays lossless PCM, .mp3
            # goes through libmp3lame).
            cmd = [ffmpeg, "-y", "-v", "error", "-i", restored_wav, output_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, creationflags=_NO_WINDOW_FLAGS)
        if res.returncode != 0:
            raise RuntimeError(f"No se pudo combinar el audio restaurado con el resultado final: {res.stderr}")
