"""
Frame Interpolation Engine (RIFE, x2/x4 FPS) — resumable, disk-conscious.

Deliberately separate from core/engine.py and core/generative_engine.py:
raising frame rate via optical-flow interpolation is a third, unrelated AI
pipeline with its own tab, worker and failure modes.

Resumability & disk usage, by design:
- Progress is checkpointed to disk every SEGMENT_SOURCE_FRAMES source frames
  as a small encoded MPEG-TS segment (compact — already-compressed video,
  not raw frames) in a job-scoped temp folder. This is what makes a crash
  cheap to recover from: at most one checkpoint's worth of work is repeated.
- Small segments are batched: once CONSOLIDATE_BATCH of them have piled up,
  they're merged with whatever was already consolidated into one
  `accumulated.ts` via ffmpeg's concat demuxer (`-c copy`, no re-encoding —
  and the technically correct way to join multiple independently-encoded
  TS files, since it rebases each file's timestamps; naive byte-level TS
  concatenation would leave every segment's PTS restarting near zero and
  produce a broken output). The now-redundant small segments are deleted
  right after — this is what keeps disk usage bounded to "encoded output
  produced so far" instead of piling up loose interpolated frames for the
  whole video. Batching (rather than consolidating on every single
  checkpoint) keeps the repeated re-copy of `accumulated.ts` from growing
  quadratically with video length.
- `progress.json` records how many source frames are safely accounted for
  (whether already inside accumulated.ts or still sitting as pending
  segment files) plus the source file's size/mtime, so a resume against a
  since-changed source file is detected and discarded rather than silently
  producing a mismatched result.
- On cancel (or a crash), this state is left in place on purpose — the next
  run with the same input+output+multiplier resumes via `-ss` input seeking
  instead of restarting. Only a successful full completion does the final
  consolidation + audio remux and deletes the temp folder.
"""

import os
import json
import time
import shutil
import hashlib
import logging
import threading
import subprocess
import numpy as np
from typing import Callable, Optional, Dict, Any, List

from .amd_backend import amd_hardware
from .media_pipeline import FFmpegLocator, VideoMetadataReader, FFmpegFrameReader, FFmpegFrameWriter
from .interpolation_models import create_interpolation_model, timesteps_for_multiplier

logger = logging.getLogger("RadeonVideoAI.InterpolationEngine")

SEGMENT_SOURCE_FRAMES = 120  # checkpoint roughly every ~4-5s of source video
CONSOLIDATE_BATCH = 10       # merge into accumulated.ts every ~10 checkpoints
_NO_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _job_temp_dir(input_path: str, output_path: str, multiplier: int) -> str:
    key = f"{os.path.abspath(input_path)}|{os.path.abspath(output_path)}|x{multiplier}"
    job_hash = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    base = os.path.dirname(os.path.abspath(output_path)) or "."
    return os.path.join(base, f".radeonvideoai_interp_{job_hash}")


def _progress_path(temp_dir: str) -> str:
    return os.path.join(temp_dir, "progress.json")


def _load_progress(temp_dir: str, input_path: str) -> Dict[str, Any]:
    path = _progress_path(temp_dir)
    default = {"completed_source_frames": 0, "pending_segments": []}
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        st = os.stat(input_path)
        if state.get("source_size") != st.st_size or state.get("source_mtime") != st.st_mtime:
            logger.warning("El video original cambió desde el progreso guardado; se reinicia desde cero.")
            return default
        pending = [p for p in state.get("pending_segments", []) if os.path.isfile(p)]
        return {"completed_source_frames": state.get("completed_source_frames", 0), "pending_segments": pending}
    except Exception:
        return default


def _save_progress(temp_dir: str, input_path: str, completed_source_frames: int, pending_segments: List[str]):
    st = os.stat(input_path)
    state = {
        "completed_source_frames": completed_source_frames,
        "pending_segments": pending_segments,
        "source_size": st.st_size,
        "source_mtime": st.st_mtime,
    }
    path = _progress_path(temp_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp_path, path)


def _consolidate(accumulated_ts: str, pending_segments: List[str], temp_dir: str):
    """Merges accumulated.ts (if any) + all pending segments into a new
    accumulated.ts using ffmpeg's `concat:` PROTOCOL (raw stream-level TS
    concatenation), not the general-purpose concat DEMUXER. The demuxer
    re-derives each file's duration/timestamp offset and was empirically
    found to introduce a small (~half a frame) timestamp jump at every
    join — the concat protocol instead relies on MPEG-TS's own continuous
    PCR/PTS stream design and joins cleanly with no drift (verified frame
    by frame with `showinfo` before adopting this)."""
    if not pending_segments:
        return
    if not os.path.isfile(accumulated_ts) and len(pending_segments) == 1:
        os.replace(pending_segments[0], accumulated_ts)
        return

    files = ([accumulated_ts] if os.path.isfile(accumulated_ts) else []) + pending_segments
    concat_arg = "concat:" + "|".join(os.path.abspath(fp) for fp in files)

    new_accumulated = accumulated_ts + ".new"
    ffmpeg = FFmpegLocator.get_ffmpeg_path()
    cmd = [ffmpeg, "-y", "-v", "error", "-i", concat_arg, "-c", "copy", "-f", "mpegts", new_accumulated]
    res = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW_FLAGS)
    if res.returncode != 0:
        raise RuntimeError(f"Fallo al consolidar segmentos de interpolación: {res.stderr}")

    for fp in files:
        try:
            os.remove(fp)
        except OSError:
            pass
    os.replace(new_accumulated, accumulated_ts)


class InterpolationEngine:
    """Resumable RIFE frame-interpolation pipeline (x2/x4 FPS)."""

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
        multiplier: int = 2,
        encoder: str = "hevc_amf",
        bitrate_mbps: int = 25,
        progress_callback: Optional[Callable[[int, int, float, str, Dict[str, Any]], None]] = None,
        preview_callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        self.is_running = True
        self._stop_requested = False
        self.is_paused = False
        self._pause_event.set()

        reader: Optional[FFmpegFrameReader] = None
        segment_writer: Optional[FFmpegFrameWriter] = None

        def _status(msg):
            if status_callback:
                status_callback(msg)

        try:
            meta = VideoMetadataReader.probe(input_path)
            w, h = meta["width"], meta["height"]
            fps = meta["fps"]
            total_frames = meta["total_frames"]
            has_audio = meta["has_audio"]
            out_fps = fps * multiplier

            temp_dir = _job_temp_dir(input_path, output_path, multiplier)
            os.makedirs(temp_dir, exist_ok=True)
            accumulated_ts = os.path.join(temp_dir, "accumulated.ts")

            state = _load_progress(temp_dir, input_path)
            resume_frame = state["completed_source_frames"]
            pending_segments: List[str] = state["pending_segments"]
            if resume_frame == 0 and not pending_segments and os.path.isfile(accumulated_ts):
                os.remove(accumulated_ts)  # stale leftover from a different/changed source

            if resume_frame > 0:
                logger.info(f"Reanudando interpolación desde el fotograma {resume_frame}.")
                _status(f"Progreso anterior encontrado — reanudando desde el fotograma {resume_frame}/{total_frames}...")
            else:
                _status("Preparando modelo de interpolación (puede descargar la primera vez)...")

            model = create_interpolation_model(providers=amd_hardware.providers, progress_cb=_status)
            timesteps = timesteps_for_multiplier(multiplier)
            _status(f"Modelo de interpolación listo — backend: {model.active_backend}")

            if "amf" in encoder and not FFmpegLocator.check_amf_support():
                fallback_enc = "libx265" if "hevc" in encoder else "libx264"
                logger.warning(f"AMD AMF no detectado. Conmutando a {fallback_enc}.")
                encoder = fallback_enc

            start_time = (resume_frame / fps) if (fps > 0 and resume_frame > 0) else 0.0
            reader = FFmpegFrameReader(input_path, w, h, start_time=start_time)

            prev_frame = reader.read_frame()
            if prev_frame is None:
                # Nothing left — either an empty video or resuming at/after
                # the last frame. Treat whatever was accumulated as final.
                reader.close()
                reader = None
                _consolidate(accumulated_ts, pending_segments, temp_dir)
                if os.path.isfile(accumulated_ts):
                    self._finalize(accumulated_ts, input_path, output_path, has_audio, _status)
                shutil.rmtree(temp_dir, ignore_errors=True)
                return True

            frame_idx = resume_frame
            segment_count = 0
            segment_path = None

            def open_segment():
                nonlocal segment_path
                segment_path = os.path.join(temp_dir, f"segment_{frame_idx:08d}.ts")
                return FFmpegFrameWriter(
                    output_path=segment_path,
                    input_source_for_audio=input_path,
                    width=w, height=h, fps=out_fps,
                    has_audio=False,  # audio is muxed once at the final remux
                    encoder=encoder, bitrate_mbps=bitrate_mbps,
                    disable_bframes=True  # avoid PTS jumps where segments join, see media_pipeline.py
                )

            segment_writer = open_segment()
            if resume_frame == 0:
                segment_writer.write_frame(prev_frame)
            # else: prev_frame is source frame `resume_frame`, already the
            # last frame written by the previous run — it's only needed here
            # as the pairing reference for the next interpolation, not to be
            # written again (that would duplicate it in the output).

            start_time_wall = time.time()
            last_fps_calc_time = start_time_wall
            last_fps_frame_idx = frame_idx
            current_fps = 0.0

            while not self._stop_requested:
                self._pause_event.wait()
                if self._stop_requested:
                    break

                next_frame = reader.read_frame()
                if next_frame is None:
                    break

                mids = model.interpolate(prev_frame, next_frame, timesteps)
                for mid in mids:
                    segment_writer.write_frame(mid)
                segment_writer.write_frame(next_frame)

                if preview_callback:
                    preview_callback(prev_frame, mids[len(mids) // 2] if mids else next_frame)

                frame_idx += 1
                segment_count += 1
                prev_frame = next_frame

                if segment_count >= SEGMENT_SOURCE_FRAMES:
                    segment_writer.close()
                    pending_segments.append(segment_path)
                    segment_count = 0
                    if len(pending_segments) >= CONSOLIDATE_BATCH:
                        _consolidate(accumulated_ts, pending_segments, temp_dir)
                        pending_segments = []
                    _save_progress(temp_dir, input_path, frame_idx, pending_segments)
                    segment_writer = open_segment()

                now = time.time()
                if now - last_fps_calc_time >= 1.0:
                    current_fps = (frame_idx - last_fps_frame_idx) / (now - last_fps_calc_time)
                    last_fps_calc_time = now
                    last_fps_frame_idx = frame_idx

                eta_str = "--:--:--"
                if total_frames > 0 and current_fps > 0:
                    remaining_secs = int((total_frames - frame_idx) / current_fps)
                    m, s = divmod(max(remaining_secs, 0), 60)
                    hh, m = divmod(m, 60)
                    eta_str = f"{hh:02d}:{m:02d}:{s:02d}"

                if progress_callback:
                    telemetry = {
                        "vram_used_pct": 0,
                        "tile_size": 0,
                        "device": amd_hardware.backend_name
                    }
                    progress_callback(frame_idx, total_frames, current_fps, eta_str, telemetry)

            # Flush whatever is left in the current segment, checkpointed or not.
            segment_writer.close()
            if segment_count > 0:
                pending_segments.append(segment_path)
            segment_writer = None
            reader.close()
            reader = None
            _save_progress(temp_dir, input_path, frame_idx, pending_segments)

            if self._stop_requested:
                _status("Interpolación cancelada — el progreso se guardó y se reanudará la próxima vez.")
                logger.info(f"Interpolación cancelada en el fotograma {frame_idx}. Progreso conservado en {temp_dir}")
                return False

            _consolidate(accumulated_ts, pending_segments, temp_dir)
            self._finalize(accumulated_ts, input_path, output_path, has_audio, _status)
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Interpolación finalizada. Fotogramas de origen procesados: {frame_idx}")
            return True

        except Exception as e:
            logger.error(f"Error durante la interpolación de video: {e}", exc_info=True)
            raise e
        finally:
            if segment_writer:
                segment_writer.close()
            if reader:
                reader.close()
            self.is_running = False

    @staticmethod
    def _finalize(accumulated_ts: str, input_path: str, output_path: str, has_audio: bool, status_cb):
        """One-shot remux of the accumulated interpolated video stream into the
        requested container, muxing back the original audio untouched (frame
        interpolation doesn't change real-time duration, so the original
        audio track lines up as-is)."""
        status_cb("Finalizando: combinando con el audio original...")
        ffmpeg = FFmpegLocator.get_ffmpeg_path()
        cmd = [ffmpeg, "-y", "-v", "error", "-i", accumulated_ts]
        if has_audio and os.path.isfile(input_path):
            cmd += ["-i", input_path, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"]
        else:
            cmd += ["-map", "0:v:0"]
        cmd += ["-c:v", "copy", output_path]
        res = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW_FLAGS)
        if res.returncode != 0:
            raise RuntimeError(f"Fallo al finalizar el video interpolado: {res.stderr}")
