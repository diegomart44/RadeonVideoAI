"""
Frame Interpolation Engine (rife-ncnn-vulkan, x2/x4 FPS) — resumable, disk-conscious.

Deliberately separate from core/engine.py and core/generative_engine.py:
raising frame rate via optical-flow interpolation is a third, unrelated AI
pipeline with its own tab, worker and failure modes.

Why PNG batches instead of the raw-pipe streaming the other two engines
use: rife-ncnn-vulkan (core/interpolation_models.py) only exposes a
folder-batch interface, not a per-pair callable API — unlike this app's
first ONNX-Runtime-based attempt, which could stream frame-by-frame but
measured meaningfully slower than Flowframes' own use of this exact same
native binary. A batch of SEGMENT_SOURCE_FRAMES source frames is extracted
to a temp PNG folder, interpolated (one 2x pass, or two chained for 4x —
the same way Flowframes does 4x, confirmed from its own log output), then
immediately re-encoded and the PNGs deleted — so disk usage stays bounded
to one batch's worth of loose frames, never the whole video.

Batch boundaries and the duplicate frames they create: every batch is its
own fresh rife-ncnn-vulkan run, and the tool always emits the *first* input
frame verbatim as its own output[0], plus (confirmed empirically — see
PNG byte-for-byte comparisons during development) one extra byte-identical
duplicate of the *last* output frame when a batch ends. Consecutive batches
share one overlap frame (the previous batch's true last source frame) so
the interpolated transition between batches isn't just skipped, and the
resulting duplicates at that shared seam are trimmed by direct byte
comparison (`_trim_trailing_duplicates_inplace` / `_trim_leading_duplicates`)
rather than by precomputing exact
counts per multiplier — simpler and correct regardless of how many passes
produced the output.

Resumability & disk usage, otherwise unchanged from the first version:
- Each finished segment is a small encoded MPEG-TS file, checkpointed to
  `progress.json` in a job-scoped temp folder; batched (CONSOLIDATE_BATCH
  at a time) into one growing `accumulated.ts` via ffmpeg's `concat:`
  PROTOCOL — not the concat demuxer, which was found during development to
  introduce a small timestamp jump at every join — then deleted.
- A crash or cancel leaves this state in place on purpose; the next run
  against the same input+output+multiplier resumes via `-ss` input seeking
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
from .media_pipeline import FFmpegLocator, VideoMetadataReader
from .interpolation_models import RifeVulkanModel

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
    concatenation), not the general-purpose concat DEMUXER — see module
    docstring for why."""
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


def _extract_frames_png(input_path: str, seek_time: float, count: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ffmpeg = FFmpegLocator.get_ffmpeg_path()
    cmd = [ffmpeg, "-y", "-v", "error"]
    if seek_time > 0:
        cmd += ["-ss", f"{seek_time:.6f}"]
    cmd += ["-i", input_path, "-frames:v", str(count), os.path.join(out_dir, "%08d.png")]
    subprocess.run(cmd, check=True, capture_output=True, text=True, creationflags=_NO_WINDOW_FLAGS)


def _is_identical_pixels(path_a: str, path_b: str) -> bool:
    """Pixel-level (not byte-level) comparison: the leading-duplicate check
    compares a PNG written by ffmpeg against one written by rife-ncnn-vulkan
    for the same source frame — pixel-identical, but not necessarily
    byte-identical, since the two tools' PNG encoders don't produce the
    same bytes for identical pixel content (different metadata/compression).
    The trailing-duplicate check compares two rife-ncnn-vulkan outputs,
    which already are byte-identical, but pixel comparison is correct there
    too and keeps this one code path simple."""
    a = _read_png_rgb(path_a)
    b = _read_png_rgb(path_b)
    if a is None or b is None or a.shape != b.shape:
        return False
    return np.array_equal(a, b)


def _trim_trailing_duplicates_inplace(out_dir: str, sorted_names: List[str]) -> List[str]:
    """Deletes (from disk, not just the list) the trailing duplicate
    rife-ncnn-vulkan leaves at the end of every batch, and returns the
    remaining names. Must run right after EVERY pass, not just once at the
    very end: for a second (4x) pass, the duplicate's two copies of the
    final frame would otherwise both feed into that pass, which then
    computes an actual (floating-point, not exactly identical) interpolated
    "midpoint" between two bit-identical images — a near-duplicate the
    exact-pixel check below can no longer catch, compounding into 2-3
    leftover near-duplicate frames instead of the expected one. Trimming
    the clean, exactly-duplicate pair before it ever reaches a next pass
    avoids creating that fuzzy near-duplicate in the first place."""
    names = list(sorted_names)
    while len(names) >= 2 and _is_identical_pixels(os.path.join(out_dir, names[-1]), os.path.join(out_dir, names[-2])):
        os.remove(os.path.join(out_dir, names[-1]))
        names.pop()
    return names


def _trim_leading_duplicates(out_dir: str, sorted_names: List[str], leading_ref_path: Optional[str]) -> List[str]:
    """Drops the leading frame(s) that match leading_ref_path: the segment's
    own shared overlap frame, already written as the previous segment's
    last output frame. Only needs to run once, on the final pass's output —
    unlike the trailing duplicate, nothing about chaining two passes makes
    this fuzzy, since there's no equivalent adjacent-identical-frame pair
    on the leading side to compound."""
    names = list(sorted_names)
    if leading_ref_path:
        while names and _is_identical_pixels(os.path.join(out_dir, names[0]), leading_ref_path):
            names.pop(0)
    return names


def _encode_png_sequence(png_dir: str, start_number: int, frame_count: int, out_fps: float,
                          segment_path: str, encoder: str, bitrate_mbps: int):
    ffmpeg = FFmpegLocator.get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-start_number", str(start_number),
        "-r", f"{out_fps:.4f}",
        "-i", os.path.join(png_dir, "%08d.png"),
        "-frames:v", str(frame_count),
    ]
    if encoder == "hevc_amf":
        cmd += ["-c:v", "hevc_amf", "-quality", "quality", "-rc", "cbr",
                "-b:v", f"{bitrate_mbps}M", "-maxrate", f"{bitrate_mbps + 10}M",
                "-bufsize", f"{bitrate_mbps * 2}M", "-pix_fmt", "yuv420p"]
    elif encoder == "h264_amf":
        cmd += ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cbr",
                "-b:v", f"{bitrate_mbps}M", "-pix_fmt", "yuv420p"]
    elif encoder == "libx265":
        cmd += ["-c:v", "libx265", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
    # -bf 0: avoids PTS jumps where independently-encoded segments later join
    # (see _consolidate / the module docstring on the concat demuxer finding).
    cmd += ["-bf", "0", "-f", "mpegts", segment_path]
    res = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW_FLAGS)
    if res.returncode != 0:
        raise RuntimeError(f"Fallo al codificar un segmento interpolado: {res.stderr}")


def _read_png_rgb(path: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        return None


class InterpolationEngine:
    """Resumable rife-ncnn-vulkan frame-interpolation pipeline (x2/x4 FPS)."""

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

        def _status(msg):
            if status_callback:
                status_callback(msg)

        try:
            meta = VideoMetadataReader.probe(input_path)
            fps = meta["fps"]
            total_frames = meta["total_frames"]
            has_audio = meta["has_audio"]
            out_fps = fps * multiplier
            passes = 1 if multiplier == 2 else 2

            temp_dir = _job_temp_dir(input_path, output_path, multiplier)
            os.makedirs(temp_dir, exist_ok=True)
            accumulated_ts = os.path.join(temp_dir, "accumulated.ts")

            state = _load_progress(temp_dir, input_path)
            frame_idx = state["completed_source_frames"]
            pending_segments: List[str] = state["pending_segments"]
            if frame_idx == 0 and not pending_segments and os.path.isfile(accumulated_ts):
                os.remove(accumulated_ts)  # stale leftover from a different/changed source

            if frame_idx > 0:
                logger.info(f"Reanudando interpolación desde el fotograma {frame_idx}.")
                _status(f"Progreso anterior encontrado — reanudando desde el fotograma {frame_idx}/{total_frames}...")
            else:
                _status("Preparando motor de interpolación (puede descargar la primera vez)...")

            model = RifeVulkanModel(progress_cb=_status)

            if "amf" in encoder and not FFmpegLocator.check_amf_support():
                fallback_enc = "libx265" if "hevc" in encoder else "libx264"
                logger.warning(f"AMD AMF no detectado. Conmutando a {fallback_enc}.")
                encoder = fallback_enc

            work_dir = os.path.join(temp_dir, "work")

            start_time_wall = time.time()
            last_fps_calc_time = start_time_wall
            last_fps_frame_idx = frame_idx
            current_fps = 0.0

            while not self._stop_requested:
                self._pause_event.wait()
                if self._stop_requested:
                    break

                is_first_segment = frame_idx == 0
                extract_start_frame = frame_idx if is_first_segment else frame_idx - 1
                extract_count = SEGMENT_SOURCE_FRAMES if is_first_segment else SEGMENT_SOURCE_FRAMES + 1
                seek_time = extract_start_frame / fps if fps > 0 else 0.0

                if os.path.isdir(work_dir):
                    shutil.rmtree(work_dir, ignore_errors=True)
                in_dir = os.path.join(work_dir, "in")
                _extract_frames_png(input_path, seek_time, extract_count, in_dir)

                in_names = sorted(os.listdir(in_dir))
                extracted = len(in_names)
                new_source_frames = extracted - (0 if is_first_segment else 1)
                if new_source_frames <= 0:
                    shutil.rmtree(work_dir, ignore_errors=True)
                    break  # reached the true end of the video

                leading_ref = None if is_first_segment else os.path.join(in_dir, in_names[0])

                cur_dir = in_dir
                for p in range(passes):
                    next_dir = os.path.join(work_dir, f"pass{p}")
                    model.run_2x_pass(cur_dir, next_dir)
                    next_names = sorted(os.listdir(next_dir))
                    _trim_trailing_duplicates_inplace(next_dir, next_names)  # see docstring: must happen before this feeds the next pass
                    cur_dir = next_dir

                out_names = sorted(os.listdir(cur_dir))
                kept = _trim_leading_duplicates(cur_dir, out_names, leading_ref)

                if preview_callback and kept:
                    orig_img = _read_png_rgb(os.path.join(in_dir, in_names[0 if is_first_segment else 1]))
                    mid_img = _read_png_rgb(os.path.join(cur_dir, kept[len(kept) // 2]))
                    if orig_img is not None and mid_img is not None:
                        preview_callback(orig_img, mid_img)

                segment_path = os.path.join(temp_dir, f"segment_{frame_idx:08d}.ts")
                start_number = int(os.path.splitext(kept[0])[0])
                _encode_png_sequence(cur_dir, start_number, len(kept), out_fps, segment_path, encoder, bitrate_mbps)
                shutil.rmtree(work_dir, ignore_errors=True)

                frame_idx += new_source_frames
                pending_segments.append(segment_path)
                if len(pending_segments) >= CONSOLIDATE_BATCH:
                    _consolidate(accumulated_ts, pending_segments, temp_dir)
                    pending_segments = []
                _save_progress(temp_dir, input_path, frame_idx, pending_segments)

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
                        "vram_used_pct": 0,  # no per-process VRAM API for the Vulkan binary
                        "tile_size": 0,
                        "device": amd_hardware.backend_name
                    }
                    progress_callback(frame_idx, total_frames, current_fps, eta_str, telemetry)

            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)

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
