"""
Binary In-Memory Streaming Media Pipeline with FFmpeg & AMD AMF Hardware Acceleration.
Preserves original audio streams intact (-c:a copy) and streams raw frames via pipes.
"""

import os
import sys
import json
import shutil
import logging
import subprocess
import numpy as np
from typing import Optional, Dict, Any, Generator

logger = logging.getLogger("RadeonVideoAI.MediaPipeline")

# On Windows, launching a console executable (ffmpeg/ffprobe) from a windowed
# (non-console) app spawns a visible console window per process unless this
# flag is passed. CREATE_NO_WINDOW only exists on win32.
_NO_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class FFmpegLocator:
    """Locates FFmpeg and FFprobe binaries prioritizing portable local root."""
    
    @staticmethod
    def get_ffmpeg_path() -> str:
        # 1. Local portable app root
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_ffmpeg = os.path.join(app_dir, "ffmpeg.exe")
        if os.path.isfile(local_ffmpeg):
            return local_ffmpeg

        # 2. System PATH
        which_ffmpeg = shutil.which("ffmpeg")
        if which_ffmpeg:
            return which_ffmpeg

        # 3. Common Windows locations
        known_locations = [
            r"C:\Program Files\BlueStacks_nxt\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        ]
        for loc in known_locations:
            if os.path.isfile(loc):
                return loc

        return "ffmpeg"

    @staticmethod
    def get_ffprobe_path() -> str:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_ffprobe = os.path.join(app_dir, "ffprobe.exe")
        if os.path.isfile(local_ffprobe):
            return local_ffprobe

        which_ffprobe = shutil.which("ffprobe")
        if which_ffprobe:
            return which_ffprobe

        known_locations = [
            r"C:\Program Files\BlueStacks_nxt\ffprobe.exe",
            r"C:\ffmpeg\bin\ffprobe.exe",
        ]
        for loc in known_locations:
            if os.path.isfile(loc):
                return loc

        return "ffprobe"

    @classmethod
    def check_amf_support(cls) -> bool:
        """Checks if FFmpeg binary and system driver support AMD AMF encoding."""
        ffmpeg = cls.get_ffmpeg_path()
        try:
            res = subprocess.run([ffmpeg, "-encoders"], capture_output=True, text=True, timeout=5,
                                  creationflags=_NO_WINDOW_FLAGS)
            return "hevc_amf" in res.stdout or "h264_amf" in res.stdout
        except Exception:
            return False


class VideoMetadataReader:
    """Extracts video resolution, framerate, duration, and audio stream info."""
    
    @staticmethod
    def probe(file_path: str) -> Dict[str, Any]:
        ffprobe = FFmpegLocator.get_ffprobe_path()
        cmd = [
            ffprobe,
            "-v", "error",
            "-show_entries", "stream=index,codec_type,codec_name,width,height,r_frame_rate,duration,nb_frames:format=duration",
            "-of", "json",
            file_path
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                  creationflags=_NO_WINDOW_FLAGS)
            if res.returncode != 0:
                raise RuntimeError(f"Error ejecutando ffprobe: {res.stderr}")
            
            data = json.loads(res.stdout)
            video_stream = None
            audio_streams = []

            for st in data.get("streams", []):
                if st.get("codec_type") == "video" and not video_stream:
                    video_stream = st
                elif st.get("codec_type") == "audio":
                    audio_streams.append(st)

            if not video_stream:
                raise ValueError("No se detectó pista de video en el archivo.")

            width = int(video_stream.get("width", 1920))
            height = int(video_stream.get("height", 1080))
            
            # Parse framerate (e.g. "30000/1001" or "30/1")
            fps_str = video_stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = map(float, fps_str.split("/"))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(fps_str)

            # Parse duration and total frames
            duration = float(video_stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
            nb_frames = video_stream.get("nb_frames")
            if nb_frames and nb_frames.isdigit():
                total_frames = int(nb_frames)
            elif duration > 0:
                total_frames = int(duration * fps)
            else:
                total_frames = 0

            return {
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames,
                "has_audio": len(audio_streams) > 0,
                "audio_count": len(audio_streams)
            }
        except Exception as e:
            logger.warning(f"ffprobe no disponible o falló ({e}). Intentando sondeo directo con ffmpeg...")
            return VideoMetadataReader._probe_with_ffmpeg(file_path)

    @staticmethod
    def _probe_with_ffmpeg(file_path: str) -> Dict[str, Any]:
        import re
        ffmpeg = FFmpegLocator.get_ffmpeg_path()
        try:
            res = subprocess.run([ffmpeg, "-i", file_path], capture_output=True, text=True, timeout=5,
                                  creationflags=_NO_WINDOW_FLAGS)
            output = res.stderr

            width, height = 1920, 1080
            m_res = re.search(r"(\d{3,5})x(\d{3,5})", output)
            if m_res:
                width, height = int(m_res.group(1)), int(m_res.group(2))

            fps = 30.0
            m_fps = re.search(r"(\d+(?:\.\d+)?)\s*fps", output)
            if m_fps:
                fps = float(m_fps.group(1))

            duration = 0.0
            m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", output)
            if m_dur:
                h, m, s = float(m_dur.group(1)), float(m_dur.group(2)), float(m_dur.group(3))
                duration = h * 3600 + m * 60 + s

            total_frames = int(duration * fps) if duration > 0 else 0
            has_audio = "Audio:" in output

            return {
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames,
                "has_audio": has_audio,
                "audio_count": 1 if has_audio else 0
            }
        except Exception as err:
            logger.error(f"Fallback de sondeo con ffmpeg falló: {err}")
            return {
                "width": 1920, "height": 1080, "fps": 30.0,
                "duration": 0.0, "total_frames": 0, "has_audio": False, "audio_count": 0
            }

    @staticmethod
    def extract_first_frame(file_path: str) -> Optional[np.ndarray]:
        """
        Grabs just the first decoded frame as an (H, W, 3) RGB24 numpy array,
        so the GUI can show something in the comparison viewer the instant a
        video is loaded, without waiting for a processing job to start.
        Returns None if the file can't be probed/decoded.
        """
        try:
            meta = VideoMetadataReader.probe(file_path)
            w, h = meta["width"], meta["height"]
            if w <= 0 or h <= 0:
                return None

            ffmpeg = FFmpegLocator.get_ffmpeg_path()
            cmd = [
                ffmpeg, "-v", "error", "-i", file_path,
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=15, creationflags=_NO_WINDOW_FLAGS)
            expected_bytes = w * h * 3
            if len(res.stdout) < expected_bytes:
                return None
            return np.frombuffer(res.stdout[:expected_bytes], dtype=np.uint8).reshape((h, w, 3)).copy()
        except Exception as e:
            logger.warning(f"No se pudo extraer el primer fotograma para la vista previa: {e}")
            return None


class FFmpegFrameReader:
    """Streams raw uncompressed video frames via binary pipe directly from FFmpeg stdout."""

    def __init__(self, input_path: str, width: int, height: int, start_time: float = 0.0):
        self.input_path = input_path
        self.width = width
        self.height = height
        self.frame_bytes_size = width * height * 3  # RGB24 format
        self.start_time = start_time
        self.process: Optional[subprocess.Popen] = None
        self._open_pipe()

    def _open_pipe(self):
        ffmpeg = FFmpegLocator.get_ffmpeg_path()
        cmd = [ffmpeg, "-v", "error"]
        # Input seeking (-ss before -i) so a resumed job can skip straight to
        # its checkpointed frame instead of re-decoding everything before it.
        if self.start_time > 0:
            cmd += ["-ss", f"{self.start_time:.6f}"]
        cmd += [
            "-i", self.input_path,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-"
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 * self.frame_bytes_size,
            creationflags=_NO_WINDOW_FLAGS
        )

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads one RGB24 frame from pipe into a numpy array of shape (H, W, 3)."""
        if not self.process or not self.process.stdout:
            return None

        raw_bytes = self.process.stdout.read(self.frame_bytes_size)
        if len(raw_bytes) < self.frame_bytes_size:
            return None

        frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame

    def close(self):
        """Terminates reader process and closes pipe."""
        if self.process:
            try:
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                pass
            self.process = None


class FFmpegFrameWriter:
    """
    Streams raw processed frames into FFmpeg stdin, encoding with AMD AMF
    hardware acceleration (hevc_amf / h264_amf) and muxing original audio (-c:a copy).
    """

    def __init__(
        self,
        output_path: str,
        input_source_for_audio: str,
        width: int,
        height: int,
        fps: float,
        has_audio: bool = True,
        encoder: str = "hevc_amf",
        bitrate_mbps: int = 25,
        disable_bframes: bool = False
    ):
        self.output_path = output_path
        self.input_source_for_audio = input_source_for_audio
        self.width = width
        self.height = height
        self.fps = fps
        self.has_audio = has_audio
        self.encoder = encoder
        self.bitrate_mbps = bitrate_mbps
        # Used only when this writer produces one of several independently
        # encoded chunks later joined by stream copy (interpolation engine's
        # resumable segments): B-frame reordering makes a short segment's
        # last frame duration slightly ambiguous, which showed up as a small
        # timestamp jump at every segment join once concatenated. Disabling
        # B-frames for just those segments removes the ambiguity; a normal
        # single-shot encode (restoration/generative tabs) never sets this.
        self.disable_bframes = disable_bframes
        self.process: Optional[subprocess.Popen] = None
        self._start_encoding()

    def _start_encoding(self):
        ffmpeg = FFmpegLocator.get_ffmpeg_path()
        
        # Base input from pipe
        cmd = [
            ffmpeg,
            "-y",  # Overwrite output
            "-v", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps:.4f}",
            "-i", "-"  # Pipe stdin
        ]

        # Audio stream input mapping from source
        if self.has_audio and os.path.isfile(self.input_source_for_audio):
            cmd += ["-i", self.input_source_for_audio, "-map", "0:v:0", "-map", "1:a?"]
        else:
            cmd += ["-map", "0:v:0"]

        # Configure AMD AMF encoder or software fallback
        if self.encoder == "hevc_amf":
            cmd += [
                "-c:v", "hevc_amf",
                "-quality", "quality",
                "-rc", "cbr",
                "-b:v", f"{self.bitrate_mbps}M",
                "-maxrate", f"{self.bitrate_mbps + 10}M",
                "-bufsize", f"{self.bitrate_mbps * 2}M",
                "-pix_fmt", "yuv420p"
            ]
        elif self.encoder == "h264_amf":
            cmd += [
                "-c:v", "h264_amf",
                "-quality", "quality",
                "-rc", "cbr",
                "-b:v", f"{self.bitrate_mbps}M",
                "-pix_fmt", "yuv420p"
            ]
        elif self.encoder == "libx265":
            cmd += [
                "-c:v", "libx265",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p"
            ]
        else:  # libx264 fallback
            cmd += [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-pix_fmt", "yuv420p"
            ]

        if self.disable_bframes:
            cmd += ["-bf", "0"]

        # Preserve original audio without re-encoding
        if self.has_audio:
            cmd += ["-c:a", "copy"]

        cmd += ["-shortest", self.output_path]

        logger.info(f"Iniciando FFmpeg Writer con encoder: {self.encoder}")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 * self.width * self.height * 3,
            creationflags=_NO_WINDOW_FLAGS
        )

    def write_frame(self, frame: np.ndarray):
        """Writes an RGB24 frame (H, W, 3) to FFmpeg stdin pipe."""
        if not self.process or not self.process.stdin:
            return
        self.process.stdin.write(frame.tobytes())

    def close(self):
        """Closes stdin and waits for FFmpeg to finalize the container."""
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stderr:
                    self.process.stderr.close()
                self.process.wait(timeout=15)
            except Exception as e:
                logger.warning(f"Error al cerrar FFmpeg writer: {e}")
                self.process.kill()
            self.process = None
