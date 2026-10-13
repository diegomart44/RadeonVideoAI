"""
Integration Test: End-to-End Synthetic Audio Restoration.
Generates a short synthetic video with a noisy tone track, runs it through
the AI audio restoration pipeline (VoiceFixer), and verifies that the
output video keeps its original video stream while its audio track has
been replaced by a restored one.

NOTE: downloads ~600MB of VoiceFixer pretrained checkpoints on first run;
the test is skipped automatically if that download is unavailable.
"""

import os
import sys
import subprocess
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.media_pipeline import FFmpegLocator, VideoMetadataReader
from core.audio_engine import AudioRestorationEngine


class TestAudioRestorationPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.join(BASE_DIR, "tests", "temp")
        os.makedirs(self.test_dir, exist_ok=True)
        self.input_video = os.path.join(self.test_dir, "test_audio_input.mp4")
        self.output_video = os.path.join(self.test_dir, "test_audio_output.mp4")

        # Short synthetic video (64x64, 10fps, 2s) with a noisy sine-tone
        # audio track — short enough that even a single ~30s-capped
        # VoiceFixer segment covers the whole thing in one chunk.
        ffmpeg = FFmpegLocator.get_ffmpeg_path()
        cmd = [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10",
            "-f", "lavfi", "-i",
            "sine=frequency=220:duration=2,volume=0.3,"
            "aeval=val(0)+0.05*random(0):c=mono",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest", self.input_video
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def tearDown(self):
        for f in [self.input_video, self.output_video]:
            if os.path.isfile(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_end_to_end_audio_restoration(self):
        # Isolate the network-dependent step (model construction, which
        # downloads ~600MB of pretrained checkpoints on first run) from the
        # actual processing logic below — only a failure *here* should skip
        # the test; a bug in the processing logic itself must still fail it,
        # the same way test_03_ai_model_forward only skips around
        # create_model(), not around the inference that follows.
        try:
            from core.audio_models import AudioRestorationModel
            AudioRestorationModel()
        except Exception as e:
            self.skipTest(f"No se pudo preparar el modelo de audio (¿sin red para descargar los pesos?): {e}")
            return

        engine = AudioRestorationEngine()

        progress_records = []
        def on_prog(curr, tot, rate, eta, telemetry):
            progress_records.append(curr)

        status_records = []
        def on_status(msg):
            status_records.append(msg)

        success = engine.process_audio(
            input_path=self.input_video,
            output_path=self.output_video,
            mode=0,
            audio_bitrate_kbps=128,
            progress_callback=on_prog,
            status_callback=on_status
        )

        self.assertTrue(success, "La restauración de audio debe finalizar exitosamente.")
        self.assertTrue(os.path.isfile(self.output_video), "El archivo de salida debe existir.")
        self.assertGreater(os.path.getsize(self.output_video), 1000, "El video de salida debe contener datos.")
        self.assertGreater(len(progress_records), 0, "Debe haber reportado progreso de al menos un segmento.")

        meta = VideoMetadataReader.probe(self.output_video)
        print(f"[TEST AUDIO E2E] Video de salida: {meta['width']}x{meta['height']}, Audio: {meta['has_audio']}, segmentos: {len(progress_records)}")
        self.assertEqual(meta["width"], 64, "El video no debe re-codificarse (misma resolución).")
        self.assertEqual(meta["height"], 64)
        self.assertTrue(meta["has_audio"], "El audio restaurado debe estar presente en la salida.")


if __name__ == "__main__":
    unittest.main()
