"""
Integration Test: End-to-End Synthetic Video Upscaling.
Generates a 1-second video with audio, runs 2x upscaling through the pipeline,
and verifies output file generation and audio preservation.
"""

import os
import sys
import subprocess
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.media_pipeline import FFmpegLocator, VideoMetadataReader
from core.engine import VideoProcessingEngine

class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.join(BASE_DIR, "tests", "temp")
        os.makedirs(self.test_dir, exist_ok=True)
        self.input_video = os.path.join(self.test_dir, "test_input.mp4")
        self.output_video = os.path.join(self.test_dir, "test_output_2x.mp4")

        # Generate 1-second synthetic video (128x128, 30fps) with a sine beep audio track
        ffmpeg = FFmpegLocator.get_ffmpeg_path()
        cmd = [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=128x128:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            self.input_video
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def tearDown(self):
        # Cleanup test files
        for f in [self.input_video, self.output_video]:
            if os.path.isfile(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_end_to_end_upscale(self):
        engine = VideoProcessingEngine()
        
        frames_captured = []
        def on_preview(orig, ai):
            frames_captured.append((orig.shape, ai.shape))

        progress_records = []
        def on_prog(curr, tot, fps, eta, vram):
            progress_records.append(curr)

        success = engine.process_video(
            input_path=self.input_video,
            output_path=self.output_video,
            scale=2,
            tile_size=128,
            overlap=32,
            denoise=0.1,
            sharpen=0.2,
            encoder="libx264",  # Universal test encoder
            bitrate_mbps=10,
            vram_safety=0.90,
            progress_callback=on_prog,
            preview_callback=on_preview
        )

        self.assertTrue(success, "El procesamiento debe finalizar exitosamente.")
        self.assertTrue(os.path.isfile(self.output_video), "El archivo de salida debe existir.")
        self.assertGreater(os.path.getsize(self.output_video), 1000, "El video de salida debe contener datos.")

        # Probe output video to verify 2x upscale dimensions
        meta = VideoMetadataReader.probe(self.output_video)
        print(f"[TEST E2E] Video original: 128x128 -> Video de salida: {meta['width']}x{meta['height']}, Audio: {meta['has_audio']}")
        self.assertEqual(meta["width"], 256)
        self.assertEqual(meta["height"], 256)
        self.assertTrue(meta["has_audio"], "El audio original debe haberse preservado intacto.")
        self.assertGreater(len(frames_captured), 0, "Debe haber emitido fotogramas para la vista previa.")

if __name__ == "__main__":
    unittest.main()

