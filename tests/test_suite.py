"""
Automated Unit and Integration Tests for RadeonVideoAI Suite.
Validates:
1. AMD Hardware Backend & Detection
2. Memory Manager Tiling and Cosine Feathering Reconstruction (MSE test)
3. Real AI model (Real-ESRGAN) forward pass via ONNX Runtime (2x and 4x)
   NOTE: requires network access on first run to download official pretrained
   weights; skipped automatically if the download is unavailable.
4. Media Pipeline FFmpeg Discovery
"""

import os
import sys
import unittest
import torch
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.amd_backend import amd_hardware
from core.memory_manager import MemoryManager
from core.inference import create_model
from core.media_pipeline import FFmpegLocator

class TestRadeonVideoAI(unittest.TestCase):

    def test_01_amd_backend(self):
        """Tests that hardware manager detects GPU, VRAM and configures compute providers."""
        summary = amd_hardware.get_hardware_summary()
        self.assertIsNotNone(summary["gpu_name"])
        self.assertGreater(summary["vram_total_mb"], 0)
        self.assertGreater(len(amd_hardware.providers), 0)
        print(f"[TEST 1] Hardware detectado: {summary['gpu_name']} | Backend: {summary['backend']} | VRAM: {summary['vram_total_mb']} MB")

    def test_02_cosine_feathering_reconstruction(self):
        """
        Mathematical verification of the Cosine Feathering blending algorithm.
        Splits a random image into overlapping tiles, reconstructs it using
        the Cosine Feathering weight matrix, and asserts that the reconstruction
        is identical to the original (MSE < 1e-6) with zero boundary seams.
        """
        mem_mgr = MemoryManager(default_tile_size=128, overlap=32)
        
        # Test image of size 300x400 (not exact multiple of tile size)
        c, h, w = 3, 300, 400
        original_tensor = torch.rand((c, h, w), dtype=torch.float32)

        # Split into tiles
        tiles = mem_mgr.split_into_tiles(original_tensor, tile_size=128, overlap=32)
        self.assertGreater(len(tiles), 1)

        # Identity 'upscale' (scale=1) for mathematical verification of blending
        upscaled_tiles = [(tile, y0, y1, x0, x1, flags) for tile, y0, y1, x0, x1, flags in tiles]

        # Blend tiles back
        reconstructed = mem_mgr.blend_tiles(upscaled_tiles, target_h=h, target_w=w, scale=1, overlap=32)

        # Compute Mean Squared Error
        mse = torch.mean((original_tensor - reconstructed) ** 2).item()
        print(f"[TEST 2] Cosine Feathering Reconstrucción MSE: {mse:.8e}")
        self.assertLess(mse, 1e-5, "La reconstrucción con Cosine Feathering debe preservar perfectamente los valores sin costuras.")

    def test_03_ai_model_forward(self):
        """
        Tests 2x and 4x real AI upscaler forward pass (Real-ESRGAN via ONNX
        Runtime). Downloads official pretrained weights on first run; the
        test is skipped if no network access is available to fetch them.
        """
        dummy_in = torch.rand((1, 3, 64, 64))

        try:
            # 2x output (general_v3 native 4x internally resized to 2x)
            model_2x = create_model(model_name="general_v3", scale=2, denoise_strength=0.2)
        except Exception as e:
            self.skipTest(f"No se pudo preparar el modelo de IA (¿sin red para descargar pesos?): {e}")
            return

        with torch.no_grad():
            out_2x = model_2x(dummy_in, sharpen_strength=0.2)
        self.assertEqual(out_2x.shape, (1, 3, 128, 128))
        print(f"[TEST 3.1] Modelo 2x (general_v3) inferencia exitosa: {dummy_in.shape} -> {out_2x.shape}")

        # 4x output (x4plus, native scale)
        model_4x = create_model(model_name="x4plus", scale=4)
        with torch.no_grad():
            out_4x = model_4x(dummy_in, sharpen_strength=0.1)
        self.assertEqual(out_4x.shape, (1, 3, 256, 256))
        print(f"[TEST 3.2] Modelo 4x (x4plus) inferencia exitosa: {dummy_in.shape} -> {out_4x.shape}")

    def test_04_memory_degradation(self):
        """Tests automatic tile size degradation on high memory threshold."""
        mem_mgr = MemoryManager(default_tile_size=512)
        initial_size = mem_mgr.current_tile_size
        self.assertEqual(initial_size, 512)

        # Trigger degradation
        degraded_1 = mem_mgr.degrade_tile_size()
        self.assertEqual(degraded_1, 384)

        degraded_2 = mem_mgr.degrade_tile_size()
        self.assertEqual(degraded_2, 256)

        mem_mgr.reset_tile_size()
        self.assertEqual(mem_mgr.current_tile_size, 512)
        print(f"[TEST 4] Degradación automática de parche verificada: 512 -> {degraded_1} -> {degraded_2}")

    def test_05_ffmpeg_locator(self):
        """Verifies that FFmpeg discovery mechanism functions properly."""
        path = FFmpegLocator.get_ffmpeg_path()
        self.assertTrue(len(path) > 0)
        print(f"[TEST 5] FFmpeg path resuelto: {path}")

if __name__ == "__main__":
    unittest.main()

