"""
VRAM & Tiling Memory Manager
Implements dynamic patch division (Tiling), Cosine Feathering blending,
and dynamic 90% VRAM threshold protection with automatic tile size degradation.
"""

import math
import logging
import psutil
import torch
import numpy as np

logger = logging.getLogger("RadeonVideoAI.MemoryManager")

class MemoryManager:
    """
    Manages VRAM allocation, dynamic tiling, cosine feathering,
    and automatic degradation to prevent Out-Of-Memory (OOM) errors.
    """

    AVAILABLE_TILE_SIZES = [1024, 768, 512, 384, 256, 192, 128]

    def __init__(self, vram_total_mb: int = 16384, max_vram_ratio: float = 0.90, default_tile_size: int = 512, overlap: int = 48):
        self.vram_total_mb = vram_total_mb
        self.max_vram_ratio = max_vram_ratio  # 90% default limit
        self.default_tile_size = default_tile_size
        self.current_tile_size = default_tile_size
        self.overlap = overlap
        self._oom_occurred = False

    def get_current_memory_usage_ratio(self) -> float:
        """
        Estimates total system/GPU memory pressure.
        Uses psutil virtual memory as an active gauge for unified/DirectX shared memory.
        """
        try:
            mem = psutil.virtual_memory()
            return mem.percent / 100.0
        except Exception:
            return 0.50

    def check_vram_safety(self) -> bool:
        """
        Checks if current memory is within the safe 90% threshold.
        If exceeding threshold, triggers degradation.
        """
        ratio = self.get_current_memory_usage_ratio()
        if ratio > self.max_vram_ratio:
            logger.warning(f"Presión de memoria alta detectada ({ratio*100:.1f}% > {self.max_vram_ratio*100:.1f}%). Degradando tile size.")
            self.degrade_tile_size()
            return False
        return True

    def degrade_tile_size(self) -> int:
        """
        Automatically degrades current tile size to the next lower step to relieve VRAM pressure.
        """
        try:
            curr_idx = self.AVAILABLE_TILE_SIZES.index(self.current_tile_size)
            if curr_idx < len(self.AVAILABLE_TILE_SIZES) - 1:
                self.current_tile_size = self.AVAILABLE_TILE_SIZES[curr_idx + 1]
                logger.info(f"Tamaño de parche degradado automáticamente a: {self.current_tile_size}x{self.current_tile_size}")
            else:
                logger.warning("El tamaño de parche ya se encuentra en el mínimo soportado (128x128).")
        except ValueError:
            self.current_tile_size = 256
        return self.current_tile_size

    def reset_tile_size(self):
        """Resets tile size to configured default."""
        self.current_tile_size = self.default_tile_size

    def generate_cosine_weights(self, height: int, width: int, overlap: int, is_top: bool, is_bottom: bool, is_left: bool, is_right: bool) -> torch.Tensor:
        """
        Generates a 2D Cosine Feathering weight matrix W(y, x) for a tile:
        W(t) = 0.5 * (1 - cos(pi * t / overlap)) on overlapping edges, 1.0 in the center.
        Ensures continuous C1 smooth transitions across tile boundaries with zero seam artifacts.
        """
        # Horizontal 1D window
        wx = torch.ones(width, dtype=torch.float32)
        if overlap > 0:
            ramp = 0.5 * (1.0 - torch.cos(torch.linspace(0, math.pi, overlap, dtype=torch.float32)))
            if not is_left:
                wx[:overlap] = ramp
            if not is_right:
                wx[-overlap:] = torch.flip(ramp, dims=[0])

        # Vertical 1D window
        wy = torch.ones(height, dtype=torch.float32)
        if overlap > 0:
            ramp_y = 0.5 * (1.0 - torch.cos(torch.linspace(0, math.pi, overlap, dtype=torch.float32)))
            if not is_top:
                wy[:overlap] = ramp_y
            if not is_bottom:
                wy[-overlap:] = torch.flip(ramp_y, dims=[0])

        # 2D outer product: W(y, x) = wy[y] * wx[x]
        w2d = torch.outer(wy, wx)
        return w2d

    def split_into_tiles(self, tensor: torch.Tensor, tile_size: int = None, overlap: int = None):
        """
        Splits a frame tensor of shape (B, C, H, W) or (C, H, W) into overlapping tiles with metadata.
        Returns a list of tuples: (tile_tensor, y0, y1, x0, x1, (is_top, is_bottom, is_left, is_right))
        """
        if tensor.dim() == 4:
            c, h, w = tensor.shape[1], tensor.shape[2], tensor.shape[3]
            t = tensor.squeeze(0)
        else:
            c, h, w = tensor.shape[0], tensor.shape[1], tensor.shape[2]
            t = tensor

        ts = tile_size or self.current_tile_size
        ov = overlap or self.overlap

        # If image is smaller than tile size, return whole image as single tile
        if h <= ts and w <= ts:
            return [(t, 0, h, 0, w, (True, True, True, True))]

        stride = ts - ov
        y_steps = max(1, math.ceil((h - ov) / stride))
        x_steps = max(1, math.ceil((w - ov) / stride))

        tiles = []
        for yi in range(y_steps):
            y0 = yi * stride
            y1 = min(y0 + ts, h)
            if y1 - y0 < ts and y1 == h:
                y0 = max(0, h - ts)

            is_top = (y0 == 0)
            is_bottom = (y1 == h)

            for xi in range(x_steps):
                x0 = xi * stride
                x1 = min(x0 + ts, w)
                if x1 - x0 < ts and x1 == w:
                    x0 = max(0, w - ts)

                is_left = (x0 == 0)
                is_right = (x1 == w)

                tile = t[:, y0:y1, x0:x1]
                tiles.append((tile, y0, y1, x0, x1, (is_top, is_bottom, is_left, is_right)))

        return tiles

    def blend_tiles(self, upscaled_tiles, target_h: int, target_w: int, scale: int, overlap: int = None) -> torch.Tensor:
        """
        Reconstructs the full upscaled frame by accumulating upscaled tiles weighted by
        their 2D Cosine Feathering masks, then normalizing by the accumulated weights.
        Guarantees exact photometric consistency and seamless blending.
        """
        ov = (overlap or self.overlap) * scale
        c = upscaled_tiles[0][0].shape[0]

        device = upscaled_tiles[0][0].device
        canvas = torch.zeros((c, target_h, target_w), dtype=torch.float32, device=device)
        weight_canvas = torch.zeros((1, target_h, target_w), dtype=torch.float32, device=device)

        for tile, y0, y1, x0, x1, flags in upscaled_tiles:
            is_top, is_bottom, is_left, is_right = flags
            th, tw = tile.shape[1], tile.shape[2]

            # Generate cosine feathering mask on same device
            mask = self.generate_cosine_weights(th, tw, ov, is_top, is_bottom, is_left, is_right).to(device)
            mask_expanded = mask.unsqueeze(0)  # (1, th, tw)

            # Accumulate weighted tile
            canvas[:, y0*scale : y0*scale + th, x0*scale : x0*scale + tw] += tile * mask_expanded
            weight_canvas[:, y0*scale : y0*scale + th, x0*scale : x0*scale + tw] += mask_expanded

        # Normalize to prevent seam boundaries
        weight_canvas = torch.clamp(weight_canvas, min=1e-6)
        output = canvas / weight_canvas
        return output

