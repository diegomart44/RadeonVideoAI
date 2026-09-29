"""
Real-time Hardware & Processing Telemetry Status Bar.
Displays GPU device, VRAM usage, FPS speed, ETA, and progress bar.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar
from core.amd_backend import amd_hardware

class StatusBarWidget(QWidget):
    """Bottom telemetry status bar displaying real-time AMD GPU metrics and job progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(38)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 4, 14, 4)
        layout.setSpacing(16)

        # Compute backend indicator (GPU name/VRAM already shown in the header)
        hw_info = amd_hardware.get_hardware_summary()
        self.lbl_gpu = QLabel(f"⚙ {hw_info['backend']}")
        self.lbl_gpu.setStyleSheet("font-weight: 600; color: #00D2FF;")
        self.lbl_gpu.setToolTip(f"GPU: {hw_info['gpu_name']}")
        layout.addWidget(self.lbl_gpu)

        # Status / phase label (model loading, downloads, etc.)
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #FFD166;")
        layout.addWidget(self.lbl_status)

        # VRAM Gauge
        self.lbl_vram = QLabel(f"VRAM: -- / {hw_info['vram_total_mb'] // 1024:.1f} GB")
        self.lbl_vram.setStyleSheet("color: #A0A6B2;")
        layout.addWidget(self.lbl_vram)

        # FPS speed
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setStyleSheet("color: #FFFFFF; font-weight: 500;")
        layout.addWidget(self.lbl_fps)

        # ETA
        self.lbl_eta = QLabel("ETA: --:--:--")
        self.lbl_eta.setStyleSheet("color: #FFFFFF; font-weight: 500;")
        layout.addWidget(self.lbl_eta)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(220)
        layout.addWidget(self.progress_bar)

        # Frame counter
        self.lbl_frames = QLabel("0 / 0")
        self.lbl_frames.setStyleSheet("color: #A0A6B2; font-size: 11px;")
        layout.addWidget(self.lbl_frames)

    def update_telemetry(self, current_frame: int, total_frames: int, fps: float, eta_str: str, vram_info: dict):
        """Updates UI status widgets from processing worker telemetry."""
        pct = int((current_frame / total_frames * 100)) if total_frames > 0 else 0
        self.progress_bar.setValue(pct)
        self.lbl_frames.setText(f"{current_frame} / {total_frames}")
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_eta.setText(f"ETA: {eta_str}")

        vram_pct = vram_info.get("vram_used_pct", 0)
        tile_sz = vram_info.get("tile_size", 512)
        total_gb = amd_hardware.vram_total_mb / 1024.0
        used_gb = (vram_pct / 100.0) * total_gb
        # tile_size is only meaningful for the tiled engines (Reescalar/
        # Generar); the interpolation engine processes full frames and
        # reports 0/None here, so the "[Tile: N]" suffix is omitted rather
        # than showing a confusing "[Tile: 0]".
        tile_suffix = f" [Tile: {tile_sz}]" if tile_sz else ""
        self.lbl_vram.setText(f"VRAM: {used_gb:.1f} / {total_gb:.1f} GB ({vram_pct}%){tile_suffix}")

    def reset_telemetry(self):
        self.progress_bar.setValue(0)
        self.lbl_frames.setText("0 / 0")
        self.lbl_fps.setText("FPS: --")
        self.lbl_eta.setText("ETA: --:--:--")
        self.lbl_status.setText("")

    def set_status(self, message: str):
        self.lbl_status.setText(message)

