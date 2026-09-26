"""
Main Application Window for RadeonVideoAI Desktop Suite.
Connects UI controls, real-time side-by-side viewport, and async processing worker.
"""

import os
import subprocess
import threading
import numpy as np
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QMessageBox, QLabel
)

from core.engine import VideoProcessingEngine
from core.generative_engine import GenerativeVideoEngine
from core.amd_backend import amd_hardware
from core.media_pipeline import VideoMetadataReader
from gui.theme import DARK_THEME_QSS
from gui.widgets.preview_viewport import SideBySideViewport
from gui.widgets.controls_panel import ControlsPanel
from gui.widgets.status_bar import StatusBarWidget


class FirstFrameWorker(QObject):
    """
    Grabs a video's first frame off the GUI thread so the comparison viewer
    can show something immediately after a file is selected, without
    freezing the UI while ffmpeg decodes it (relevant for network drives or
    unusual codecs where this can take a moment).

    Runs on a plain Python thread rather than QThread — see the note on
    ProcessingWorker below for why.
    """

    frame_ready = pyqtSignal(object)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        frame = VideoMetadataReader.extract_first_frame(self.path)
        if frame is not None:
            self.frame_ready.emit(frame)


class ProcessingWorker(QObject):
    """
    Asynchronous worker executing the AI video enhancement engine.

    IMPORTANT: this runs the job on a plain Python `threading.Thread`, NOT a
    QThread. Running the ONNX Runtime DirectML inference on a QThread was
    found to deadlock on the very first frame: DirectML/D3D12 calls appear
    to need COM apartment/message-pump behavior that a bare QThread doesn't
    provide the way the main Qt thread does, so the GPU call hangs forever
    and the job looks frozen with no error. A plain thread does not have
    this problem, and pyqtSignal emission across threads works the same way
    regardless of whether the emitting thread is a QThread — Qt determines
    queuing by the *receiver's* thread affinity, not the sender's.
    """

    progress_signal = pyqtSignal(int, int, float, str, dict)
    # NOTE: declared as `object`, not `np.ndarray`. numpy arrays are not a
    # registered Qt metatype, and this signal crosses threads (worker ->
    # GUI), so Qt must queue it; without `object` this can silently drop or
    # crash the process natively instead of raising a catchable exception.
    preview_signal = pyqtSignal(object, object)
    finished_signal = pyqtSignal(bool, str)
    status_signal = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.engine = VideoProcessingEngine()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            success = self.engine.process_video(
                input_path=self.config["input_path"],
                output_path=self.config["output_path"],
                scale=self.config["scale"],
                model_name=self.config.get("model_name", "general_v3"),
                tile_size=self.config["tile_size"],
                overlap=self.config["overlap"],
                denoise=self.config["denoise"],
                sharpen=self.config["sharpen"],
                encoder=self.config["encoder"],
                bitrate_mbps=self.config["bitrate_mbps"],
                vram_safety=self.config["vram_safety"],
                progress_callback=self._on_progress,
                preview_callback=self._on_preview,
                status_callback=self._on_status
            )
            self.finished_signal.emit(success, "Procesamiento completado con éxito." if success else "Procesamiento cancelado por el usuario.")
        except Exception as e:
            self.finished_signal.emit(False, f"Error en el motor: {str(e)}")

    def _on_progress(self, curr, total, fps, eta, vram_info):
        self.progress_signal.emit(curr, total, fps, eta, vram_info)

    def _on_preview(self, orig_rgb, ai_rgb):
        # Explicit .copy() so the arrays crossing to the GUI thread are
        # plain, independently-owned numpy buffers with no lingering tie to
        # any PyTorch/ONNX Runtime-managed memory the worker thread keeps
        # allocating from concurrently.
        self.preview_signal.emit(orig_rgb.copy(), ai_rgb.copy())

    def _on_status(self, message: str):
        self.status_signal.emit(message)

    def pause(self):
        self.engine.pause()

    def resume(self):
        self.engine.resume()

    def cancel(self):
        self.engine.cancel()


class GenerativeProcessingWorker(QObject):
    """
    Asynchronous worker for the generative (Stable Diffusion img2img) video
    engine. Intentionally its own class, not a subclass/variant of
    ProcessingWorker, mirroring core/generative_engine.py's separation from
    core/engine.py: a bug in one pipeline's threading must never touch the
    other. Same plain-thread pattern as ProcessingWorker (see its docstring
    for why QThread specifically deadlocks with DirectML).
    """

    progress_signal = pyqtSignal(int, int, float, str, dict)
    preview_signal = pyqtSignal(object, object)
    finished_signal = pyqtSignal(bool, str)
    status_signal = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.engine = GenerativeVideoEngine()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            success = self.engine.process_video(
                input_path=self.config["input_path"],
                output_path=self.config["output_path"],
                steps=self.config["steps"],
                strength=self.config["strength"],
                overlap=self.config["overlap"],
                encoder=self.config["encoder"],
                bitrate_mbps=self.config["bitrate_mbps"],
                vram_safety=self.config["vram_safety"],
                progress_callback=self._on_progress,
                preview_callback=self._on_preview,
                status_callback=self._on_status
            )
            self.finished_signal.emit(success, "Generación completada con éxito." if success else "Generación cancelada por el usuario.")
        except Exception as e:
            self.finished_signal.emit(False, f"Error en el motor generativo: {str(e)}")

    def _on_progress(self, curr, total, fps, eta, vram_info):
        self.progress_signal.emit(curr, total, fps, eta, vram_info)

    def _on_preview(self, orig_rgb, ai_rgb):
        self.preview_signal.emit(orig_rgb.copy(), ai_rgb.copy())

    def _on_status(self, message: str):
        self.status_signal.emit(message)

    def pause(self):
        self.engine.pause()

    def resume(self):
        self.engine.resume()

    def cancel(self):
        self.engine.cancel()


class MainWindow(QMainWindow):
    """Main window integrating the Topaz-style interface, live preview, and pipeline."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RadeonVideoAI Suite — Generative Video AI & Upscaling")
        self.resize(1360, 860)
        self.setMinimumSize(1024, 640)
        self.setStyleSheet(DARK_THEME_QSS)
        self.setAcceptDrops(True)

        self.worker = None
        self.gen_worker = None

        self._build_ui()

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("AppHeader")
        header.setFixedHeight(56)
        lay = QHBoxLayout(header)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(4)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        lbl_title = QLabel("⚡ RadeonVideoAI")
        lbl_title.setObjectName("AppTitle")
        lbl_subtitle = QLabel("Restauración y reescalado de vídeo con IA")
        lbl_subtitle.setObjectName("AppSubtitle")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_subtitle)
        lay.addLayout(title_box)
        lay.addStretch(1)

        gpu_name = amd_hardware.gpu_name
        vram_gb = amd_hardware.vram_total_mb / 1024.0
        lbl_gpu_badge = QLabel(f"{gpu_name} · {vram_gb:.0f} GB")
        lbl_gpu_badge.setObjectName("AppBadge")
        lay.addWidget(lbl_gpu_badge)

        return header

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addWidget(self._build_header())

        # Splitter between Viewport and Controls Panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # 1. Left: Real-time Side-by-Side Viewport
        self.viewport = SideBySideViewport()
        splitter.addWidget(self.viewport)

        # 2. Right: Controls Panel
        self.controls = ControlsPanel()
        self.controls.start_requested.connect(self._start_processing)
        self.controls.pause_requested.connect(self._pause_processing)
        self.controls.resume_requested.connect(self._resume_processing)
        self.controls.cancel_requested.connect(self._cancel_processing)
        self.controls.generate_requested.connect(self._start_generation)
        self.controls.generate_pause_requested.connect(self._pause_generation)
        self.controls.generate_resume_requested.connect(self._resume_generation)
        self.controls.generate_cancel_requested.connect(self._cancel_generation)
        self.controls.file_selected.connect(self._on_file_selected)
        splitter.addWidget(self.controls)

        # Adjust stretch factors
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter)

        # 3. Bottom: Status bar telemetry
        self.status_bar = StatusBarWidget()
        main_layout.addWidget(self.status_bar)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isfile(path):
                self.controls.set_active_tab_input_path(path)

    def _on_file_selected(self, path: str):
        """Shows the video's first frame in the comparison viewer right away,
        so Split/Original/Enhanced and zoom/pan are immediately testable
        instead of only working once a processing job starts sending frames."""
        self._frame_worker = FirstFrameWorker(path)
        self._frame_worker.frame_ready.connect(self._on_first_frame_ready)
        self._frame_worker.start()

    def _on_first_frame_ready(self, frame):
        # Both slots show the original frame until a job actually processes
        # it; the "enhanced" side will start reflecting real AI output once
        # processing begins.
        self.viewport.update_preview_frames(frame, frame)

    def _start_processing(self, config: dict):
        self.status_bar.reset_telemetry()
        # Show the "generating" banner immediately — including during the
        # initial model download/compile phase, before any AI frame exists —
        # so it is never mistaken for the app doing nothing.
        self.viewport.set_processing_state(True)
        self.worker = ProcessingWorker(config)
        self.worker.progress_signal.connect(self.status_bar.update_telemetry)
        self.worker.progress_signal.connect(self._on_viewport_progress)
        self.worker.preview_signal.connect(self.viewport.update_preview_frames)
        self.worker.finished_signal.connect(self._on_processing_finished)
        self.worker.status_signal.connect(self.status_bar.set_status)
        self.worker.start()

    def _on_viewport_progress(self, curr, total, fps, eta, vram_info):
        text = f"Generando fotograma {curr} / {total} · {fps:.1f} fps · ETA {eta}"
        self.viewport.set_processing_state(True, text)

    def _pause_processing(self):
        if self.worker:
            self.worker.pause()

    def _resume_processing(self):
        if self.worker:
            self.worker.resume()

    def _cancel_processing(self):
        if self.worker:
            self.worker.cancel()

    def _on_processing_finished(self, success: bool, message: str):
        self.controls.set_processing_finished()
        self.viewport.set_processing_state(False)
        if success:
            out_file = self.controls.txt_output.text().strip()
            reply = QMessageBox.information(
                self, "Procesamiento Finalizado",
                f"{message}\n\n¿Deseas abrir la carpeta con el video procesado?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes and os.path.isfile(out_file):
                folder = os.path.dirname(os.path.abspath(out_file))
                subprocess.run(f'explorer /select,"{os.path.abspath(out_file)}"', shell=True)
        else:
            QMessageBox.warning(self, "Aviso de Procesamiento", message)

    # =====================================================================
    # Generar (IA Generativa) — fully independent flow, own worker/engine
    # =====================================================================
    def _start_generation(self, config: dict):
        self.status_bar.reset_telemetry()
        self.viewport.set_processing_state(True)
        self.gen_worker = GenerativeProcessingWorker(config)
        self.gen_worker.progress_signal.connect(self.status_bar.update_telemetry)
        self.gen_worker.progress_signal.connect(self._on_viewport_progress)
        self.gen_worker.preview_signal.connect(self.viewport.update_preview_frames)
        self.gen_worker.finished_signal.connect(self._on_generation_finished)
        self.gen_worker.status_signal.connect(self.status_bar.set_status)
        self.gen_worker.start()

    def _pause_generation(self):
        if self.gen_worker:
            self.gen_worker.pause()

    def _resume_generation(self):
        if self.gen_worker:
            self.gen_worker.resume()

    def _cancel_generation(self):
        if self.gen_worker:
            self.gen_worker.cancel()

    def _on_generation_finished(self, success: bool, message: str):
        self.controls.set_generation_finished()
        self.viewport.set_processing_state(False)
        if success:
            out_file = self.controls.txt_gen_output.text().strip()
            reply = QMessageBox.information(
                self, "Generación Finalizada",
                f"{message}\n\n¿Deseas abrir la carpeta con el video generado?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes and os.path.isfile(out_file):
                subprocess.run(f'explorer /select,"{os.path.abspath(out_file)}"', shell=True)
        else:
            QMessageBox.warning(self, "Aviso de Generación", message)

