"""
Topaz-style Comparison Viewport with 30 FPS Throttling for Real-Time AI Video Preview.
Supports split / original-only / enhanced-only view modes, mouse-wheel zoom,
click-and-drag panning, and live updates while a job is processing.
Decoupled from the inference thread to prevent GPU and UI stalls.
"""

import threading
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPixmap, QColor, QPen, QFont, QBrush
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QButtonGroup, QSizePolicy

MIN_ZOOM = 1.0
MAX_ZOOM = 8.0


class _ComparisonCanvas(QWidget):
    """Custom-painted canvas doing the actual side-by-side/zoom/pan rendering."""

    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)

        self.view_mode = "split"  # "split" | "original" | "enhanced"
        self.split_ratio = 0.50
        self.zoom = 1.0
        self.pan_offset = QPointF(0.0, 0.0)

        self._dragging_splitter = False
        self._panning = False
        self._pan_start_mouse = QPoint()
        self._pan_start_offset = QPointF(0.0, 0.0)

        self._lock = threading.Lock()
        self._latest_orig_np = None
        self._latest_ai_np = None
        self._has_new_frame = False

        self._pixmap_orig = None
        self._pixmap_ai = None

        self._processing_active = False
        self._progress_text = ""
        self._pulse_tick = 0

        # IMPORTANT: this timer is intentionally NOT a continuously-repeating
        # 30fps timer. An always-on Qt timer competes for the GIL with the
        # worker thread's CPU-bound tensor operations (tiling/blending) badly
        # enough to turn sub-second processing into a multi-minute stall —
        # measured directly, not a theoretical concern. Instead it's a
        # single-shot timer that only fires (debounced to ~15fps) when a new
        # frame has actually arrived, and stays completely idle otherwise.
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(66)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._on_render_tick)

        # Separate, low-frequency timer for the "generating..." pulse dots,
        # active only while a job is actually running.
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(500)
        self._pulse_timer.timeout.connect(self._on_pulse_tick)

    # --- Public API -------------------------------------------------
    def update_preview_frames(self, orig_rgb: np.ndarray, ai_rgb: np.ndarray):
        """Called by worker thread. Stores raw numpy frames without blocking;
        actual QPixmap conversion happens on the GUI thread shortly after,
        debounced by _render_timer rather than an always-on poll loop."""
        with self._lock:
            self._latest_orig_np = orig_rgb
            self._latest_ai_np = ai_rgb
            self._has_new_frame = True
        if not self._render_timer.isActive():
            self._render_timer.start()

    def set_view_mode(self, mode: str):
        self.view_mode = mode
        self.update()

    def reset_zoom(self):
        self.zoom = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self.zoom_changed.emit(self.zoom)
        self.update()

    def set_zoom(self, zoom: float):
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        self.zoom_changed.emit(self.zoom)
        self.update()

    def set_processing_state(self, active: bool, progress_text: str = ""):
        """Toggles the 'generating' overlay banner. Called when a job starts
        (with no text yet, e.g. while the model downloads/compiles) and on
        every progress tick (with a live frame counter), so it's always
        obvious a new video is actively being generated — even before the
        first AI frame has streamed back to update the preview images."""
        self._processing_active = active
        self._progress_text = progress_text
        if active and not self._pulse_timer.isActive():
            self._pulse_timer.start()
        elif not active:
            self._pulse_timer.stop()
        self.update()

    # --- Rendering ----------------------------------------------------
    def _on_pulse_tick(self):
        self._pulse_tick += 1
        self.update()

    def _on_render_tick(self):
        if not self._has_new_frame:
            return

        with self._lock:
            orig_np = self._latest_orig_np
            ai_np = self._latest_ai_np
            self._has_new_frame = False

        if orig_np is not None and ai_np is not None:
            # .copy() forces QImage to own its pixel buffer immediately,
            # rather than wrapping the numpy array's memory directly. That
            # numpy array is reused/overwritten by the reader/writer on the
            # next frame, so without a copy the QImage could alias memory
            # that changes out from under it before it's actually painted.
            h1, w1, _ = orig_np.shape
            self._pixmap_orig = QPixmap.fromImage(
                QImage(orig_np.data, w1, h1, 3 * w1, QImage.Format.Format_RGB888).copy()
            )
            h2, w2, _ = ai_np.shape
            self._pixmap_ai = QPixmap.fromImage(
                QImage(ai_np.data, w2, h2, 3 * w2, QImage.Format.Format_RGB888).copy()
            )
            self.update()

    def _base_render_rect(self, w: int, h: int) -> QRectF:
        """Fit-to-window rect at zoom=1.0, before applying zoom/pan."""
        img_w = self._pixmap_ai.width()
        img_h = self._pixmap_ai.height()
        aspect = img_w / img_h
        target_w = w
        target_h = int(w / aspect)
        if target_h > h:
            target_h = h
            target_w = int(h * aspect)
        return QRectF(0, 0, target_w, target_h)

    def _current_render_rect(self, w: int, h: int) -> QRectF:
        base = self._base_render_rect(w, h)
        target_w = base.width() * self.zoom
        target_h = base.height() * self.zoom
        offset_x = (w - target_w) / 2.0 + self.pan_offset.x()
        offset_y = (h - target_h) / 2.0 + self.pan_offset.y()
        return QRectF(offset_x, offset_y, target_w, target_h)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#101216"))

        if self._pixmap_orig is None or self._pixmap_ai is None:
            self._draw_empty_state(painter, w, h)
            if self._processing_active:
                self._draw_processing_banner(painter, w, h)
            return

        painter.setClipRect(0, 0, w, h)
        render_rect = self._current_render_rect(w, h)

        if self.view_mode == "original":
            painter.drawPixmap(render_rect.toRect(), self._pixmap_orig)
            self._draw_badge(painter, "ORIGINAL", 16, 16, QColor("#1E222B"), QColor("#CFD3DC"))
        elif self.view_mode == "enhanced":
            painter.drawPixmap(render_rect.toRect(), self._pixmap_ai)
            self._draw_badge(painter, "MEJORADO CON IA", 16, 16, QColor("#E0115F"), QColor("#FFFFFF"))
        else:
            self._paint_split(painter, render_rect)

        if self.zoom > 1.01:
            zoom_text = f"🔍 {int(self.zoom * 100)}%"
            self._draw_badge(painter, zoom_text, w - 96, h - 42, QColor("#1E222B"), QColor("#00D2FF"), width=80)

        if self._processing_active:
            self._draw_processing_banner(painter, w, h)

    def _draw_processing_banner(self, painter: QPainter, w: int, h: int):
        """Persistent 'generating' banner shown for the whole duration of a
        job — including the initial model-download/compile phase before any
        frame has arrived — so it is unmistakable that video generation is
        actively running."""
        dots = "." * (1 + self._pulse_tick % 3)
        text = f"🎬 Generando video con IA{dots}"
        if self._progress_text:
            text = f"🎬 {self._progress_text}"

        painter.save()
        font = QFont("Segoe UI", 10, QFont.Weight.DemiBold)
        painter.setFont(font)
        text_w = painter.fontMetrics().horizontalAdvance(text) + 36
        rect = QRectF((w - text_w) / 2.0, h - 56, text_w, 34)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(15, 17, 21, 235)))
        painter.drawRoundedRect(rect, 17, 17)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#E0115F")))
        painter.drawRoundedRect(rect, 17, 17)

        painter.setPen(QColor("#00D2FF"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def _paint_split(self, painter: QPainter, render_rect: QRectF):
        offset_x, offset_y = render_rect.x(), render_rect.y()
        target_w, target_h = render_rect.width(), render_rect.height()
        split_x = offset_x + target_w * self.split_ratio

        painter.save()
        painter.setClipRect(QRectF(max(offset_x, 0), 0, split_x - max(offset_x, 0), self.height()))
        painter.drawPixmap(render_rect.toRect(), self._pixmap_orig)
        painter.restore()

        painter.save()
        painter.setClipRect(QRectF(split_x, 0, self.width() - split_x, self.height()))
        painter.drawPixmap(render_rect.toRect(), self._pixmap_ai)
        painter.restore()

        pen = QPen(QColor("#FFFFFF"), 2)
        painter.setPen(pen)
        vis_top = max(offset_y, 0)
        vis_bottom = min(offset_y + target_h, self.height())
        painter.drawLine(int(split_x), int(vis_top), int(split_x), int(vis_bottom))

        handle_y = (vis_top + vis_bottom) / 2.0
        painter.setBrush(QBrush(QColor("#E0115F")))
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.drawEllipse(QPointF(split_x, handle_y), 12, 12)
        painter.setPen(QPen(QColor("#FFFFFF"), 1))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(QRectF(split_x - 10, handle_y - 8, 20, 16), Qt.AlignmentFlag.AlignCenter, "◀ ▶")

        self._draw_badge(painter, "ORIGINAL", 16, 16, QColor("#1E222B"), QColor("#CFD3DC"))
        self._draw_badge(painter, "MEJORADO CON IA", self.width() - 176, 16, QColor("#E0115F"), QColor("#FFFFFF"))

    def _draw_empty_state(self, painter: QPainter, w: int, h: int):
        box_w, box_h = min(420, w - 48), min(220, h - 48)
        box_x = (w - box_w) // 2
        box_y = (h - box_h) // 2
        box_rect = QRectF(box_x, box_y, box_w, box_h)

        painter.save()
        pen = QPen(QColor("#2E3542"), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor("#14161C")))
        painter.drawRoundedRect(box_rect, 16, 16)
        painter.restore()

        icon_rect = QRectF(box_x, box_y + 24, box_w, 48)
        painter.setPen(QColor("#3A4250"))
        painter.setFont(QFont("Segoe UI", 26))
        painter.drawText(icon_rect, Qt.AlignmentFlag.AlignCenter, "⤒")

        title_rect = QRectF(box_x + 20, box_y + 84, box_w - 40, 30)
        painter.setPen(QColor("#CFD3DC"))
        painter.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "Arrastra un video aquí para comenzar")

        subtitle_rect = QRectF(box_x + 20, box_y + 116, box_w - 40, 44)
        painter.setPen(QColor("#6B7280"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(
            subtitle_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            "o usa \"Examinar\" en el panel de la derecha\nSoporta MP4, MKV, MOV, AVI y WebM"
        )

    def _draw_badge(self, painter: QPainter, text: str, x: int, y: int, bg_color: QColor, text_color: QColor, width: int = 160):
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg_color))
        rect = QRectF(x, y, width, 26)
        painter.drawRoundedRect(rect, 4, 4)

        painter.setPen(QPen(text_color))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    # --- Mouse & wheel interaction --------------------------------------
    def _near_split_handle(self, pos, tolerance: int = 14) -> bool:
        if self.view_mode != "split" or self._pixmap_ai is None:
            return False
        rect = self._current_render_rect(self.width(), self.height())
        split_x = rect.x() + rect.width() * self.split_ratio
        return abs(pos.x() - split_x) <= tolerance

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._near_split_handle(event.pos()):
            self._dragging_splitter = True
        else:
            self._panning = True
            self._pan_start_mouse = event.pos()
            self._pan_start_offset = QPointF(self.pan_offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging_splitter:
            self._update_split_from_mouse(event.pos().x())
        elif self._panning:
            delta = event.pos() - self._pan_start_mouse
            self.pan_offset = QPointF(
                self._pan_start_offset.x() + delta.x(),
                self._pan_start_offset.y() + delta.y(),
            )
            self.update()
        elif self._near_split_handle(event.pos()):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.zoom > 1.01 else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_splitter = False
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        self.reset_zoom()

    def wheelEvent(self, event):
        if self._pixmap_ai is None:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1 / 1.15)
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        if new_zoom == self.zoom:
            return

        # Zoom centered on the cursor: keep the point under the mouse fixed.
        cursor = event.position()
        rect = self._current_render_rect(self.width(), self.height())
        rel_x = (cursor.x() - rect.x()) / rect.width() if rect.width() else 0.5
        rel_y = (cursor.y() - rect.y()) / rect.height() if rect.height() else 0.5

        self.zoom = new_zoom
        new_rect = self._current_render_rect(self.width(), self.height())
        target_cursor_x = new_rect.x() + rel_x * new_rect.width()
        target_cursor_y = new_rect.y() + rel_y * new_rect.height()
        self.pan_offset += QPointF(cursor.x() - target_cursor_x, cursor.y() - target_cursor_y)

        self.zoom_changed.emit(self.zoom)
        self.update()

    def _update_split_from_mouse(self, mouse_x: int):
        rect = self._current_render_rect(self.width(), self.height())
        if rect.width() <= 0:
            return
        ratio = max(0.02, min(0.98, (mouse_x - rect.x()) / rect.width()))
        self.split_ratio = ratio
        self.update()


class SideBySideViewport(QWidget):
    """
    Comparison panel: a top toolbar (view mode + zoom controls) above the
    live-rendered canvas. Kept as the public class other modules import so
    existing signal wiring (`update_preview_frames`) needs no changes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_toolbar())

        self.canvas = _ComparisonCanvas(self)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        layout.addWidget(self.canvas, stretch=1)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("ViewerToolbar")
        bar.setFixedHeight(44)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 6, 14, 6)
        lay.setSpacing(6)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)

        def make_mode_btn(text, mode, checked=False):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setChecked(checked)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setObjectName("ViewModeButton")
            btn.clicked.connect(lambda: self.canvas.set_view_mode(mode))
            self._mode_group.addButton(btn)
            return btn

        lay.addWidget(make_mode_btn("🔀 Dividida", "split", checked=True))
        lay.addWidget(make_mode_btn("◀ Original", "original"))
        lay.addWidget(make_mode_btn("▶ Mejorado", "enhanced"))

        lay.addStretch(1)

        btn_zoom_out = QPushButton("－")
        btn_zoom_out.setObjectName("ZoomButton")
        btn_zoom_out.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_zoom_out.clicked.connect(lambda: self.canvas.set_zoom(self.canvas.zoom / 1.25))
        lay.addWidget(btn_zoom_out)

        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setObjectName("ZoomLabel")
        self.lbl_zoom.setFixedWidth(46)
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.lbl_zoom)

        btn_zoom_in = QPushButton("＋")
        btn_zoom_in.setObjectName("ZoomButton")
        btn_zoom_in.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_zoom_in.clicked.connect(lambda: self.canvas.set_zoom(self.canvas.zoom * 1.25))
        lay.addWidget(btn_zoom_in)

        btn_fit = QPushButton("Ajustar")
        btn_fit.setObjectName("ZoomButton")
        btn_fit.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_fit.setToolTip("Restablecer zoom y centrar (doble clic en la imagen)")
        btn_fit.clicked.connect(lambda: self.canvas.reset_zoom())
        lay.addWidget(btn_fit)

        return bar

    def _on_zoom_changed(self, zoom: float):
        self.lbl_zoom.setText(f"{int(round(zoom * 100))}%")

    def update_preview_frames(self, orig_rgb: np.ndarray, ai_rgb: np.ndarray):
        self.canvas.update_preview_frames(orig_rgb, ai_rgb)

    def set_processing_state(self, active: bool, progress_text: str = ""):
        self.canvas.set_processing_state(active, progress_text)
