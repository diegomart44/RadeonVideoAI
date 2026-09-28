"""
Controls and Configuration Panel.

Two fully independent tabs, each with its own file pickers and its own
Start/Pause/Cancel row — deliberately NOT sharing a single "start" button,
because they are two unrelated AI pipelines:

- "🔍 Reescalar con IA": Real-ESRGAN-family restoration/upscaling
  (core/engine.py + core/inference.py). Faithfully improves what's already
  in the frame; can change resolution.
- "🌌 Generar (IA Generativa)": Stable Diffusion img2img recreation
  (core/generative_engine.py + core/generative_models.py). Hallucinates new
  plausible detail; never changes resolution; much slower.
- "🎞️ Interpolar Fotogramas (FPS)": RIFE optical-flow frame interpolation
  (core/interpolation_engine.py + core/interpolation_models.py). Raises
  frame rate (x2/x4) without touching resolution or per-frame detail;
  resumable via on-disk checkpoints if interrupted.

Keeping them as separate tabs with separate buttons (rather than one shared
flow) was an explicit requirement — the features must not be able to break
each other, and the UI should not imply they're the same operation.
"""

import os
from PyQt6.QtCore import pyqtSignal, Qt
from core.media_pipeline import VideoMetadataReader
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QSlider, QSpinBox, QCheckBox,
    QGroupBox, QFileDialog, QScrollArea, QSizePolicy, QTabWidget
)


class ControlsPanel(QWidget):
    """Configuration controls styled with modern obsidian & Radeon Ruby aesthetics."""

    # Reescalar con IA (Real-ESRGAN)
    start_requested = pyqtSignal(dict)
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    # Generar (IA Generativa / difusión)
    generate_requested = pyqtSignal(dict)
    generate_pause_requested = pyqtSignal()
    generate_resume_requested = pyqtSignal()
    generate_cancel_requested = pyqtSignal()

    # Interpolar Fotogramas (RIFE, x2/x4 FPS)
    interpolate_requested = pyqtSignal(dict)
    interpolate_pause_requested = pyqtSignal()
    interpolate_resume_requested = pyqtSignal()
    interpolate_cancel_requested = pyqtSignal()

    # Fired whenever ANY tab's input file changes, so the viewer can show
    # the first frame right away regardless of which pipeline gets used.
    file_selected = pyqtSignal(str)

    PANEL_WIDTH = 400

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self.PANEL_WIDTH)
        self._is_paused = False
        self._is_gen_paused = False
        self._is_interp_paused = False
        self._interp_source_meta = None

        self._build_ui()

    def _combo(self, items_with_tooltips):
        """Builds a QComboBox that never grows past the panel width, no
        matter how long an item's label is; the full description is
        available as a tooltip on the item and on the box itself."""
        box = QComboBox()
        box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        box.setMinimumContentsLength(1)
        box.setMaximumWidth(self.PANEL_WIDTH - 56)
        for text, tooltip in items_with_tooltips:
            box.addItem(text)
            if tooltip:
                box.setItemData(box.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)
        if items_with_tooltips and items_with_tooltips[0][1]:
            box.setToolTip(items_with_tooltips[0][1])
            box.currentIndexChanged.connect(
                lambda i, b=box: b.setToolTip(b.itemData(i, Qt.ItemDataRole.ToolTipRole) or "")
            )
        return box

    def _section_label(self, text: str):
        lbl = QLabel(text)
        lbl.setObjectName("FieldLabel")
        lbl.setWordWrap(True)
        return lbl

    def _scroll_wrap(self, container: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        scroll.setWidget(container)
        return scroll

    def _action_row(self, start_text: str, start_tooltip: str, on_start, on_pause, on_cancel):
        """Builds a self-contained Start/Pause/Cancel row. Each tab gets its
        own — returns the three buttons so callers can enable/disable them."""
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        btn_start = QPushButton(start_text)
        btn_start.setObjectName("PrimaryButton")
        btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_start.setToolTip(start_tooltip)
        btn_start.clicked.connect(on_start)
        lay.addWidget(btn_start)

        h_actions = QHBoxLayout()
        h_actions.setSpacing(10)
        btn_pause = QPushButton("⏸  Pausar")
        btn_pause.setEnabled(False)
        btn_pause.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_pause.clicked.connect(on_pause)
        h_actions.addWidget(btn_pause)

        btn_cancel = QPushButton("✕  Cancelar")
        btn_cancel.setObjectName("SecondaryButton")
        btn_cancel.setEnabled(False)
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(on_cancel)
        h_actions.addWidget(btn_cancel)
        lay.addLayout(h_actions)

        return wrap, btn_start, btn_pause, btn_cancel

    # -----------------------------------------------------------------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_rescale_tab(), "🔍  Reescalar con IA")
        self.tabs.addTab(self._build_generate_tab(), "🌌  Generar (IA Generativa)")
        self.tabs.addTab(self._build_interpolate_tab(), "🎞️  Interpolar (FPS)")
        main_layout.addWidget(self.tabs, stretch=1)

    # =====================================================================
    # Tab 1: Reescalar con IA (Real-ESRGAN restoration/upscaling)
    # =====================================================================
    def _build_rescale_tab(self) -> QWidget:
        outer = QWidget()
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(12)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        grp_files = QGroupBox("📁  Video a Reescalar")
        lay_files = QVBoxLayout(grp_files)
        lay_files.setSpacing(6)

        lay_files.addWidget(self._section_label("Video original"))
        h_in = QHBoxLayout()
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Seleccionar archivo...")
        btn_in = QPushButton("Examinar")
        btn_in.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_in.clicked.connect(self._browse_input)
        h_in.addWidget(self.txt_input)
        h_in.addWidget(btn_in)
        lay_files.addLayout(h_in)

        lay_files.addWidget(self._section_label("Video reescalado (destino)"))
        h_out = QHBoxLayout()
        self.txt_output = QLineEdit()
        self.txt_output.setPlaceholderText("Ruta de salida generada...")
        btn_out = QPushButton("Guardar")
        btn_out.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_out.clicked.connect(self._browse_output)
        h_out.addWidget(self.txt_output)
        h_out.addWidget(btn_out)
        lay_files.addLayout(h_out)
        layout.addWidget(grp_files)

        grp_ai = QGroupBox("🧠  Modelo de IA")
        lay_ai = QVBoxLayout(grp_ai)
        lay_ai.setSpacing(6)
        lay_ai.addWidget(self._section_label("Arquitectura / checkpoint"))
        self.cmb_model = self._combo([
            ("Universal (rápido)", "general_v3 — Rápido, uso general, con control de desruido real."),
            ("Máxima calidad 4x", "x4plus — Mayor detalle, más lento. Ideal para escala 4x."),
            ("Alta fidelidad 2x", "x2plus — Red nativa 2x, mejor que reescalar desde un modelo 4x."),
            ("Vídeo real / compresión", "BSRGAN — Entrenado para ruido y artefactos de compresión reales."),
            ("Animación y CG", "anime6B — Optimizado para animación y contenido dibujado."),
            ("Ultra nitidez (comunidad)", "4x-UltraSharp — Muy popular por su nitidez. Licencia CC BY-NC-SA: SOLO uso no comercial/personal."),
        ])
        self._model_name_map = ["general_v3", "x4plus", "x2plus", "bsrgan", "anime6b", "ultrasharp"]
        lay_ai.addWidget(self.cmb_model)

        lay_ai.addWidget(self._section_label("Factor de reescalado"))
        self.cmb_scale = self._combo([
            ("2x — Recomendado", "Duplica ancho y alto. Mejor equilibrio velocidad/calidad."),
            ("4x — Ultra HD / 4K", "Cuadruplica ancho y alto. Más lento, máximo detalle."),
            ("1x — Solo restauración", "Mantiene la resolución; solo desruido y nitidez."),
        ])
        lay_ai.addWidget(self.cmb_scale)

        self.lbl_denoise = QLabel("Filtro de desruido — 15%")
        self.lbl_denoise.setObjectName("FieldLabel")
        self.lbl_denoise.setWordWrap(True)
        lay_ai.addWidget(self.lbl_denoise)
        self.sld_denoise = QSlider(Qt.Orientation.Horizontal)
        self.sld_denoise.setRange(0, 100)
        self.sld_denoise.setValue(15)
        self.sld_denoise.valueChanged.connect(lambda v: self.lbl_denoise.setText(f"Filtro de desruido — {v}%"))
        lay_ai.addWidget(self.sld_denoise)

        self.lbl_sharpen = QLabel("Nitidez y detalle — 20%")
        self.lbl_sharpen.setObjectName("FieldLabel")
        self.lbl_sharpen.setWordWrap(True)
        lay_ai.addWidget(self.lbl_sharpen)
        self.sld_sharpen = QSlider(Qt.Orientation.Horizontal)
        self.sld_sharpen.setRange(0, 100)
        self.sld_sharpen.setValue(20)
        self.sld_sharpen.valueChanged.connect(lambda v: self.lbl_sharpen.setText(f"Nitidez y detalle — {v}%"))
        lay_ai.addWidget(self.sld_sharpen)
        layout.addWidget(grp_ai)

        grp_vram = QGroupBox("🧩  Memoria GPU y Parches")
        lay_vram = QVBoxLayout(grp_vram)
        lay_vram.setSpacing(6)

        lay_vram.addWidget(self._section_label("Tamaño de parche (tile)"))
        self.cmb_tile = self._combo([
            ("768 px — Recomendado 16GB", "Máximo aprovechamiento de GPU en tarjetas de 16GB como la RX 9060 XT."),
            ("1024 px — Máximo", "Menos parches por fotograma; exprime más la GPU en tarjetas de 16GB con vídeo de alta resolución."),
            ("512 px — Equilibrado", "Buen balance entre velocidad y uso de memoria."),
            ("384 px", "Menor uso de memoria, algo más lento por fotograma."),
            ("256 px — Bajo consumo", "Para GPUs con poca VRAM."),
        ])
        lay_vram.addWidget(self.cmb_tile)

        lay_vram.addWidget(self._section_label("Solapamiento (fusión de bordes)"))
        self.cmb_overlap = self._combo([
            ("48 px — Fusión suave", "Sin costuras visibles entre parches. Recomendado."),
            ("32 px — Rápido", "Menos solapamiento, procesa algo más rápido."),
            ("64 px — Ultra suave", "Máxima suavidad en los bordes, más lento."),
        ])
        lay_vram.addWidget(self.cmb_overlap)

        lay_vram.addWidget(self._section_label("Umbral de seguridad anti-OOM"))
        self.cmb_vram_limit = self._combo([
            ("90% VRAM — Automático", "Degrada el tamaño de parche automáticamente cerca del límite."),
            ("85% VRAM — Conservador", "Deja más margen libre de memoria."),
            ("95% VRAM — Agresivo", "Exprime al máximo la memoria disponible."),
        ])
        lay_vram.addWidget(self.cmb_vram_limit)
        layout.addWidget(grp_vram)

        grp_enc = QGroupBox("🎬  Codificación (AMD AMF)")
        lay_enc = QVBoxLayout(grp_enc)
        lay_enc.setSpacing(6)

        lay_enc.addWidget(self._section_label("Códec de salida"))
        self.cmb_encoder = self._combo([
            ("HEVC (H.265) — Hardware AMD", "hevc_amf — Acelerado por la GPU AMD. Mejor compresión."),
            ("H.264 — Hardware AMD", "h264_amf — Acelerado por la GPU AMD. Máxima compatibilidad."),
            ("HEVC (H.265) — Software", "libx265 — Usa la CPU. Más lento, sin AMF."),
            ("H.264 — Software", "libx264 — Usa la CPU. Más lento, sin AMF."),
        ])
        lay_enc.addWidget(self.cmb_encoder)

        lay_enc.addWidget(self._section_label("Tasa de bits"))
        h_bitrate = QHBoxLayout()
        self.spn_bitrate = QSpinBox()
        self.spn_bitrate.setRange(5, 150)
        self.spn_bitrate.setValue(25)
        self.spn_bitrate.setSuffix(" Mbps")
        h_bitrate.addWidget(self.spn_bitrate)
        lay_enc.addLayout(h_bitrate)

        self.chk_audio = QCheckBox("Preservar audio original")
        self.chk_audio.setToolTip("Copia las pistas de audio sin recodificar (-c:a copy).")
        self.chk_audio.setChecked(True)
        lay_enc.addWidget(self.chk_audio)
        layout.addWidget(grp_enc)

        layout.addStretch(1)
        outer_lay.addWidget(self._scroll_wrap(container), stretch=1)

        row, self.btn_start, self.btn_pause, self.btn_cancel = self._action_row(
            "✨  REESCALAR VIDEO",
            "Reescala/restaura este video con Real-ESRGAN (fiel al original, más nítido y sin ruido).",
            self._on_start_clicked, self._on_pause_clicked, self._on_cancel_clicked
        )
        outer_lay.addWidget(row)
        return outer

    # =====================================================================
    # Tab 2: Generar (IA Generativa / difusión) — NO relation to rescaling
    # =====================================================================
    def _build_generate_tab(self) -> QWidget:
        outer = QWidget()
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(12)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        grp_files = QGroupBox("📁  Video a Generar")
        lay_files = QVBoxLayout(grp_files)
        lay_files.setSpacing(6)

        lay_files.addWidget(self._section_label("Video original"))
        h_in = QHBoxLayout()
        self.txt_gen_input = QLineEdit()
        self.txt_gen_input.setPlaceholderText("Seleccionar archivo...")
        btn_in = QPushButton("Examinar")
        btn_in.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_in.clicked.connect(self._browse_gen_input)
        h_in.addWidget(self.txt_gen_input)
        h_in.addWidget(btn_in)
        lay_files.addLayout(h_in)

        lay_files.addWidget(self._section_label("Video generado (destino)"))
        h_out = QHBoxLayout()
        self.txt_gen_output = QLineEdit()
        self.txt_gen_output.setPlaceholderText("Ruta de salida generada...")
        btn_out = QPushButton("Guardar")
        btn_out.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_out.clicked.connect(self._browse_gen_output)
        h_out.addWidget(self.txt_gen_output)
        h_out.addWidget(btn_out)
        lay_files.addLayout(h_out)
        layout.addWidget(grp_files)

        grp_gen = QGroupBox("🌌  Recreación Generativa")
        lay_gen = QVBoxLayout(grp_gen)
        lay_gen.setSpacing(6)

        notice = QLabel(
            "Esta IA NO reescala — el video generado mantiene la MISMA resolución. "
            "En vez de restaurar fielmente, reinterpreta cada fotograma agregando "
            "detalle nuevo. Es una técnica distinta (difusión) y mucho más lenta: "
            "esperá varios segundos por fotograma, no en tiempo real. "
            "Modelo: Stable Diffusion 1.5 (licencia CreativeML Open RAIL-M)."
        )
        notice.setObjectName("FieldLabel")
        notice.setWordWrap(True)
        lay_gen.addWidget(notice)

        self.lbl_gen_strength = QLabel("Fuerza / creatividad — 40%")
        self.lbl_gen_strength.setObjectName("FieldLabel")
        self.lbl_gen_strength.setWordWrap(True)
        lay_gen.addWidget(self.lbl_gen_strength)
        self.sld_gen_strength = QSlider(Qt.Orientation.Horizontal)
        self.sld_gen_strength.setRange(5, 95)
        self.sld_gen_strength.setValue(40)
        self.sld_gen_strength.valueChanged.connect(
            lambda v: self.lbl_gen_strength.setText(f"Fuerza / creatividad — {v}%")
        )
        lay_gen.addWidget(self.sld_gen_strength)

        self.lbl_gen_steps = QLabel("Pasos de generación — 15")
        self.lbl_gen_steps.setObjectName("FieldLabel")
        self.lbl_gen_steps.setWordWrap(True)
        lay_gen.addWidget(self.lbl_gen_steps)
        self.sld_gen_steps = QSlider(Qt.Orientation.Horizontal)
        self.sld_gen_steps.setRange(10, 50)
        self.sld_gen_steps.setValue(15)
        self.sld_gen_steps.valueChanged.connect(
            lambda v: self.lbl_gen_steps.setText(f"Pasos de generación — {v}")
        )
        lay_gen.addWidget(self.sld_gen_steps)
        layout.addWidget(grp_gen)

        grp_enc2 = QGroupBox("🎬  Codificación (AMD AMF)")
        lay_enc2 = QVBoxLayout(grp_enc2)
        lay_enc2.setSpacing(6)

        lay_enc2.addWidget(self._section_label("Códec de salida"))
        self.cmb_gen_encoder = self._combo([
            ("HEVC (H.265) — Hardware AMD", "hevc_amf — Acelerado por la GPU AMD. Mejor compresión."),
            ("H.264 — Hardware AMD", "h264_amf — Acelerado por la GPU AMD. Máxima compatibilidad."),
            ("HEVC (H.265) — Software", "libx265 — Usa la CPU. Más lento, sin AMF."),
            ("H.264 — Software", "libx264 — Usa la CPU. Más lento, sin AMF."),
        ])
        lay_enc2.addWidget(self.cmb_gen_encoder)

        lay_enc2.addWidget(self._section_label("Tasa de bits"))
        h_bitrate2 = QHBoxLayout()
        self.spn_gen_bitrate = QSpinBox()
        self.spn_gen_bitrate.setRange(5, 150)
        self.spn_gen_bitrate.setValue(25)
        self.spn_gen_bitrate.setSuffix(" Mbps")
        h_bitrate2.addWidget(self.spn_gen_bitrate)
        lay_enc2.addLayout(h_bitrate2)

        self.chk_gen_audio = QCheckBox("Preservar audio original")
        self.chk_gen_audio.setToolTip("Copia las pistas de audio sin recodificar (-c:a copy).")
        self.chk_gen_audio.setChecked(True)
        lay_enc2.addWidget(self.chk_gen_audio)
        layout.addWidget(grp_enc2)

        layout.addStretch(1)
        outer_lay.addWidget(self._scroll_wrap(container), stretch=1)

        row, self.btn_gen_start, self.btn_gen_pause, self.btn_gen_cancel = self._action_row(
            "🌌  GENERAR VIDEO CON IA",
            "Genera un video nuevo reinterpretando cada fotograma con IA generativa (difusión). Mucho más lento que reescalar.",
            self._on_gen_start_clicked, self._on_gen_pause_clicked, self._on_gen_cancel_clicked
        )
        outer_lay.addWidget(row)
        return outer

    # =====================================================================
    # Tab 3: Interpolar Fotogramas (RIFE, x2/x4 FPS) — resumable, own engine
    # =====================================================================
    def _build_interpolate_tab(self) -> QWidget:
        outer = QWidget()
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(12)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        grp_files = QGroupBox("📁  Video a Interpolar")
        lay_files = QVBoxLayout(grp_files)
        lay_files.setSpacing(6)

        lay_files.addWidget(self._section_label("Video original"))
        h_in = QHBoxLayout()
        self.txt_interp_input = QLineEdit()
        self.txt_interp_input.setPlaceholderText("Seleccionar archivo...")
        btn_in = QPushButton("Examinar")
        btn_in.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_in.clicked.connect(self._browse_interp_input)
        h_in.addWidget(self.txt_interp_input)
        h_in.addWidget(btn_in)
        lay_files.addLayout(h_in)

        lay_files.addWidget(self._section_label("Video interpolado (destino)"))
        h_out = QHBoxLayout()
        self.txt_interp_output = QLineEdit()
        self.txt_interp_output.setPlaceholderText("Ruta de salida generada...")
        btn_out = QPushButton("Guardar")
        btn_out.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_out.clicked.connect(self._browse_interp_output)
        h_out.addWidget(self.txt_interp_output)
        h_out.addWidget(btn_out)
        lay_files.addLayout(h_out)
        layout.addWidget(grp_files)

        grp_interp = QGroupBox("🎞️  Interpolación (RIFE)")
        lay_interp = QVBoxLayout(grp_interp)
        lay_interp.setSpacing(6)

        notice = QLabel(
            "Aumenta los FPS del video generando fotogramas intermedios reales "
            "por flujo óptico (RIFE) — NO cambia la resolución ni tiene relación "
            "con las otras dos pestañas. Si se corta por cualquier motivo "
            "(cierre, corte de luz, cancelar), el progreso queda guardado en un "
            "archivo temporal junto al destino: al reiniciar el mismo trabajo "
            "continúa donde quedó en vez de arrancar de cero, y va liberando ese "
            "espacio temporal a medida que confirma cada tramo ya interpolado."
        )
        notice.setObjectName("FieldLabel")
        notice.setWordWrap(True)
        lay_interp.addWidget(notice)

        lay_interp.addWidget(self._section_label("Multiplicador de FPS"))
        self.cmb_interp_mult = self._combo([
            ("x2 — Recomendado", "Duplica los FPS (ej. 24 → 48). Un fotograma nuevo entre cada par de originales."),
            ("x4 — Máxima fluidez", "Cuadruplica los FPS (ej. 24 → 96). Tres fotogramas nuevos entre cada par de originales; más lento."),
        ])
        self.cmb_interp_mult.currentIndexChanged.connect(self._update_interp_fps_preview)
        lay_interp.addWidget(self.cmb_interp_mult)

        self.lbl_interp_preview = QLabel("Elegí un video para ver los FPS y la resolución resultantes.")
        self.lbl_interp_preview.setObjectName("FieldLabel")
        self.lbl_interp_preview.setWordWrap(True)
        lay_interp.addWidget(self.lbl_interp_preview)
        layout.addWidget(grp_interp)

        grp_enc3 = QGroupBox("🎬  Codificación (AMD AMF)")
        lay_enc3 = QVBoxLayout(grp_enc3)
        lay_enc3.setSpacing(6)

        lay_enc3.addWidget(self._section_label("Códec de salida"))
        self.cmb_interp_encoder = self._combo([
            ("HEVC (H.265) — Hardware AMD", "hevc_amf — Acelerado por la GPU AMD. Mejor compresión."),
            ("H.264 — Hardware AMD", "h264_amf — Acelerado por la GPU AMD. Máxima compatibilidad."),
            ("HEVC (H.265) — Software", "libx265 — Usa la CPU. Más lento, sin AMF."),
            ("H.264 — Software", "libx264 — Usa la CPU. Más lento, sin AMF."),
        ])
        lay_enc3.addWidget(self.cmb_interp_encoder)

        lay_enc3.addWidget(self._section_label("Tasa de bits"))
        h_bitrate3 = QHBoxLayout()
        self.spn_interp_bitrate = QSpinBox()
        self.spn_interp_bitrate.setRange(5, 150)
        self.spn_interp_bitrate.setValue(25)
        self.spn_interp_bitrate.setSuffix(" Mbps")
        h_bitrate3.addWidget(self.spn_interp_bitrate)
        lay_enc3.addLayout(h_bitrate3)
        layout.addWidget(grp_enc3)

        layout.addStretch(1)
        outer_lay.addWidget(self._scroll_wrap(container), stretch=1)

        row, self.btn_interp_start, self.btn_interp_pause, self.btn_interp_cancel = self._action_row(
            "🎞️  INTERPOLAR FOTOGRAMAS",
            "Aumenta los FPS del video generando fotogramas intermedios reales por flujo óptico (RIFE). Resumible si se interrumpe.",
            self._on_interp_start_clicked, self._on_interp_pause_clicked, self._on_interp_cancel_clicked
        )
        outer_lay.addWidget(row)
        return outer

    def set_active_tab_input_path(self, path: str):
        """Routes a dropped/selected file to whichever tab is currently
        showing (0 = Reescalar, 1 = Generar, 2 = Interpolar), so drag-and-drop
        feels consistent with whichever pipeline the user is currently looking at."""
        idx = self.tabs.currentIndex()
        if idx == 1:
            self.set_gen_input_path(path)
        elif idx == 2:
            self.set_interp_input_path(path)
        else:
            self.set_input_path(path)

    # =====================================================================
    # Reescalar: file pickers & start/pause/cancel
    # =====================================================================
    def set_input_path(self, path: str):
        """Sets the Reescalar tab's input path and notifies listeners (used
        both by the Browse dialog and by drag-and-drop) so the viewer can
        immediately show the first frame instead of waiting for a job."""
        self.txt_input.setText(path)
        base, ext = os.path.splitext(path)
        scale_tag = "2x" if self.cmb_scale.currentIndex() == 0 else ("4x" if self.cmb_scale.currentIndex() == 1 else "restored")
        self.txt_output.setText(f"{base}_AI_{scale_tag}.mp4")
        self.file_selected.emit(path)

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Video", "",
            "Archivos de Video (*.mp4 *.mkv *.mov *.avi *.webm *.ts);;Todos los Archivos (*.*)"
        )
        if path:
            self.set_input_path(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Video Reescalado", self.txt_output.text() or "",
            "Archivo MP4 (*.mp4);;Archivo MKV (*.mkv)"
        )
        if path:
            self.txt_output.setText(path)

    def _on_start_clicked(self):
        in_p = self.txt_input.text().strip()
        out_p = self.txt_output.text().strip()
        if not in_p or not os.path.isfile(in_p):
            return

        scale_idx = self.cmb_scale.currentIndex()
        scale = 2 if scale_idx == 0 else (4 if scale_idx == 1 else 1)

        tile_str = "".join(ch for ch in self.cmb_tile.currentText().split(" ")[0] if ch.isdigit())
        tile_size = int(tile_str) if tile_str.isdigit() else 512

        overlap_str = "".join(ch for ch in self.cmb_overlap.currentText().split(" ")[0] if ch.isdigit())
        overlap = int(overlap_str) if overlap_str.isdigit() else 48

        enc_raw = self.cmb_encoder.currentText()
        if "HEVC" in enc_raw and "Hardware" in enc_raw:
            encoder = "hevc_amf"
        elif "H.264" in enc_raw and "Hardware" in enc_raw:
            encoder = "h264_amf"
        elif "HEVC" in enc_raw:
            encoder = "libx265"
        else:
            encoder = "libx264"

        model_idx = self.cmb_model.currentIndex()
        model_name = self._model_name_map[model_idx] if 0 <= model_idx < len(self._model_name_map) else "general_v3"

        config = {
            "input_path": in_p,
            "output_path": out_p,
            "scale": scale,
            "model_name": model_name,
            "tile_size": tile_size,
            "overlap": overlap,
            "denoise": self.sld_denoise.value() / 100.0,
            "sharpen": self.sld_sharpen.value() / 100.0,
            "encoder": encoder,
            "bitrate_mbps": self.spn_bitrate.value(),
            "vram_safety": 0.90,
            "has_audio": self.chk_audio.isChecked()
        }

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.start_requested.emit(config)

    def _on_pause_clicked(self):
        if not self._is_paused:
            self._is_paused = True
            self.btn_pause.setText("▶  Reanudar")
            self.pause_requested.emit()
        else:
            self._is_paused = False
            self.btn_pause.setText("⏸  Pausar")
            self.resume_requested.emit()

    def _on_cancel_clicked(self):
        self.cancel_requested.emit()

    def set_processing_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_pause.setText("⏸  Pausar")
        self._is_paused = False

    # =====================================================================
    # Generar: file pickers & start/pause/cancel (fully independent)
    # =====================================================================
    def set_gen_input_path(self, path: str):
        self.txt_gen_input.setText(path)
        base, ext = os.path.splitext(path)
        self.txt_gen_output.setText(f"{base}_Generado_IA.mp4")
        self.file_selected.emit(path)

    def _browse_gen_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Video", "",
            "Archivos de Video (*.mp4 *.mkv *.mov *.avi *.webm *.ts);;Todos los Archivos (*.*)"
        )
        if path:
            self.set_gen_input_path(path)

    def _browse_gen_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Video Generado", self.txt_gen_output.text() or "",
            "Archivo MP4 (*.mp4);;Archivo MKV (*.mkv)"
        )
        if path:
            self.txt_gen_output.setText(path)

    def _on_gen_start_clicked(self):
        in_p = self.txt_gen_input.text().strip()
        out_p = self.txt_gen_output.text().strip()
        if not in_p or not os.path.isfile(in_p):
            return

        enc_raw = self.cmb_gen_encoder.currentText()
        if "HEVC" in enc_raw and "Hardware" in enc_raw:
            encoder = "hevc_amf"
        elif "H.264" in enc_raw and "Hardware" in enc_raw:
            encoder = "h264_amf"
        elif "HEVC" in enc_raw:
            encoder = "libx265"
        else:
            encoder = "libx264"

        config = {
            "input_path": in_p,
            "output_path": out_p,
            "steps": self.sld_gen_steps.value(),
            "strength": self.sld_gen_strength.value() / 100.0,
            "overlap": 32,
            "encoder": encoder,
            "bitrate_mbps": self.spn_gen_bitrate.value(),
            "vram_safety": 0.90,
            "has_audio": self.chk_gen_audio.isChecked()
        }

        self.btn_gen_start.setEnabled(False)
        self.btn_gen_pause.setEnabled(True)
        self.btn_gen_cancel.setEnabled(True)
        self.generate_requested.emit(config)

    def _on_gen_pause_clicked(self):
        if not self._is_gen_paused:
            self._is_gen_paused = True
            self.btn_gen_pause.setText("▶  Reanudar")
            self.generate_pause_requested.emit()
        else:
            self._is_gen_paused = False
            self.btn_gen_pause.setText("⏸  Pausar")
            self.generate_resume_requested.emit()

    def _on_gen_cancel_clicked(self):
        self.generate_cancel_requested.emit()

    def set_generation_finished(self):
        self.btn_gen_start.setEnabled(True)
        self.btn_gen_pause.setEnabled(False)
        self.btn_gen_cancel.setEnabled(False)
        self.btn_gen_pause.setText("⏸  Pausar")
        self._is_gen_paused = False

    # =====================================================================
    # Interpolar: file pickers & start/pause/cancel (fully independent)
    # =====================================================================
    def set_interp_input_path(self, path: str):
        self.txt_interp_input.setText(path)
        base, ext = os.path.splitext(path)
        mult_tag = "x2" if self.cmb_interp_mult.currentIndex() == 0 else "x4"
        self.txt_interp_output.setText(f"{base}_{mult_tag}FPS.mp4")

        try:
            self._interp_source_meta = VideoMetadataReader.probe(path)
        except Exception:
            self._interp_source_meta = None
        self._update_interp_fps_preview()

        self.file_selected.emit(path)

    def _update_interp_fps_preview(self):
        meta = self._interp_source_meta
        if not meta or not meta.get("fps"):
            self.lbl_interp_preview.setText("Elegí un video para ver los FPS y la resolución resultantes.")
            return
        multiplier = 2 if self.cmb_interp_mult.currentIndex() == 0 else 4
        src_fps = meta["fps"]
        out_fps = src_fps * multiplier
        w, h = meta.get("width", 0), meta.get("height", 0)
        self.lbl_interp_preview.setText(
            f"Original: {w}x{h} @ {src_fps:.2f} fps  →  Resultado: {w}x{h} @ {out_fps:.2f} fps "
            f"(x{multiplier}). La resolución NO cambia, solo los FPS."
        )

    def _browse_interp_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Video", "",
            "Archivos de Video (*.mp4 *.mkv *.mov *.avi *.webm *.ts);;Todos los Archivos (*.*)"
        )
        if path:
            self.set_interp_input_path(path)

    def _browse_interp_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Video Interpolado", self.txt_interp_output.text() or "",
            "Archivo MP4 (*.mp4);;Archivo MKV (*.mkv)"
        )
        if path:
            self.txt_interp_output.setText(path)

    def _on_interp_start_clicked(self):
        in_p = self.txt_interp_input.text().strip()
        out_p = self.txt_interp_output.text().strip()
        if not in_p or not os.path.isfile(in_p):
            return

        multiplier = 2 if self.cmb_interp_mult.currentIndex() == 0 else 4

        enc_raw = self.cmb_interp_encoder.currentText()
        if "HEVC" in enc_raw and "Hardware" in enc_raw:
            encoder = "hevc_amf"
        elif "H.264" in enc_raw and "Hardware" in enc_raw:
            encoder = "h264_amf"
        elif "HEVC" in enc_raw:
            encoder = "libx265"
        else:
            encoder = "libx264"

        config = {
            "input_path": in_p,
            "output_path": out_p,
            "multiplier": multiplier,
            "encoder": encoder,
            "bitrate_mbps": self.spn_interp_bitrate.value(),
        }

        self.btn_interp_start.setEnabled(False)
        self.btn_interp_pause.setEnabled(True)
        self.btn_interp_cancel.setEnabled(True)
        self.interpolate_requested.emit(config)

    def _on_interp_pause_clicked(self):
        if not self._is_interp_paused:
            self._is_interp_paused = True
            self.btn_interp_pause.setText("▶  Reanudar")
            self.interpolate_pause_requested.emit()
        else:
            self._is_interp_paused = False
            self.btn_interp_pause.setText("⏸  Pausar")
            self.interpolate_resume_requested.emit()

    def _on_interp_cancel_clicked(self):
        self.interpolate_cancel_requested.emit()

    def set_interpolation_finished(self):
        self.btn_interp_start.setEnabled(True)
        self.btn_interp_pause.setEnabled(False)
        self.btn_interp_cancel.setEnabled(False)
        self.btn_interp_pause.setText("⏸  Pausar")
        self._is_interp_paused = False
