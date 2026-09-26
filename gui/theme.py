"""
Modern Obsidian & AMD Radeon Dark Theme for PyQt6.
Inspired by Topaz Video AI professional interface aesthetic.
"""

DARK_THEME_QSS = """
QMainWindow, QWidget {
    background-color: #0F1115;
    color: #E6E8EC;
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
    font-size: 13px;
}

QGroupBox {
    background-color: #171A21;
    border: 1px solid #262B35;
    border-radius: 10px;
    margin-top: 24px;
    font-weight: 600;
    font-size: 12px;
    color: #00D2FF;
    letter-spacing: 0.3px;
    padding-top: 16px;
    padding-bottom: 4px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 2px 8px;
    background-color: #171A21;
}

QLabel {
    color: #CFD3DC;
}

QLabel#FieldLabel {
    color: #8B92A3;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.2px;
}

QWidget#AppHeader {
    background-color: #14161C;
    border-bottom: 1px solid #262B35;
}

QLabel#AppTitle {
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

QLabel#AppSubtitle {
    color: #8B92A3;
    font-size: 11px;
}

QLabel#AppBadge {
    color: #00D2FF;
    font-size: 11px;
    font-weight: 600;
    background-color: rgba(0, 210, 255, 0.10);
    border: 1px solid rgba(0, 210, 255, 0.35);
    border-radius: 10px;
    padding: 4px 10px;
}

QWidget#ViewerToolbar {
    background-color: #14161C;
    border-bottom: 1px solid #262B35;
}

QPushButton#ViewModeButton {
    background-color: transparent;
    border: 1px solid #2B313D;
    border-radius: 6px;
    padding: 6px 12px;
    color: #A0A6B2;
    font-weight: 600;
    font-size: 12px;
}

QPushButton#ViewModeButton:hover {
    border-color: #00D2FF;
    color: #FFFFFF;
}

QPushButton#ViewModeButton:checked {
    background-color: rgba(224, 17, 95, 0.18);
    border-color: #E0115F;
    color: #FFFFFF;
}

QPushButton#ZoomButton {
    background-color: #1B1E26;
    border: 1px solid #2B313D;
    border-radius: 6px;
    padding: 4px 10px;
    color: #FFFFFF;
    font-weight: 600;
}

QPushButton#ZoomButton:hover {
    border-color: #00D2FF;
}

QLabel#ZoomLabel {
    color: #8B92A3;
    font-size: 11px;
    font-weight: 600;
}

QToolTip {
    background-color: #1E222B;
    color: #E6E8EC;
    border: 1px solid #333A48;
    border-radius: 4px;
    padding: 6px 8px;
}

QTabWidget::pane {
    border: 1px solid #262B35;
    border-radius: 8px;
    background-color: #14161C;
    top: -1px;
}

QTabBar::tab {
    background-color: #171A21;
    color: #8B92A3;
    border: 1px solid #262B35;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 9px 16px;
    font-weight: 700;
    font-size: 12px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #14161C;
    color: #FFFFFF;
    border-bottom: 2px solid #E0115F;
}

QTabBar::tab:hover:!selected {
    color: #CFD3DC;
}

QLineEdit {
    background-color: #1B1E26;
    border: 1px solid #2B313D;
    border-radius: 6px;
    padding: 8px 12px;
    color: #FFFFFF;
    selection-background-color: #E0115F;
}

QLineEdit:focus {
    border: 1px solid #00D2FF;
}

QPushButton {
    background-color: #242934;
    border: 1px solid #333A48;
    border-radius: 6px;
    padding: 8px 16px;
    color: #FFFFFF;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2C3240;
    border-color: #434B5C;
}

QPushButton:pressed {
    background-color: #1E222B;
}

QPushButton#PrimaryButton {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E0115F, stop:1 #FF2A6D);
    border: none;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
    padding: 12px 24px;
    border-radius: 6px;
}

QPushButton#PrimaryButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF2A6D, stop:1 #FF5388);
}

QPushButton#PrimaryButton:disabled {
    background-color: #353B47;
    color: #717885;
}

QPushButton#SecondaryButton {
    background-color: #1E222B;
    border: 1px solid #FF1744;
    color: #FF1744;
    font-weight: 600;
    padding: 10px 20px;
    border-radius: 6px;
}

QPushButton#SecondaryButton:hover {
    background-color: rgba(255, 23, 68, 0.15);
}

QComboBox {
    background-color: #1B1E26;
    border: 1px solid #2B313D;
    border-radius: 6px;
    padding: 6px 12px;
    color: #FFFFFF;
}

QComboBox:hover {
    border-color: #00D2FF;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left-width: 0px;
}

QComboBox QAbstractItemView {
    background-color: #171A21;
    border: 1px solid #2B313D;
    selection-background-color: #E0115F;
    color: #FFFFFF;
    padding: 4px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #252A34;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D2FF, stop:1 #E0115F);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 2px solid #E0115F;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #00D2FF;
    border-color: #FFFFFF;
}

QProgressBar {
    background-color: #1B1E26;
    border: 1px solid #2B313D;
    border-radius: 6px;
    height: 18px;
    text-align: center;
    color: #FFFFFF;
    font-size: 11px;
    font-weight: 600;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E0115F, stop:1 #00D2FF);
    border-radius: 5px;
}

QScrollBar:vertical {
    background: #0F1115;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #2B313D;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #3B4352;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    height: 0px;
}

QCheckBox {
    color: #CFD3DC;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #333A48;
    border-radius: 4px;
    background-color: #1B1E26;
}

QCheckBox::indicator:checked {
    background-color: #00D2FF;
    border-color: #00D2FF;
}

QSpinBox {
    background-color: #1B1E26;
    border: 1px solid #2B313D;
    border-radius: 6px;
    padding: 6px 10px;
    color: #FFFFFF;
}

QSpinBox:hover {
    border-color: #00D2FF;
}

QSplitter::handle {
    background-color: #242934;
}

QSplitter::handle:hover {
    background-color: #00D2FF;
}
"""

