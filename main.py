"""
RadeonVideoAI - Generative Video AI & Upscaling Suite for Windows & AMD Radeon
Entrypoint script.
"""

import os
import sys
import logging
import traceback

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

# In a PyInstaller onedir build, __file__ resolves inside _internal/, which
# is not where a user looks for a log next to the .exe. Use the executable's
# own directory instead when frozen, so radeonvideoai.log sits beside it.
LOG_DIR = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False) else BASE_DIR
LOG_PATH = os.path.join(LOG_DIR, "radeonvideoai.log")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        ]
    )

    def log_unhandled_exception(exc_type, exc_value, exc_tb):
        logging.getLogger("RadeonVideoAI.Crash").critical(
            "Excepción no controlada, la aplicación se va a cerrar:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = log_unhandled_exception
    logging.info(f"Log persistente en: {LOG_PATH}")

def main():
    setup_logging()

    # Configure High DPI display behavior
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("RadeonVideoAI")
    app.setOrganizationName("RadeonAI")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()

