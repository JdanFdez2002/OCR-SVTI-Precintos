"""Punto de entrada mínimo para el futuro motor OCR genérico.

El motor anterior fue retirado. Esta ventana se conserva como interfaz de
compatibilidad y deja explícito que todavía no existe reconocimiento.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

from ui.styles import APP_STYLESHEET


class MotorWindow(QMainWindow):
    """Placeholder compatible, sin carga de modelos ni procesamiento OCR."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Motor OCR - no implementado")
        self.resize(640, 180)
        self.setStyleSheet(APP_STYLESHEET)
        label = QLabel(
            "El reconocimiento OCR no está implementado todavía.\n"
            "Esta ventana queda preparada para el futuro lector genérico."
        )
        label.setWordWrap(True)
        label.setContentsMargins(24, 24, 24, 24)
        self.setCentralWidget(label)


def main() -> int:
    app = QApplication(sys.argv)
    ventana = MotorWindow()
    ventana.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
