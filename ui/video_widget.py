import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from config import ALTO_RETICULA_PCT, ANCHO_RETICULA_PCT, GROSOR_RETICULA


class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._placeholder = "Configurá la cámara para ver el video"
        self.setMinimumSize(720, 405)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_placeholder(self, texto: str):
        self._placeholder = texto
        self.update()

    def mostrar_frame(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        alto, ancho, canales = rgb.shape
        imagen = QImage(
            rgb.data, ancho, alto, canales * ancho, QImage.Format.Format_RGB888
        ).copy()
        self._pixmap = QPixmap.fromImage(imagen)
        self.update()

    def limpiar(self):
        self._pixmap = None
        self.update()

    def tiene_imagen(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    def _rect_destino(self) -> QRect | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        vw, vh = self.width(), self.height()
        if pw <= 0 or ph <= 0 or vw <= 0 or vh <= 0:
            return None
        escala = min(vw / pw, vh / ph)
        dw, dh = int(pw * escala), int(ph * escala)
        x = (vw - dw) // 2
        y = (vh - dh) // 2
        return QRect(x, y, dw, dh)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#11111B"))
        dest = self._rect_destino()
        if dest is None:
            painter.setPen(QColor("#A6ADC8"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        painter.drawPixmap(dest, self._pixmap)
        rw = max(8, int(dest.width() * ANCHO_RETICULA_PCT))
        rh = max(8, int(dest.height() * ALTO_RETICULA_PCT))
        rx = dest.center().x() - rw // 2
        ry = dest.center().y() - rh // 2
        pen = QPen(QColor(0, 180, 80))
        pen.setWidth(GROSOR_RETICULA)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rx, ry, rw, rh)
