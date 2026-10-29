from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
)

from config import CANAL_RTSP_FOTO
from modulos.settings_store import CameraSettings


class SettingsDialog(QDialog):
    def __init__(self, cfg: CameraSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de cámara")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._ip = QLineEdit(cfg.ip)
        self._usuario = QLineEdit(cfg.usuario)
        self._password = QLineEdit(cfg.password)
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        self._canal = QLineEdit(CANAL_RTSP_FOTO)
        self._canal.setReadOnly(True)
        self._canal.setToolTip("El canal 101 se reserva para fotografías destinadas al OCR.")

        form = QFormLayout()
        form.addRow("IP de la cámara", self._ip)
        form.addRow("Usuario", self._usuario)
        form.addRow("Contraseña", self._password)
        form.addRow("Canal RTSP de fotografías", self._canal)

        botones = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(botones)

    def valores(self) -> CameraSettings:
        return CameraSettings(
            ip=self._ip.text().strip(),
            usuario=self._usuario.text().strip(),
            password=self._password.text(),
            canal=CANAL_RTSP_FOTO,
        )
