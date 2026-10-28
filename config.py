import os
from pathlib import Path

# Debe setearse antes del primer VideoCapture
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

GROSOR_RETICULA = 2
ANCHO_RETICULA_PCT = 0.30
ALTO_RETICULA_PCT = 0.20

RAIZ_PROYECTO = Path(__file__).resolve().parent
CARPETA_CAPTURAS = RAIZ_PROYECTO / "capturas_precintos"

CANAL_RTSP_DEFAULT = "101"
CANAL_RTSP_MEDIO = "102"
PUERTO_RTSP = 554
VEL_FOCO = 5

def asegurar_carpeta_capturas() -> Path:
    CARPETA_CAPTURAS.mkdir(parents=True, exist_ok=True)
    return CARPETA_CAPTURAS
