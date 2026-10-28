import os
from pathlib import Path

# Debe setearse antes del primer VideoCapture
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

COLOR_RETICULA = (0, 180, 80)
GROSOR_RETICULA = 2
ANCHO_RETICULA_PCT = 0.30
ALTO_RETICULA_PCT = 0.20

RAIZ_PROYECTO = Path(__file__).resolve().parent
CARPETA_CAPTURAS = RAIZ_PROYECTO / "capturas_precintos"

CANAL_RTSP_DEFAULT = "101"
CANAL_RTSP_MEDIO = "102"
PUERTO_RTSP = 554
VEL_FOCO = 5

# Valores iniciales de investigación. Medir por equipo antes de aumentarlos.
DETECCION_INTERVALO_S = float(os.environ.get('SEAI_DETECTION_INTERVAL_S', '2.0'))
SEGUIMIENTO_INTERVALO_S = .25
REGION_MAX_AGE_S = 1.5
OCR_DINAMICO_INTERVALO_S = 4.0
DETECCION_EN_VIVO = os.environ.get('SEAI_LIVE_DETECTION', '0').lower() in {'1', 'true', 'yes'}
OCR_EN_VIVO = os.environ.get('SEAI_LIVE_OCR', '0').lower() in {'1', 'true', 'yes'}
OCR_ENGINE = os.environ.get('SEAI_OCR_ENGINE', 'original')
OCR_MAX_CALLS = int(os.environ.get('SEAI_OCR_MAX_CALLS', '6'))
OCR_MAX_SECONDS = float(os.environ.get('SEAI_OCR_MAX_SECONDS', '5.0'))


def recorte_con_margen(frame, bounds, margen=.35):
    """Margen proporcional a la altura del texto; nunca inventa píxeles."""
    alto, ancho = frame.shape[:2]
    x1, y1, x2, y2 = bounds
    padding = max(4, int((y2 - y1) * margen))
    box = (max(0, int(x1) - padding), max(0, int(y1) - padding),
           min(ancho, int(x2) + padding), min(alto, int(y2) + padding))
    return frame[box[1]:box[3], box[0]:box[2]].copy(), box


def limites_reticula(frame):
    alto, ancho = frame.shape[:2]
    rw, rh = max(1, int(ancho * ANCHO_RETICULA_PCT)), max(1, int(alto * ALTO_RETICULA_PCT))
    return (ancho // 2 - rw // 2, alto // 2 - rh // 2,
            ancho // 2 - rw // 2 + rw, alto // 2 - rh // 2 + rh)


def asegurar_carpeta_capturas() -> Path:
    CARPETA_CAPTURAS.mkdir(parents=True, exist_ok=True)
    return CARPETA_CAPTURAS


def recorte_reticula(frame):
    """Recorta la zona central equivalente a la retícula de la UI."""
    alto, ancho = frame.shape[:2]
    rw = max(1, int(ancho * ANCHO_RETICULA_PCT))
    rh = max(1, int(alto * ALTO_RETICULA_PCT))
    x1 = max(0, ancho // 2 - rw // 2)
    y1 = max(0, alto // 2 - rh // 2)
    x2 = min(ancho, x1 + rw)
    y2 = min(alto, y1 + rh)
    return frame[y1:y2, x1:x2].copy()
