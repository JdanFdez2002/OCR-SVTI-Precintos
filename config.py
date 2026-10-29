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
ARCHIVO_REGISTRO_LECTURAS = RAIZ_PROYECTO / "registro_lecturas.sqlite3"
ARCHIVO_REPORTE_LECTURAS = RAIZ_PROYECTO / "reporte_lecturas.xlsx"

# El canal 101 queda reservado para fotografías de alta resolución/OCR.
# Sondeo de solo lectura realizado sobre la cámara configurada:
#   103: sin frames frescos
#   102: 704x576, adecuado para visualización/PTZ
#   101: 2688x1520, adecuado para fotografía OCR
CANAL_RTSP_FOTO = "101"
CANAL_RTSP_VIDEO = "102"

# Alias históricos conservados para consumidores y pruebas existentes.
CANAL_RTSP_DEFAULT = CANAL_RTSP_FOTO
CANAL_RTSP_MEDIO = CANAL_RTSP_VIDEO
PUERTO_RTSP = 554
# PTZ continuo de la cÃ¡mara configurada: -100..100 para paneo, inclinaciÃ³n
# y zoom. La UI usa el mÃ³dulo positivo 1..100; el signo indica direcciÃ³n.
VEL_MOV_MIN = 1
VEL_MOV_MAX = 100
VEL_ZOOM_MIN = 1
VEL_ZOOM_MAX = 100
VEL_ZOOM = VEL_ZOOM_MAX
# La cÃ¡mara no publica focusSpeed en sus capacidades disponibles. Se usa
# el rango conservador 1..10 de la capacidad ISAPI general; validar efecto
# Ã³ptico en terreno.
VEL_FOCO_MIN = 1
VEL_FOCO_MAX = 10
VEL_FOCO = VEL_FOCO_MAX

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
