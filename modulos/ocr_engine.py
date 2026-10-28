"""
Módulo de Reconocimiento Óptico de Caracteres (OCR)

 Instanciación Única y Thread-Safe (Singleton)
   - Carga el motor PaddleOCR en memoria una sola vez para evitar sobrecarga y bloqueos

 Preprocesamiento Avanzado de Imagen
   - Genera múltiples filtros sobre la imagen filtro bilateral para maximizar la legibilidad

 Pipeline de Reconocimiento resiliente con Salida Temprana
   - Intenta primero una lectura directa rápida. Si cumple con un umbral alto de confianza  y longitud, devuelve el resultado al instante para ahorrar cómputo.
   - Si no es concluyente, prueba en cascada todas las variantes preprocesadas y rotaciones, seleccionando el texto con mayor puntuación ponderada

Uso del Scoring queda PaddleOCR:
   - Evalúa y premia la confianza del OCR, la cantidad de caracteres alfanuméricos y el ajuste a patrones típicos de precintos de seguridad
"""

import logging
import re
import threading
from collections.abc import Iterator

import cv2
import numpy as np
from paddleocr import PaddleOCR

logging.getLogger("ppocr").setLevel(logging.ERROR)

_ocr = None
_ocr_lock = threading.Lock()
# El singleton protege la carga; este lock protege las inferencias reales si
# en el futuro varios consumidores comparten el mismo predictor nativo.
_inferencia_lock = threading.Lock()

SCORE_RAPIDO_UMBRAL = 0.88
MIN_CHARS_RAPIDO = 5
PATRON_PRECINTO = re.compile(r"[A-Z0-9][A-Z0-9\-]{3,}")


def get_ocr() -> PaddleOCR:
    """Carga PaddleOCR una sola vez, fuera del hilo de la interfaz."""
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            _ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                rec_algorithm="SVTR_LCNet",
            )
        return _ocr


def warmup_ocr() -> None:
    get_ocr()


def score(conf: float, texto: str) -> float:
    if not texto:
        return 0.0

    texto_upper = texto.upper()
    chars_validos = len(re.sub(r"[^A-Z0-9\-]", "", texto_upper))
    if chars_validos == 0:
        return 0.0

    base = (chars_validos ** 0.7) * (conf ** 1.3)
    if PATRON_PRECINTO.search(texto_upper):
        base *= 1.25
    return base


def _limpiar_texto(texto: str) -> str:
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    if len(lineas) <= 3 and all(len(l) <= 20 for l in lineas):
        return " ".join(lineas)
    return "\n".join(lineas)


def _iter_preprocesar(img: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
    """Mismos píxeles y orden original; calcula sólo lo que llega a utilizarse."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yield "Original", g

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(g)
    yield "CLAHE", clahe_img

    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    yield "Sharp", cv2.filter2D(g, -1, kernel_sharp)

    yield "Bilateral", cv2.bilateralFilter(g, 9, 75, 75)

    _, binarizada = cv2.threshold(
        clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    yield "Otsu-CLAHE", binarizada

    adaptativa = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    yield "AdaptThresh", adaptativa


def preprocesar(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Interfaz original conservada para los consumidores existentes."""
    return list(_iter_preprocesar(img))


def _rotar_90(img: np.ndarray, sentido: str) -> np.ndarray:
    if sentido == "izquierda":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def ejecutar_ocr(imagen: np.ndarray, *, conservar_fragmentos=False):
    """Resultado nativo completo, sin descartar cajas ni confianzas por línea."""
    engine = get_ocr()
    with _inferencia_lock:
        # PaddleOCR 2.7 descarta score < drop_score (0.5) en __call__.
        # El análisis conserva esos fragmentos como evidencia; nunca los acepta
        # por bajar este filtro. El umbral de decisión sigue siendo independiente.
        original = getattr(engine, "drop_score", None)
        if conservar_fragmentos and original is not None:
            engine.drop_score = 0.0
        try:
            return engine.ocr(imagen, cls=True)
        finally:
            if conservar_fragmentos and original is not None:
                engine.drop_score = original


def detectar_texto(imagen: np.ndarray):
    """Sólo detector Paddle, sin reconocimiento ni segundo predictor."""
    engine = get_ocr()
    with _inferencia_lock:
        # En PaddleOCR 2.7.3 ocr(rec=False) hace `if not dt_boxes` sobre
        # ndarray y falla cuando hay texto. Usar el mismo detector nativo,
        # sin ese wrapper y sin cargar otra red ni ejecutar reconocimiento.
        boxes, _elapsed = engine.text_detector(imagen)
        return [boxes.tolist() if boxes is not None and len(boxes) else None]


def _ocr_imagen(imagen: np.ndarray) -> tuple[str, float]:
    res = ejecutar_ocr(imagen)
    if not res or not res[0]:
        return "", 0.0
    lineas = res[0]
    texto = "\n".join(l[1][0] for l in lineas)
    conf = sum(l[1][1] for l in lineas) / len(lineas)
    return texto, conf


def mejor_ocr(img: np.ndarray, *, read=None) -> tuple[str, float, str, float]:
    """Compatibilidad histórica y comparación de línea base.

    OcrWorker usa ocr_reader para anexar evidencia mediante ``read`` sin cambiar
    el recorrido original. Esta API no autoriza confirmaciones operativas.
    """
    mejor_texto = ""
    mejor_score = -1.0
    mejor_metodo = "ninguno"
    mejor_conf = 0.0

    def _leer(nombre, imagen, rotation=0):
        return read(imagen, nombre, rotation) if read is not None else _ocr_imagen(imagen)

    def _evaluar(nombre: str, imagen: np.ndarray, rotation=0) -> bool:
        nonlocal mejor_texto, mejor_score, mejor_metodo, mejor_conf
        texto, conf = _leer(nombre, imagen, rotation)
        if not texto:
            return False
        s = score(conf, texto)
        if s > mejor_score:
            mejor_score = s
            mejor_texto = _limpiar_texto(texto)
            mejor_metodo = nombre
            mejor_conf = conf
        chars_validos = len(re.sub(r"[^A-Z0-9\-]", "", texto.upper()))
        return conf >= SCORE_RAPIDO_UMBRAL and chars_validos >= MIN_CHARS_RAPIDO

    texto_rapido, conf_rapido = _leer('Directo-BGR', img)
    if texto_rapido:
        s = score(conf_rapido, texto_rapido)
        mejor_score = s
        mejor_texto = _limpiar_texto(texto_rapido)
        mejor_metodo = "Directo-BGR"
        mejor_conf = conf_rapido
        chars_validos = len(re.sub(r"[^A-Z0-9\-]", "", texto_rapido.upper()))
        if conf_rapido >= SCORE_RAPIDO_UMBRAL and chars_validos >= MIN_CHARS_RAPIDO:
            return mejor_texto, mejor_score, mejor_metodo + " [early-exit]", mejor_conf

    for nombre, imagen_proc in _iter_preprocesar(img):
        if _evaluar(f"Proc:{nombre}", imagen_proc):
            return mejor_texto, mejor_score, mejor_metodo + " [early-exit]", mejor_conf

    for sentido in ("izquierda", "derecha"):
        img_rot = _rotar_90(img, sentido)
        rotation = -90 if sentido == 'izquierda' else 90
        if _evaluar(f"Rot90{sentido.capitalize()}-BGR", img_rot, rotation):
            return mejor_texto, mejor_score, mejor_metodo + " [early-exit]", mejor_conf
        for nombre, imagen_proc in _iter_preprocesar(img_rot):
            if _evaluar(f"Rot90{sentido.capitalize()}-{nombre}", imagen_proc, rotation):
                return mejor_texto, mejor_score, mejor_metodo + " [early-exit]", mejor_conf

    return mejor_texto, mejor_score, mejor_metodo, mejor_conf
