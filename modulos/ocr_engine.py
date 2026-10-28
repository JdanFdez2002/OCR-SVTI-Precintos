"""Punto de compatibilidad para el motor OCR.

El reconocimiento está desactivado mientras se prepara un lector genérico.
Las funciones públicas se conservan para que la UI y herramientas antiguas
puedan importar el módulo sin cargar un modelo ni ejecutar procesamiento.
"""

OCR_IMPLEMENTADO = False
OCR_NO_IMPLEMENTADO = "El reconocimiento OCR no está implementado todavía."


def get_ocr():
    """Compatibilidad histórica; no se carga ningún predictor."""
    return None


def warmup_ocr() -> None:
    """Mantiene la señal de inicialización del worker sin cargar OCR."""
    return None


def score(conf: float, texto: str) -> float:
    """Compatibilidad histórica; no se calcula puntuación de reconocimiento."""
    return 0.0


def _iter_preprocesar(imagen):
    """No genera variantes: el preprocesamiento OCR está desactivado."""
    return iter(())


def preprocesar(imagen):
    """Interfaz histórica que ahora devuelve una lista vacía."""
    return []


def _rotar_90(imagen, sentido: str):
    """Compatibilidad mínima sin transformar la imagen."""
    return imagen


def ejecutar_ocr(imagen, *, conservar_fragmentos=False):
    """Devuelve una respuesta vacía compatible con la interfaz anterior."""
    return [None]


def detectar_texto(imagen):
    """No detecta regiones mientras el lector OCR está desactivado."""
    return [None]


def _ocr_imagen(imagen):
    return "", 0.0


def mejor_ocr(imagen, *, read=None):
    """Compatibilidad histórica sin reconocimiento ni selección de candidatos."""
    return "", 0.0, "no_implementado", 0.0
