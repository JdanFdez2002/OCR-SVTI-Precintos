"""Interfaz de lector OCR desactivada temporalmente.

No carga modelos, no transforma imágenes y no devuelve texto inventado. Las
funciones conservan los nombres usados por el worker y por herramientas
históricas para facilitar la implementación posterior de un lector genérico.
"""

from dataclasses import asdict, dataclass, field

from modulos.ocr_analysis import AnalysisResult, OCR_NO_IMPLEMENTADO


@dataclass
class RecognitionResult(AnalysisResult):
    recognized_text: str = ""
    selected_variant: str = "no_implementado"
    confidence: float = 0.0
    selection_score: float = 0.0
    raw_observations: list[dict] = field(default_factory=list)
    stop_reason: str = "ocr_disabled"
    budget_exhausted: bool = False


@dataclass(frozen=True)
class ReaderConfig:
    max_calls: int = 6
    max_seconds: float = 5.0
    confidence: float = 0.88
    orientations: tuple[int, ...] = (0, 90, -90)


def _empty_result(image, *, image_id="", input_mode="full_photo", config=None,
                  cancelled=None) -> RecognitionResult:
    if cancelled is not None and cancelled():
        return RecognitionResult(
            image_id,
            input_mode,
            status="cancelado",
            reason="Solicitud invalidada por una acción posterior",
            engine_version="disabled",
        )
    result = RecognitionResult(
        image_id,
        input_mode,
        config=asdict(config or ReaderConfig()),
        reason=OCR_NO_IMPLEMENTADO,
        engine_version="disabled",
        evidence_kind="ocr_not_implemented",
    )
    shape = getattr(image, "shape", None)
    if shape is not None:
        result.source_shape = list(shape)
    return result


def recognize_fast(image, *, image_id="", input_mode="full_photo", config=None,
                   infer=None, cancelled=None):
    return _empty_result(
        image,
        image_id=image_id,
        input_mode=input_mode,
        config=config,
        cancelled=cancelled,
    )


def recognize_original(image, *, image_id="", input_mode="full_photo", infer=None,
                       cancelled=None):
    return _empty_result(
        image,
        image_id=image_id,
        input_mode=input_mode,
        cancelled=cancelled,
    )


def recognize(image, *, image_id="", input_mode="full_photo", cancelled=None):
    """Punto único de entrada: devuelve siempre OCR no implementado."""
    return _empty_result(
        image,
        image_id=image_id,
        input_mode=input_mode,
        cancelled=cancelled,
    )
