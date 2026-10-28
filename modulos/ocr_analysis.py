"""Estructuras de compatibilidad para el OCR actualmente desactivado.

Este módulo no reconoce texto, no preprocesa imágenes y no selecciona ni
valida candidatos. Las estructuras se conservan para que la aplicación pueda
mostrar un resultado vacío y registrar que el lector aún no está implementado.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time


OCR_NO_IMPLEMENTADO = "El reconocimiento OCR no está implementado todavía."


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuración heredada aceptada únicamente por compatibilidad."""

    confidence: float = 0.88
    confirmations: int = 2
    margin: float = 0.05
    typical_length: int = 15
    excess_length_penalty: float = 0.01
    max_calls: int = 21


@dataclass
class Detection:
    raw_text: str
    normalized_text: str
    confidence: float
    box: list[list[float]]
    variant: str
    rotation: int
    observation: str
    fragments: list[str] = field(default_factory=list)
    fragment_boxes: list = field(default_factory=list)


@dataclass
class Candidate:
    text: str
    score: float
    confidence_mean: float
    confidence_min: float
    observations: int
    syntax_compatible: bool
    warnings: list[str]
    detections: list[Detection]


@dataclass
class AnalysisResult:
    image_id: str
    input_mode: str
    status: str = "no_implementado"
    confirmed_identifier: str | None = None
    suggested_text: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    ocr_calls: int = 0
    stages: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    reason: str = OCR_NO_IMPLEMENTADO
    error: str | None = None
    config: dict = field(default_factory=dict)
    schema_version: int = 3
    engine_version: str = "disabled"
    evidence_kind: str = "ocr_not_implemented"
    detections: list[Detection] = field(default_factory=list)
    source_shape: list[int] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def normalized(text: str) -> str:
    """Compatibilidad mínima: no corrige caracteres ni aplica formatos."""
    return text.strip()


def _original_box(points, rotation, shape):
    """Compatibilidad mínima sin rotar ni reinterpretar coordenadas."""
    return points


def _overlap(a, b) -> float:
    """Compatibilidad mínima; no se agrupan detecciones."""
    return 0.0


def join_fragments(detections):
    """No concatena fragmentos mientras no exista lector OCR."""
    return []


def candidates_from(detections, config=None):
    """No crea candidatos ni aplica reglas de selección."""
    return []


def _decide(candidates, config=None, input_mode="full_photo"):
    return "no_implementado", None, OCR_NO_IMPLEMENTADO


def _stages(image):
    return iter(())


def analyze(image, *, input_mode="full_photo", image_id="", config=None,
            infer=None, cancelled=None) -> AnalysisResult:
    """Devuelve un resultado vacío sin invocar inferencia."""
    start = time.perf_counter()
    if cancelled is not None and cancelled():
        result = AnalysisResult(
            image_id,
            input_mode,
            status="cancelado",
            reason="Solicitud invalidada por una acción posterior",
        )
    else:
        result = AnalysisResult(
            image_id,
            input_mode,
            config=asdict(config or AnalysisConfig()),
        )
    shape = getattr(image, "shape", None)
    if shape is not None:
        result.source_shape = list(shape)
    result.elapsed_s = time.perf_counter() - start
    return result
