"""Estructuras de resultados usadas por el lector y las dos aplicaciones."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Detection:
    raw_text: str
    normalized_text: str
    confidence: float
    box: list[list[float]]
    variant: str
    rotation: float
    observation: str


@dataclass
class AnalysisResult:
    image_id: str
    input_mode: str
    status: str = "sin_texto"
    ocr_calls: int = 0
    stages: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    reason: str = ""
    error: str | None = None
    config: dict = field(default_factory=dict)
    schema_version: int = 4
    engine_version: str = "tesseract-full-frame-v4"
    evidence_kind: str = "visual_ocr"
    detections: list[Detection] = field(default_factory=list)
    source_shape: list[int] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
