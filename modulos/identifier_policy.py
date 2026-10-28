"""Compatibilidad de manifiesto con el OCR desactivado.

No compara identificadores ni cambia estados automáticamente. La interfaz se
mantiene para que la UI pueda registrar que se requiere revisión manual.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json


def normalize_identifier(text: str) -> str:
    """Normalización mínima para claves de UI; no corrige caracteres."""
    return text.strip().upper()


@dataclass(frozen=True)
class ArrivalDecision:
    accepted: bool
    row: int | None
    reason: str
    raw_text: str | None
    expected: str | None = None
    previous_state: str | None = None
    next_state: str | None = None


def decide_arrival(records, analysis, min_confidence=0.88):
    observed = getattr(analysis, "recognized_text", "") or getattr(
        analysis, "suggested_text", None
    )
    return ArrivalDecision(
        False,
        None,
        "OCR no implementado; revisión manual requerida.",
        observed,
    )


def append_decision(path, decision, evidence):
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "ocr-disabled-manual-review-v1",
        **asdict(decision),
        "evidence": evidence,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
