"""Comparación completa y conservadora; las sugerencias nunca confirman llegadas."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
import math


def normalize_identifier(text: str) -> str:
    # Sólo ASCII: upper() de Unicode puede expandir caracteres (ß -> SS).
    text = text.strip().translate(str.maketrans('abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    if not re.fullmatch(r'[A-Z0-9 -]+', text) or not any(c.isalnum() for c in text):
        return ''
    return re.sub(r' +', ' ', text)


@dataclass(frozen=True)
class ArrivalDecision:
    accepted: bool
    row: int | None
    reason: str
    raw_text: str | None
    expected: str | None = None
    previous_state: str | None = None
    next_state: str | None = None


def decide_arrival(records, analysis, min_confidence=.88):
    raw = analysis.confirmed_identifier
    if analysis.status != 'reconocido' or analysis.input_mode != 'seal_roi' or not raw:
        observed = getattr(analysis,'recognized_text','') or raw or analysis.suggested_text
        return ArrivalDecision(False, None, 'Revisión manual requerida; sin identificador visual único autorizado para llegada', observed)
    candidates = [c for c in analysis.candidates if c.text == raw]
    if len(candidates) != 1:
        return ArrivalDecision(False, None, 'Confianza insuficiente o candidato no único', raw)
    candidate = candidates[0]
    if (not math.isfinite(candidate.confidence_min) or not min_confidence <= candidate.confidence_min <= 1
            or candidate.observations < 2 or not candidate.syntax_compatible
            or 'cerca_del_borde' in candidate.warnings):
        return ArrivalDecision(False, None, 'Evidencia insuficiente, confianza no válida o extremos sin verificar', raw)
    key = normalize_identifier(raw)
    matches = [(i, r) for i, r in enumerate(records)
               if key and normalize_identifier(r.precinto_esperado) == key]
    if len(matches) != 1:
        return ArrivalDecision(False, None, 'Sin coincidencia completa única en manifiesto', raw)
    row, record = matches[0]
    return ArrivalDecision(True, row, 'Coincidencia completa; sin sustitución de caracteres', raw,
                           record.precinto_esperado, record.estado, 'Llegado')


def append_decision(path, decision, evidence):
    event = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
             'policy': 'complete-identifier-v1-preserve-spaces-hyphens',
             **asdict(decision), 'evidence': evidence}
    # Si no se pudo conservar la decisión, el llamador no debe cambiar el estado.
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
