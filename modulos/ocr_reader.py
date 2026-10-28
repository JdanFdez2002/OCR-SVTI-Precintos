"""Reconocimiento visual compartido. Nunca recibe manifiestos ni códigos esperados.

La observación completa se conserva para mostrarla; los posibles identificadores
son evidencia separada y no sustituyen el texto observado por una caja corta.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import math
import time

import numpy as np
import cv2

from modulos.ocr_analysis import AnalysisResult, Candidate, Detection, _original_box, _overlap, normalized


@dataclass
class RecognitionResult(AnalysisResult):
    recognized_text: str = ''
    selected_variant: str = 'ninguno'
    confidence: float = 0.
    selection_score: float = 0.
    raw_observations: list[dict] = field(default_factory=list)
    stop_reason: str = ''
    budget_exhausted: bool = False


@dataclass(frozen=True)
class ReaderConfig:
    max_calls: int = 6
    max_seconds: float = 5.0
    confidence: float = .88
    orientations: tuple[int,...] = (0,90,-90)

    def __post_init__(self):
        if not 1 <= self.max_calls <= 21 or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError('Presupuesto OCR inválido')
        if not 0 <= self.confidence <= 1 or set(self.orientations) != {0,90,-90} or len(self.orientations) != 3:
            raise ValueError('Orientaciones/confianza inválidas')


def _record(result, raw, variant, rotation, shape, elapsed):
    detections = []
    observation = str(len(result.raw_observations))
    for line in (raw[0] or []) if raw else []:
        box, (text, confidence) = line
        confidence = float(confidence)
        if not isinstance(text, str) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Respuesta OCR inválida')
        if text.strip():
            detection = Detection(text, normalized(text), confidence, _original_box(box, rotation, shape),
                                  variant, rotation, observation)
            detections.append(detection)
            result.detections.append(detection)
    text = '\n'.join(d.raw_text for d in detections)
    confidence = sum(d.confidence for d in detections)/len(detections) if detections else 0.
    result.raw_observations.append(dict(variant=variant,rotation=rotation,text=text,
                                       confidence=confidence,elapsed_s=elapsed,
                                       detection_indices=list(range(len(result.detections)-len(detections),len(result.detections)))))
    result.ocr_calls += 1
    return text, confidence


def _candidates(detections):
    groups = []
    for detection in detections:
        for group in groups:
            if group[0].normalized_text == detection.normalized_text and _overlap(group[0].box,detection.box)>=.3:
                group.append(detection)
                break
        else:
            groups.append([detection])
    return [Candidate(g[0].normalized_text,sum(d.confidence for d in g)/len(g),
                      sum(d.confidence for d in g)/len(g),min(d.confidence for d in g),
                      len({d.observation for d in g}),True,[],g) for g in groups]


def _key(text):
    # Sólo equivalencia de espaciado para contrastar observaciones visuales.
    # No se usa para comparación contra manifiesto ni para alterar el texto.
    return ''.join(normalized(text).split())


def _extension(longer,shorter,threshold=.88):
    a,b = _key(longer.raw_text),_key(shorter.raw_text)
    return (longer.observation != shorter.observation and longer.confidence >= threshold
            and shorter.confidence >= threshold and len(a)>len(b) and b in a
            and _overlap(longer.box,shorter.box)>=.35)


def _snapshot_lines(result,snapshot,minimum=.5):
    return [result.detections[i] for i in snapshot['detection_indices'] if result.detections[i].confidence >= minimum]


def _best_snapshot(result,config):
    snapshots = [s for s in result.raw_observations if _snapshot_lines(result,s)]
    if not snapshots:
        return result.raw_observations[0] if result.raw_observations else None
    def quality(snapshot):
        lines = _snapshot_lines(result,snapshot)
        partial = sum(any(_extension(other,d,config.confidence) for other in result.detections) for d in lines)
        # No premio por número de caracteres o de líneas. Una lectura truncada
        # de la MISMA región no desplaza a una observación completa y confiable.
        return (-partial,sum(d.confidence for d in lines)/len(lines))
    return max(snapshots,key=quality)


def _enough_evidence(result,snapshot,config):
    if len(result.raw_observations)<2 or snapshot is None:
        return False
    lines = _snapshot_lines(result,snapshot)
    if not lines or sum(d.confidence for d in lines)/len(lines)<config.confidence:
        return False
    height,width = result.source_shape[:2]
    for d in lines:
        points = np.asarray(d.box)
        if (points[:,0].min()<=2 or points[:,1].min()<=2 or points[:,0].max()>=width-3 or points[:,1].max()>=height-3):
            return False
        if any(_extension(other,d,config.confidence) for other in result.detections):
            return False
        if any(other.observation!=d.observation and other.confidence>=config.confidence
               and _overlap(other.box,d.box)>=.3 and _key(other.raw_text)!=_key(d.raw_text)
               and not _extension(d,other,config.confidence) for other in result.detections):
            return False
        if d.confidence>=config.confidence and not any(
                other.observation!=d.observation and other.confidence>=config.confidence
                and _key(other.raw_text)==_key(d.raw_text) and _overlap(other.box,d.box)>=.3
                for other in result.detections):
            return False
    return True


def _rotate(image,rotation):
    if rotation==90:
        return cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    if rotation==-90:
        return cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _fast_stages(image,result,config):
    from modulos.ocr_engine import _iter_preprocesar
    for rotation in config.orientations:
        yield 'BGR',rotation,_rotate(image,rotation)
    best = _best_snapshot(result,config)
    rotation = best['rotation'] if best else 0
    for variant,array in _iter_preprocesar(_rotate(image,rotation)):
        yield variant,rotation,array


def recognize_fast(image, *, image_id='',input_mode='full_photo',config=None,infer=None,cancelled=None):
    """Plan acotado: orientaciones BGR, contraste sólo si falta evidencia.

    El tiempo es un límite entre llamadas: no interrumpe un predictor nativo.
    Una lectura incierta sigue devolviendo todo el texto observado y alternativas.
    """
    from modulos.ocr_engine import ejecutar_ocr
    config = config or ReaderConfig()
    result = RecognitionResult(image_id,input_mode,engine_version='bounded-visual-reader-v1',
                               schema_version=3,config=asdict(config))
    start = time.perf_counter()
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or not image.size:
        result.status = 'imagen_invalida'
        return result
    result.source_shape = list(image.shape)
    infer = infer or (lambda array:ejecutar_ocr(array,conservar_fragmentos=True))
    seen = set()
    stages = iter(_fast_stages(image,result,config))
    try:
        while True:
            if cancelled and cancelled():
                result.status,result.stop_reason = 'cancelado','cancelled'
                break
            if result.ocr_calls>=config.max_calls or time.perf_counter()-start>=config.max_seconds:
                result.budget_exhausted = True
                result.stop_reason = 'max_calls' if result.ocr_calls>=config.max_calls else 'time_budget'
                break
            prep_start = time.perf_counter()
            try:
                variant,rotation,array = next(stages)
            except StopIteration:
                result.stop_reason = 'stages_exhausted'
                break
            key = (array.shape,hashlib.sha256(memoryview(np.ascontiguousarray(array))).digest())
            if key in seen:
                continue
            seen.add(key)
            if time.perf_counter()-start>=config.max_seconds:
                result.budget_exhausted,result.stop_reason = True,'time_budget'
                break
            call_start = time.perf_counter()
            raw = infer(array)
            _record(result,raw,variant,rotation,image.shape,time.perf_counter()-call_start)
            result.stages.append(dict(variant=variant,rotation=rotation,preprocessing_s=call_start-prep_start,
                                      ocr_s=time.perf_counter()-call_start))
            best = _best_snapshot(result,config)
            if _enough_evidence(result,best,config):
                result.stop_reason = 'visual_consensus'
                break
    except Exception as exc:
        result.status,result.error,result.stop_reason = 'error_tecnico',f'{type(exc).__name__}: {exc}','error'
    best = _best_snapshot(result,config)
    if best:
        lines = _snapshot_lines(result,best)
        result.recognized_text = '\n'.join(d.raw_text for d in lines) or best['text']
        result.suggested_text = result.recognized_text
        result.confidence = sum(d.confidence for d in lines)/len(lines) if lines else best['confidence']
        result.selection_score = result.confidence
        result.selected_variant = f"{best['variant']} ({best['rotation']}°)"
    result.candidates = _candidates(result.detections)
    for candidate in result.candidates:
        if any(_extension(other,d,config.confidence) for d in candidate.detections for other in result.detections):
            candidate.warnings.append('possible_partial_same_region')
    if result.status not in {'cancelado','error_tecnico'}:
        result.status = ('texto_reconocido' if result.stop_reason=='visual_consensus' else
                         'lectura_incierta' if result.recognized_text else 'sin_texto')
    result.reason = 'Texto visual conservado; el presupuesto y la incertidumbre no autorizan llegadas.'
    result.elapsed_s = time.perf_counter()-start
    return result


def recognize_original(image, *, image_id='', input_mode='full_photo', infer=None, cancelled=None):
    """Recorrido original con las mismas salidas y cajas observadas anexadas."""
    from modulos.ocr_engine import ejecutar_ocr, mejor_ocr
    result = RecognitionResult(image_id,input_mode,engine_version='original-traced-v1',schema_version=3)
    start = time.perf_counter()
    if not isinstance(image,np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        result.status = 'imagen_invalida'
        return result
    result.source_shape = list(image.shape)
    infer = infer or ejecutar_ocr
    def read(array,variant,rotation):
        if cancelled and cancelled():
            raise InterruptedError('Solicitud invalidada')
        call_start = time.perf_counter()
        raw = infer(array)
        return _record(result,raw,variant,rotation,image.shape,time.perf_counter()-call_start)
    try:
        text,score,variant,confidence = mejor_ocr(image,read=read)
        result.recognized_text = result.suggested_text = text
        result.confidence,result.selected_variant = confidence,variant
        result.selection_score = score
        result.status = ('texto_reconocido' if confidence>=.88 else 'lectura_incierta') if text else 'sin_texto'
        result.stop_reason = 'original_early_exit' if 'early-exit' in variant else 'original_exhausted'
        result.reason = 'Lectura visual original; los identificadores candidatos y la comparación se mantienen separados.'
    except InterruptedError:
        result.status = 'cancelado'
    except Exception as exc:
        result.status,result.error = 'error_tecnico',f'{type(exc).__name__}: {exc}'
    result.candidates = _candidates(result.detections)
    result.elapsed_s = time.perf_counter()-start
    return result


def recognize(image, *, image_id='', input_mode='full_photo', cancelled=None):
    from config import OCR_ENGINE, OCR_MAX_CALLS, OCR_MAX_SECONDS
    if OCR_ENGINE == 'experimental':
        from modulos.ocr_analysis import analyze
        return analyze(image,image_id=image_id,input_mode=input_mode,cancelled=cancelled)
    if OCR_ENGINE == 'fast':
        return recognize_fast(image,image_id=image_id,input_mode=input_mode,cancelled=cancelled,
                              config=ReaderConfig(max_calls=OCR_MAX_CALLS,max_seconds=OCR_MAX_SECONDS))
    if OCR_ENGINE != 'original':
        raise ValueError('SEAI_OCR_ENGINE debe ser original, fast o experimental')
    return recognize_original(image,image_id=image_id,input_mode=input_mode,cancelled=cancelled)
