"""OCR sobre la fotografía completa, sin ROI, recortes ni prioridad al centro."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import math
import os
import time

import cv2
import numpy as np

from modulos.ocr_analysis import AnalysisResult, Detection
from modulos.ocr_engine import OcrCancelled, OcrTimeout, get_ocr, read_words
from modulos.ocr_vision import estimate_skew, grayscale, page_view, prepare_view, source_box
from modulos.text_emphasis import emphasize_text


@dataclass
class RecognitionResult(AnalysisResult):
    recognized_text: str = ""
    selected_variant: str = "ninguno"
    confidence: float = 0.0
    selection_score: float = 0.0
    raw_observations: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    budget_exhausted: bool = False
    warnings: list[str] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)
    score_kind: str = "visual_quality_not_probability"
    input_policy: str = "full_frame_with_text_emphasis"
    completed_orientations: list[float] = field(default_factory=list)
    detector_metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ReaderConfig:
    max_seconds: float = 9.0
    max_calls: int = 16
    call_timeout: float = 4.0
    min_word_confidence: float = .45
    confidence: float = .78

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in
                    (self.max_seconds, self.call_timeout, self.min_word_confidence, self.confidence))
                or self.max_seconds <= 0 or self.call_timeout <= 0
                or not 1 <= self.max_calls <= 64
                or not 0 <= self.min_word_confidence <= self.confidence <= 1):
            raise ValueError("Configuración OCR inválida")

    @classmethod
    def from_environment(cls):
        return cls(
            max_seconds=float(os.environ.get("SEAI_OCR_MAX_SECONDS", "9")),
            max_calls=int(os.environ.get("SEAI_OCR_MAX_CALLS", "16")),
            call_timeout=float(os.environ.get("SEAI_OCR_CALL_TIMEOUT", "4")),
            min_word_confidence=float(os.environ.get("SEAI_OCR_MIN_CONFIDENCE", ".45")),
        )


def _confidence(words):
    """Confianza real ponderada; no representa una probabilidad calibrada."""
    weight = sum(max(1, len(word["text"])) for word in words)
    return sum(word["confidence"] * max(1, len(word["text"])) for word in words) / weight if weight else 0.0


def _bounds(words):
    # Coordenadas informativas: nunca se extrae una subimagen de estas cajas.
    polygons = np.float32([point for word in words for point in word["source_box"]])
    return cv2.convexHull(polygons).reshape(-1, 2).astype(float).tolist()


def _snapshot(raw, view, config, psm, index, shape, elapsed, policy="full_frame_native_resolution"):
    method = f"{view.name}, PSM {psm}"
    rotation = round(-math.degrees(math.atan2(view.to_source[1, 0], view.to_source[0, 0])), 3)
    # Se conserva TODO el texto devuelto en esta pasada. La confianza sólo
    # señala incertidumbre: no elimina prefijos, palabras ni partes del encuadre.
    words = [{**word, "source_box": source_box(view, word["box"], shape),
              "accepted": True, "reliable": word["confidence"] >= config.min_word_confidence}
             for word in raw.words]
    groups = {}
    for word in words:
        groups.setdefault(tuple(word["line"]), []).append(word)
    lines = [
        {"text": " ".join(w["text"] for w in line_words),
         "confidence": _confidence(line_words), "words": line_words,
         "box": _bounds(line_words), "observation": index, "variant": method,
         "rotation": rotation}
        for line_words in groups.values()
    ]
    text = "\n".join(line["text"] for line in lines)
    total = sum(max(1, len(w["text"])) for w in words)
    reliable = sum(max(1, len(w["text"])) for w in words if w["reliable"])
    coverage = reliable / total if total else 0.0
    isolated = sum(len(line["text"]) == 1 for line in lines) / max(1, len(lines))
    symbols = sum(not any(c.isalnum() for c in w["text"]) for w in words) / max(1, len(words))
    confidence = _confidence(words)
    quality = max(0.0, confidence * (.25 + .75 * coverage) - .18 * isolated - .15 * symbols)
    return {"id": index, "variant": method, "region": "foto",
            "input_policy": policy,
            "source_bounds": [0, 0, shape[1], shape[0]], "processed_shape": list(view.image.shape),
            "psm": psm, "text": text, "confidence": confidence, "quality": quality,
            "retained_fraction": 1.0 if words else 0.0, "reliable_fraction": coverage,
            "words": words, "lines": lines, "malformed_tsv_rows": raw.malformed_rows,
            "elapsed_s": elapsed, "rotation": rotation, "to_source": view.to_source.tolist()}


def _key(snapshot):
    # Sólo espaciado para comparar pasadas: ningún cambio de caracteres.
    return tuple(snapshot["text"].split())


def _best(snapshots):
    nonempty = [s for s in snapshots if s["text"]]
    if not nonempty:
        return None
    def rank(snapshot):
        # Una letra aislada del fondo con 90% no desplaza automáticamente una
        # línea observada de forma consistente en varias pasadas completas.
        support, confidence = _line_support(snapshot, nonempty)
        isolated = len(snapshot["words"]) == 1 and len(snapshot["words"][0]["text"]) == 1
        ranking = (snapshot["quality"] + .9 * support * math.sqrt(confidence) + .15 * confidence
                   - (.65 if _spatially_incomplete(snapshot, nonempty) else 0.0)
                   - (.25 if isolated else 0.0))
        snapshot["ranking_score"] = ranking
        return ranking
    return max(nonempty, key=rank)


def _line_support(best, snapshots):
    """Estabilidad de líneas en sus coordenadas, NO corrección de caracteres.

Se compara similitud para ordenar observaciones reales. La salida siempre
permanece literal, incluso puntuación errónea, y las discrepancias se conservan.
"""
    if best.get("_support_count") == len(snapshots):
        return best["support_score"], best["supported_line_confidence"]
    strongest, confidence = 0.0, 0.0
    for line in best["lines"]:
        total = sum(max(1, len(word["text"])) for word in line["words"])
        support = 0.0
        for word in line["words"]:
            if len(word["text"]) < 3:
                continue
            first = np.float32(word["source_box"])
            agreement = 0.0
            for other in snapshots:
                if other is best:
                    continue
                for candidate in other["words"]:
                    if len(candidate["text"]) < 3 or abs(len(candidate["text"]) - len(word["text"])) > 2:
                        continue
                    ratio = SequenceMatcher(None, word["text"], candidate["text"], autojunk=False).ratio()
                    if ratio < .8:
                        continue
                    second = np.float32(candidate["source_box"])
                    area = min(abs(cv2.contourArea(first)), abs(cv2.contourArea(second)))
                    if area and cv2.intersectConvexConvex(first, second)[0] / area >= .5:
                        agreement = max(agreement, ratio)
            support += len(word["text"]) * agreement
        fraction = support / total if total else 0.0
        if fraction * math.sqrt(line["confidence"]) > strongest * math.sqrt(confidence):
            strongest, confidence = fraction, line["confidence"]
    best["_support_count"] = len(snapshots)
    best["support_score"] = strongest
    best["supported_line_confidence"] = confidence
    return strongest, confidence


def _same_word_region(word, candidate):
    if min(len(word["text"]), len(candidate["text"])) < 3:
        return False
    if SequenceMatcher(None, word["text"], candidate["text"], autojunk=False).ratio() < .8:
        return False
    first, second = np.float32(word["source_box"]), np.float32(candidate["source_box"])
    area = min(abs(cv2.contourArea(first)), abs(cv2.contourArea(second)))
    return bool(area and cv2.intersectConvexConvex(first, second)[0] / area >= .5)


def _spatially_incomplete(snapshot, others):
    """Detecta una omisión respaldada por otra lectura, sin completar texto."""
    if snapshot.get("_coverage_count") == len(others):
        return snapshot["spatially_incomplete"]
    missing = False
    for line in snapshot["lines"]:
        polygon = np.float32(line["box"])
        for other in others:
            if other is snapshot:
                continue
            for other_line in other["lines"]:
                if not any(_same_word_region(word, candidate)
                           for word in line["words"] for candidate in other_line["words"]):
                    continue
                # El texto adicional debe estar realmente observado con buena
                # confianza y fuera de la caja que la lectura corta conservó.
                for candidate in other_line["words"]:
                    if candidate["confidence"] < .65:
                        continue
                    center = np.float32(candidate["source_box"]).mean(axis=0)
                    if cv2.pointPolygonTest(polygon, (float(center[0]), float(center[1])), False) < 0:
                        missing = True
                        break
                if missing:
                    break
            if missing:
                break
        if missing:
            break
    snapshot["_coverage_count"] = len(others)
    snapshot["spatially_incomplete"] = missing
    return missing


def _finish(result, snapshots, config):
    best = _best(snapshots)
    if best is not None:
        # Una salida completa observada, sin armar un texto nuevo con trozos
        # de diferentes lecturas. Las otras pasadas se conservan como alternativas.
        result.recognized_text = best["text"]
        result.confidence = best["confidence"]
        result.selection_score = best["quality"]
        result.selected_variant = best["variant"]
        result.alternatives = [
            {"observation": s["id"], "text": s["text"], "confidence": s["confidence"],
             "variant": s["variant"]}
            for s in sorted(snapshots, key=lambda s: s["quality"], reverse=True)
            if s["text"] and _key(s) != _key(best)
        ]
        if any(not word["reliable"] for word in best["words"]):
            result.warnings.append("Hay palabras de baja confianza; se conservaron sin corregir ni eliminar.")
        if best.get("spatially_incomplete"):
            result.warnings.append("Otra observación conserva texto contiguo que esta lectura pudo omitir.")
        if any(s["text"] and _key(s) != _key(best) and s["quality"] >= best["quality"] - .06
               for s in snapshots):
            result.warnings.append("Las pasadas completas discrepan; revisar las alternativas.")
        result.detections = [
            Detection(word["text"], word["text"], word["confidence"], word["source_box"],
                      best["variant"], best["rotation"], str(best["id"]))
            for word in best["words"]
        ]
        uncertain = (result.confidence < config.confidence or result.warnings or
                     result.budget_exhausted or result.selection_score < .65)
        result.status = "lectura_incierta" if uncertain else "texto_reconocido"
        result.reason = "Lectura de toda la fotografía, sin recortes ni corrección de caracteres."
    elif not snapshots:
        result.status = "error_tecnico"
        result.error = "No se completó ninguna pasada OCR dentro del tiempo disponible."
        result.reason = result.error
    else:
        result.status = "lectura_incierta" if result.budget_exhausted else "sin_texto"
        result.reason = "No se obtuvo texto en las pasadas completas; no se inventaron caracteres."
    if result.budget_exhausted:
        result.reason += " Presupuesto agotado; pueden faltar orientaciones o variantes."
    result.warnings = list(dict.fromkeys(result.warnings))


def recognize(image, *, image_id="", input_mode="full_photo", cancelled=None, config=None):
    """Mismo encuadre íntegro para MotorWindow y captura manual."""
    start = time.monotonic()
    config = config or ReaderConfig.from_environment()
    deadline = start + config.max_seconds
    result = RecognitionResult(image_id, input_mode, config=asdict(config),
                               engine_version="tesseract-full-frame-v4", status="sin_texto",
                               reason="", evidence_kind="visual_ocr")
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0
            or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3)):
        raise ValueError("Se requiere una imagen BGR o gris uint8 no vacía")
    result.source_shape = list(image.shape)
    snapshots = []
    executable = get_ocr()

    def user_cancelled():
        return bool(cancelled and cancelled())

    def available():
        if user_cancelled():
            raise OcrCancelled("Solicitud invalidada")
        if time.monotonic() >= deadline or result.ocr_calls >= config.max_calls:
            result.budget_exhausted = True
            return False
        return True

    def attempt(gray, angle=0, mode="gray", psm=11, emphasis=None):
        if not available():
            return None
        view = page_view(gray, angle)
        policy = "full_frame_native_resolution"
        if emphasis is not None:
            view.to_source = emphasis.to_source @ view.to_source
            view.name += "/" + emphasis.name.split("/", 1)[-1]
            policy = "full_frame_text_emphasis"
        prepared = prepare_view(view, mode)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            result.budget_exhausted = True
            return None
        result.ocr_calls += 1
        call_start = time.monotonic()
        stage = {"variant": prepared.name, "region": "foto", "psm": psm,
                 "input_policy": policy, "processed_shape": list(prepared.image.shape)}
        result.stages.append(stage)
        try:
            raw = read_words(prepared.image, executable=executable, psm=psm,
                             timeout=min(config.call_timeout, remaining), cancelled=user_cancelled)
        except OcrTimeout:
            stage["status"] = "timeout"
            result.warnings.append("Alguna pasada de la imagen completa superó su tiempo límite.")
            return None
        finally:
            stage["elapsed_s"] = time.monotonic() - call_start
        snapshot = _snapshot(raw, prepared, config, psm, len(snapshots), image.shape,
                             stage["elapsed_s"], policy)
        snapshot["page_angle"] = angle
        snapshots.append(snapshot)
        result.raw_observations.append(snapshot)
        if angle not in result.completed_orientations:
            result.completed_orientations.append(angle)
        if raw.malformed_rows:
            result.warnings.append("Se descartaron filas TSV inválidas; consultar la evidencia.")
        return snapshot

    try:
        if not available():
            return result
        gray = grayscale(image)
        emphasis = None
        if os.environ.get("SEAI_TEXT_EMPHASIS", "1").lower() not in {"0", "false", "no"}:
            emphasis, result.detector_metadata = emphasize_text(image, cancelled=user_cancelled)
            if not result.detector_metadata.get("available", True):
                result.warnings.append(result.detector_metadata["reason"])
        def read_emphasis(emphasis, metadata):
            angles = metadata["angles"]
            # Todas las propuestas permanecen en UN encuadre completo. Sólo
            # cambia el fondo; nunca se envían recortes independientes.
            for angle, psm in ((angles[0], 6), (angles[0] + 180, 6),
                               (angles[0], 11), (angles[0] + 180, 11)):
                if not available():
                    break
                attempt(emphasis.image, angle, psm=psm, emphasis=emphasis)
                best = _best(snapshots)
                if best:
                    support, line_confidence = _line_support(best, [s for s in snapshots if s["text"]])
                else:
                    support, line_confidence = 0.0, 0.0
                if support >= .6 and line_confidence >= .65:
                    result.stop_reason = "stable_full_frame_words"
                    break
        if emphasis is not None:
            read_emphasis(emphasis, result.detector_metadata)
        if emphasis is not None and not result.stop_reason and available():
            detailed = any(max(cv2.minAreaRect(np.float32(item["box"]))[1]) /
                           max(1, min(cv2.minAreaRect(np.float32(item["box"]))[1])) >= 3
                           for item in result.detector_metadata.get("boxes", []))
            if detailed:
                alternate_binary, metadata = emphasize_text(
                    image, cancelled=user_cancelled, threshold_c=4,
                    detections=result.detector_metadata)
                result.detector_metadata["alternative_binarization"] = {"threshold_c": 4}
                if alternate_binary is not None:
                    attempt(alternate_binary.image, metadata["angles"][0], psm=6,
                            emphasis=alternate_binary)
                    attempt(alternate_binary.image, metadata["angles"][0], psm=11,
                            emphasis=alternate_binary)
                if available():
                    luma_binary, metadata = emphasize_text(
                        image, cancelled=user_cancelled, detections=result.detector_metadata,
                        channel="luma")
                    if luma_binary is not None:
                        attempt(luma_binary.image, metadata["angles"][0], psm=6,
                                emphasis=luma_binary)
        if emphasis is not None and not result.stop_reason and available():
            alternative, metadata = emphasize_text(image, cancelled=user_cancelled, orientation=90)
            result.detector_metadata["alternative_detection"] = metadata
            if alternative is not None:
                read_emphasis(alternative, metadata)
        # El encuadre original sigue disponible como alternativa para no
        # depender exclusivamente de las propuestas de un detector.
        if not result.stop_reason:
            for angle in (0, 90, 180, 270):
                if not available():
                    break
                attempt(gray, angle)
        best = _best(snapshots)
        preferred = best["page_angle"] if best else 0
        if not result.stop_reason and available():
            # La estimación usa una miniatura sólo para medir el ángulo.
            # La pasada OCR, si se realiza, usa el original y lienzo expandido.
            skew = estimate_skew(gray)
            if skew and available():
                corrected = preferred + skew
                attempt(gray, corrected)
        for mode, psm in (("normalized", 11), ("gray", 3), ("otsu", 11),
                          ("adaptive", 11), ("inverted", 11), ("normalized", 6)):
            if result.stop_reason:
                break
            if not available():
                break
            best = _best(snapshots)
            if best:
                preferred = best["page_angle"]
                agreement = sum(_key(s) == _key(best) for s in snapshots if s["text"])
                if agreement >= 2 and best["quality"] >= .9:
                    result.stop_reason = "consistent_full_frame_readings"
                    break
            attempt(gray, preferred, mode, psm)
        if not result.stop_reason:
            result.stop_reason = ("time_budget" if time.monotonic() >= deadline else
                                  "max_calls" if result.ocr_calls >= config.max_calls else "plan_completed")
        result.budget_exhausted |= result.stop_reason in {"time_budget", "max_calls"}
        missing = sorted({0, 90, 180, 270} - set(result.completed_orientations))
        if missing and result.budget_exhausted:
            result.warnings.append("Orientaciones no completadas: " + ", ".join(map(str, missing)))
        _finish(result, snapshots, config)
    except OcrCancelled:
        result.status, result.reason = "cancelado", "Solicitud invalidada."
        result.stop_reason = "cancelled"
    finally:
        result.elapsed_s = time.monotonic() - start
    return result
