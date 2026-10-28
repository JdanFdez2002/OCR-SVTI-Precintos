"""API experimental trazable. No sustituye silenciosamente a ``mejor_ocr``.

Consistencia entre variantes del mismo modelo NO es verificación independiente.
Sin un ROI declarado de precinto, texto consistente no confirma su identidad.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import re
import time
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class AnalysisConfig:
    # Umbrales de investigación, NO calibrados con el banco fotográfico.
    confidence: float = 0.88
    confirmations: int = 2
    margin: float = 0.05
    typical_length: int = 15
    excess_length_penalty: float = 0.01
    max_calls: int = 21

    def __post_init__(self):
        if not 0 <= self.confidence <= 1 or not 0 <= self.margin <= 1:
            raise ValueError("Confianza y margen deben estar entre 0 y 1")
        if self.confirmations < 2 or not 1 <= self.max_calls <= 21:
            raise ValueError("Se requieren al menos dos observaciones y entre 1 y 21 llamadas")
        if self.typical_length < 1 or self.excess_length_penalty < 0:
            raise ValueError("Longitud orientativa y penalización inválidas")


@dataclass
class Detection:
    raw_text: str
    normalized_text: str
    confidence: float
    box: list[list[float]]  # Coordenadas de la entrada ORIGINAL, incluso al rotar.
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
    status: str = "sin_texto"
    confirmed_identifier: str | None = None
    suggested_text: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    ocr_calls: int = 0
    stages: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    reason: str = ""
    error: str | None = None
    config: dict = field(default_factory=dict)
    schema_version: int = 1
    engine_version: str = "experimental-candidates-v1"
    evidence_kind: str = "same_model_correlated_observations"
    detections: list[Detection] = field(default_factory=list)
    source_shape: list[int] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def normalized(text: str) -> str:
    """No convierte O/0 ni I/1, ni elimina espacios o guiones interiores."""
    return text.strip().translate(str.maketrans('abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'))


def _line_geometry(detection):
    points = np.asarray(detection.box, dtype=float)
    direction = points[1] - points[0]
    length = np.linalg.norm(direction)
    if length < 1:
        return None
    direction /= length
    normal = np.array([-direction[1], direction[0]])
    along, across = points @ direction, points @ normal
    return direction, normal, along.min(), along.max(), across.min(), across.max()


def join_fragments(detections):
    """Hipótesis geométricas, siempre con las cajas originales conservadas.

    Sólo observaciones de la misma imagen/variante, alineación, escala y distancia
    compatibles. Las palabras alfabéticas de más de dos caracteres no se unen:
    podrían ser una marca. Sin evidencia semántica pasan a revisión.
    """
    joined = []
    observations = {}
    for d in detections:
        observations.setdefault((d.observation, d.rotation), []).append(d)
    for group in observations.values():
        edges = {}
        for i, left in enumerate(group):
            ga = _line_geometry(left)
            if ga is None:
                continue
            u, v, a0, a1, b0, b1 = ga
            height = b1 - b0
            if height < 2:
                continue
            for j, right in enumerate(group):
                if i == j:
                    continue
                gb = _line_geometry(right)
                if gb is None or np.dot(u, gb[0]) < np.cos(np.deg2rad(8)):
                    continue
                points = np.asarray(right.box)
                x, y = points @ u, points @ v
                rh = y.max() - y.min()
                gap = x.min() - a1
                if not (.65 <= rh / height <= 1.55 and 0 <= gap <= 1.5 * height
                        and abs((b0 + b1) / 2 - (y.min() + y.max()) / 2) <= .22 * height):
                    continue
                if min(len(left.normalized_text), len(right.normalized_text)) > 2 and gap > .5 * height:
                    continue
                # No saltar una caja intermedia ni fusionar palabras de marca.
                if any(len(re.sub('[^A-Z]', '', d.normalized_text)) > 2
                       and not any(c.isdigit() for c in d.normalized_text) for d in (left, right)):
                    continue
                edges.setdefault(i, []).append((gap, j))
        successors = {i: min(values)[1] for i, values in edges.items()}
        predecessors = {}
        for i, j in successors.items():
            predecessors.setdefault(j, []).append(i)
        for start in successors:
            if start in predecessors:
                continue
            chain, current = [start], start
            while current in successors and len(chain) < len(group):
                following = successors[current]
                if following in chain or len(predecessors[following]) != 1:
                    break
                chain.append(following)
                current = following
            if len(chain) < 2:
                continue
            fragments = [group[k] for k in chain]
            u, v, *_ = _line_geometry(fragments[0])
            points = np.concatenate([np.asarray(d.box) for d in fragments])
            x, y = points @ u, points @ v
            corners = np.array([[x.min(), y.min()], [x.max(), y.min()],
                                [x.max(), y.max()], [x.min(), y.max()]])
            box = (corners[:, :1] * u + corners[:, 1:] * v).tolist()
            raw = ' '.join(d.raw_text for d in fragments)
            joined.append(Detection(raw, normalized(raw), min(d.confidence for d in fragments),
                                    box, fragments[0].variant + ':fragmentos',
                                    fragments[0].rotation, fragments[0].observation,
                                    [d.normalized_text for d in fragments], [d.box for d in fragments]))
    return joined


def _original_box(points, rotation, shape):
    box = np.asarray(points, dtype=float)
    if box.shape != (4, 2) or not np.isfinite(box).all():
        raise ValueError("Caja OCR inválida")
    height, width = shape[:2]
    if rotation == -90:  # inversa de ROTATE_90_COUNTERCLOCKWISE
        box = np.column_stack((width - 1 - box[:, 1], box[:, 0]))
    elif rotation == 90:
        box = np.column_stack((box[:, 1], height - 1 - box[:, 0]))
    return box.tolist()


def _overlap(a, b):
    # IoU de envolventes para asociar observaciones; no localiza precintos.
    a, b = np.asarray(a), np.asarray(b)
    amin, amax = a.min(axis=0), a.max(axis=0)
    bmin, bmax = b.min(axis=0), b.max(axis=0)
    intersection = float(np.maximum(0, np.minimum(amax, bmax) - np.maximum(amin, bmin)).prod())
    union = float(np.maximum(0, amax - amin).prod() + np.maximum(0, bmax - bmin).prod()) - intersection
    return intersection / union if union else 0.0


def candidates_from(detections: list[Detection], config: AnalysisConfig) -> list[Candidate]:
    groups: list[list[Detection]] = []
    for detection in detections:
        for group in groups:
            if (group[0].normalized_text == detection.normalized_text
                    and _overlap(group[0].box, detection.box) >= 0.3):
                group.append(detection)
                break
        else:
            groups.append([detection])
    result = []
    for group in groups:
        text = group[0].normalized_text
        compatible = bool(re.fullmatch(r"[A-Z0-9 -]+", text) and re.search(r"[A-Z0-9]", text))
        confidences = [d.confidence for d in group]
        mean = sum(confidences) / len(confidences)
        warnings = []
        if not compatible:
            warnings.append("caracteres_o_espacios_no_compatibles")
        if not any(ch.isdigit() for ch in text):
            warnings.append('texto_alfabetico_identidad_no_verificada')
        if len(text.split()) > 1 and any(token.isalpha() and len(token) > 2 for token in text.split()):
            warnings.append('posible_texto_adicional')
        excess = max(0, len(text) - config.typical_length)
        if excess:
            warnings.append("longitud_inusual_no_descartada")
        # No recompensa longitud. La penalización es suave y configurable.
        score = mean - excess * config.excess_length_penalty - (0.25 if not compatible else 0)
        result.append(Candidate(text, score, mean, min(confidences),
                                len({d.observation for d in group}), compatible, warnings, group))
    return sorted(result, key=lambda c: (not any(ch.isdigit() for ch in c.text), -c.score, -c.confidence_min, c.text))


def _decide(candidates, config, input_mode):
    if not candidates:
        return "sin_texto", None, "No se detectó texto; esto NO demuestra ausencia de precinto."
    best = candidates[0]
    if not best.syntax_compatible:
        return "texto_no_identificado", None, "Hay texto, pero no un candidato alfanumérico aislado."
    if 'texto_alfabetico_identidad_no_verificada' in best.warnings:
        return 'texto_no_identificado', None, 'Texto alfabético: no se distingue marca de identificador sin evidencia adicional.'
    if 'posible_texto_adicional' in best.warnings:
        return 'ambiguo', None, 'La caja puede incluir texto adicional al identificador.'
    if any('cerca_del_borde' in c.warnings for c in candidates):
        return 'posiblemente_incompleto', None, 'Texto cerca del límite del recorte; revisar extremos.'
    grouped = [c for c in candidates if any(':fragmentos' in d.variant for d in c.detections)]
    if grouped:
        # Una concatenación geométrica sola no prueba continuidad del identificador.
        # Aceptar sólo si otra variante lee el identificador entero en una caja.
        supported = [c for c in grouped if any(':fragmentos' not in d.variant for d in c.detections)]
        if not supported:
            return 'posiblemente_incompleto', None, 'Fragmentos alineados; falta lectura visual de la línea completa.'
        best = max(supported, key=lambda c: c.score)
    elif any(len(c.text.replace(' ', '')) <= 2 for c in candidates):
        return 'posiblemente_incompleto', None, 'Fragmento corto sin continuidad visual verificada.'
    if best.confidence_min < config.confidence or best.observations < config.confirmations:
        return "ambiguo", None, "Evidencia insuficiente o confianza baja."
    # Conserva diferencias de un carácter y detecciones en otras regiones.
    # No se equiparan guiones ausentes ni se vota para corregir caracteres.
    rivals = [c for c in candidates if c is not best and c.syntax_compatible
              and (c.confidence_mean >= config.confidence or best.score - c.score < config.margin)]
    # Las piezas contenidas geométricamente en una línea leída completa son
    # evidencia de esa línea. No eliminar rivales por mera subcadena textual.
    rivals = [c for c in rivals if not (grouped and all(
        any(bd.observation == cd.observation and any(
            text == cd.normalized_text and np.allclose(box, cd.box)
            for text, box in zip(bd.fragments, bd.fragment_boxes)) for bd in best.detections)
        for cd in c.detections))]
    if rivals:
        return "ambiguo", None, "Existen candidatos rivales; no se corrigen ni se concatenan."
    if input_mode != "seal_roi":
        return "texto_no_identificado", None, "Texto consistente sin localización semántica del precinto."
    return "reconocido", best.text, "Consistencia en ROI declarado; criterio experimental no calibrado."


def _contains_box(outer, inner):
    outer, inner = np.asarray(outer), np.asarray(inner)
    return bool((inner.min(0) >= outer.min(0) - 2).all() and
                (inner.max(0) <= outer.max(0) + 2).all())


def _stages(image):
    from modulos.ocr_engine import _iter_preprocesar

    yield "Directo-BGR", 0, image
    for name, processed in _iter_preprocesar(image):
        yield f"Proc:{name}", 0, processed
    for rotation, code in ((-90, cv2.ROTATE_90_COUNTERCLOCKWISE), (90, cv2.ROTATE_90_CLOCKWISE)):
        rotated = cv2.rotate(image, code)
        yield "Rot-BGR", rotation, rotated
        for name, processed in _iter_preprocesar(rotated):
            yield f"Rot:{name}", rotation, processed


def analyze(image, *, input_mode="full_photo", image_id="", config=None,
            infer: Callable | None = None, cancelled=None) -> AnalysisResult:
    """Analiza localmente sin escribir archivos ni cambiar la API Qt existente.

    ``full_photo`` y ``seal_roi`` describen la ENTRADA, no recortan la imagen.
    Un ROI de cámara sólo localiza por contrato del operador, no por IA.
    Sin detector especializado no distinguimos ausencia, ilegibilidad e irrelevancia.
    """
    config = config or AnalysisConfig()
    if input_mode not in {"full_photo", "seal_roi"}:
        raise ValueError("Modo de entrada desconocido")
    result = AnalysisResult(image_id, input_mode, config=asdict(config))
    result.schema_version = 2
    result.engine_version = 'geometry-candidates-v2'
    start = time.perf_counter()
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0):
        result.status, result.reason = "imagen_invalida", "Se requiere ndarray BGR uint8 no vacío."
        result.elapsed_s = time.perf_counter() - start
        return result
    if infer is None:
        from modulos.ocr_engine import ejecutar_ocr
        infer = lambda array: ejecutar_ocr(array, conservar_fragmentos=True)
    result.source_shape = list(image.shape)
    detections = []
    seen = set()
    stages = iter(_stages(image))
    try:
        while result.ocr_calls < config.max_calls:
            if cancelled is not None and cancelled():
                result.status, result.reason = 'cancelado', 'Solicitud invalidada por una acción posterior'
                result.elapsed_s = time.perf_counter() - start
                return result
            prep_start = time.perf_counter()
            try:
                variant, rotation, processed = next(stages)
            except StopIteration:
                break
            prep_s = time.perf_counter() - prep_start
            digest = hashlib.sha256(memoryview(np.ascontiguousarray(processed))).hexdigest()
            fingerprint = (processed.shape, digest)
            # La misma imagen exacta no constituye nueva evidencia. Evita gastar
            # inferencias, pero no elimina una familia de transformaciones.
            if fingerprint in seen:
                result.stages.append({"variant": variant, "rotation": rotation,
                                      "preprocessing_s": prep_s, "skipped": "identical_pixels"})
                continue
            seen.add(fingerprint)
            stage = {"variant": variant, "rotation": rotation, "preprocessing_s": prep_s}
            result.stages.append(stage)
            call_start = time.perf_counter()
            result.ocr_calls += 1
            try:
                raw = infer(processed)
            finally:
                stage["ocr_s"] = time.perf_counter() - call_start
            count = 0
            for line in (raw[0] or []) if raw else []:
                box, (text, confidence) = line
                confidence = float(confidence)
                if not isinstance(text, str) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("Detección OCR mal formada")
                if not text.strip():
                    continue
                detections.append(Detection(text, normalized(text), confidence,
                                            _original_box(box, rotation, image.shape),
                                            variant, rotation, digest))
                count += 1
            stage["detections"] = count
            result.detections = detections.copy()
            result.candidates = candidates_from(detections + join_fragments(detections), config)
            height, width = image.shape[:2]
            for candidate in result.candidates:
                for detection in candidate.detections:
                    box = np.asarray(detection.box)
                    if (box[:, 0].min() <= 2 or box[:, 1].min() <= 2 or
                            box[:, 0].max() >= width - 3 or box[:, 1].max() >= height - 3):
                        candidate.warnings.append('cerca_del_borde')
                        break
            status, identifier, reason = _decide(result.candidates, config, input_mode)
            # En foto completa, conserva alternativas: no sale sólo por tener texto.
            # No salir tras dos variantes: una transformación posterior puede
            # revelar un prefijo/sufijo que el primer reconocimiento omitió.
        result.status, result.confirmed_identifier, result.reason = _decide(result.candidates, config, input_mode)
    except Exception as exc:
        result.status = "error_tecnico"
        result.confirmed_identifier = None
        result.error = f"{type(exc).__name__}: {exc}"
        result.reason = "Procesamiento incompleto; el error se conserva y no se convierte en ausencia."
    result.suggested_text = result.candidates[0].text if result.candidates else None
    result.elapsed_s = time.perf_counter() - start
    return result
