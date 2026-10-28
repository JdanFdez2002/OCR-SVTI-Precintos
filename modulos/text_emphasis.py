"""Atenúa el fondo sobre un lienzo completo; no reconoce caracteres ni recorta.

DBNet de OpenCV propone texto en cualquier posición de la foto. Se conservan
todas sus propuestas en sus coordenadas; Tesseract recibe un solo encuadre.
"""
from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from modulos.ocr_vision import ImageView

MODEL_NAME = "DB_IC15_resnet18.onnx"
_lock = threading.Lock()


def detector_path():
    configured = os.environ.get("SEAI_TEXT_DETECTOR_MODEL", "").strip().strip('"')
    return Path(configured) if configured else Path(__file__).resolve().parents[1] / "models" / MODEL_NAME


@lru_cache(maxsize=4)
def _model(path, size=(1280, 736)):
    detector = cv2.dnn_TextDetectionModel_DB(path)
    detector.setBinaryThreshold(.3).setPolygonThreshold(.4).setUnclipRatio(2).setMaxCandidates(1000)
    detector.setInputParams(1 / 255.0, size,
                            (122.67891434, 116.66876762, 104.00698793))
    return detector


def warmup_detector():
    path = detector_path()
    if path.is_file():
        with _lock:
            _model(str(path))
    return path.is_file()


def emphasize_text(image, cancelled=None, orientation=0, threshold_c=9, detections=None, channel="max"):
    start = time.monotonic()
    path = detector_path()
    if not path.is_file():
        return None, {"available": False, "reason": "Falta el modelo de detección; ejecutar ocr_tools/setup_tesseract_models.py"}
    if cancelled and cancelled():
        return None, {"cancelled": True}
    # El detector analiza la fotografía entera, no una ROI central.
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    proposals = []
    detection_stages = []
    if detections is not None:
        proposals = [(np.float32(record["box"]), record["detector_confidence"])
                     for record in detections.get("boxes", [])]
    for turn in (() if detections is not None else (orientation,)):
        if cancelled and cancelled():
            return None, {"cancelled": True}
        oriented = bgr if turn == 0 else cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        size = (1280, 736) if oriented.shape[1] >= oriented.shape[0] else (736, 1280)
        begin = time.monotonic()
        with _lock:
            detector = _model(str(path), size)
            detected, scores = detector.detect(oriented)
        for polygon, score in zip(detected, scores):
            polygon = polygon.astype(np.float32)
            if turn:
                polygon = np.column_stack((polygon[:, 1], image.shape[0] - 1 - polygon[:, 0]))
            proposals.append((polygon, score))
        detection_stages.append({"angle": turn, "elapsed_s": time.monotonic() - begin,
                                 "count": len(detected)})
    if cancelled and cancelled():
        return None, {"cancelled": True}
    mask = np.zeros(image.shape[:2], np.uint8)
    directions, records = [], []
    for box, confidence in proposals:
        polygon = box.astype(np.float32)
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            continue
        (cx, cy), (width, height), rotation = cv2.minAreaRect(polygon)
        if min(width, height) < 1:
            continue
        # DB ya amplía las bandas transversalmente. Añadir más margen en ese
        # eje volvería a introducir bordes y códigos de barras vecinos.
        along = max(5.0, min(width, height) * .55)
        # En glifos cortos el margen transversal evita cortar trazos finos.
        # En líneas largas se mantiene en cero para no reintroducir barras.
        across = max(2.0, min(width, height) * .2) if max(width, height) / min(width, height) < 2.5 else 0
        padx, pady = (along, across) if width >= height else (across, along)
        expanded = cv2.boxPoints(((cx, cy), (width + 2 * padx, height + 2 * pady), rotation))
        cv2.fillConvexPoly(mask, np.int32(expanded), 255)
        edges = np.roll(polygon, -1, axis=0) - polygon
        lengths = np.linalg.norm(edges, axis=1)
        edge = edges[int(lengths.argmax())]
        angle = (math.degrees(math.atan2(float(edge[1]), float(edge[0]))) + 90) % 180 - 90
        directions.append((float(lengths.max()), angle))
        records.append({"box": polygon.astype(float).tolist(),
                        "detector_confidence": float(confidence), "angle": angle})
    if not records:
        return None, {"available": True, "boxes": [], "elapsed_s": time.monotonic() - start}
    # Máximo de canales: conserva el contraste de tinta oscura sobre papel
    # coloreado sin imponer un color, idioma, código o posición determinados.
    gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if channel == "luma" else np.max(image, axis=2)) if image.ndim == 3 else image
    old_h, old_w = gray.shape
    scale = 3.0 if max(gray.shape) < 1000 else 1.0
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        gray = cv2.addWeighted(gray, 1.8, cv2.GaussianBlur(gray, (0, 0), 1), -.8, 0)
    # Binarizar ANTES de colocar el fondo blanco evita que Tesseract convierta
    # el pequeño papel gris en un único bloque negro por su histograma global.
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, threshold_c)
    canvas = np.full_like(binary, 255)
    canvas[mask > 0] = binary[mask > 0]
    mapping = np.diag([old_w / canvas.shape[1], old_h / canvas.shape[0], 1.0])
    ordered = []
    for _, angle in sorted(directions, reverse=True):
        if all(abs((angle - previous + 90) % 180 - 90) > 12 for previous in ordered):
            ordered.append(angle)
    metadata = {"available": True, "model": MODEL_NAME, "boxes": records,
                "angles": ordered, "scale": scale, "cropped": False,
                "elapsed_s": time.monotonic() - start, "stages": detection_stages,
                "threshold_c": threshold_c, "detector_orientation": orientation, "channel": channel}
    return ImageView(f"foto completa/fondo atenuado/detector {orientation}°/C{threshold_c}/{channel}", canvas, mapping), metadata
