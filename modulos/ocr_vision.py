"""Variantes sobre el encuadre completo, sin ROI, recortes ni reducción."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class ImageView:
    name: str
    image: np.ndarray
    to_source: np.ndarray


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()


def page_view(gray, angle=0):
    """Conserva todos los píxeles: giros ortogonales o lienzo expandido.

    to_source permite registrar las cajas de Tesseract en la foto original;
    esas cajas son metadatos y nunca se usan para recortar el OCR.
    """
    h, w = gray.shape
    if angle % 360 == 90:
        rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        inverse = np.array([[0, -1, w - 1], [1, 0, 0], [0, 0, 1]], dtype=float)
    elif angle % 360 == 180:
        rotated = cv2.rotate(gray, cv2.ROTATE_180)
        inverse = np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], dtype=float)
    elif angle % 360 == 270:
        rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        inverse = np.array([[0, 1, 0], [-1, 0, h - 1], [0, 0, 1]], dtype=float)
    elif angle % 360:
        # Se amplía el lienzo: ni las esquinas ni texto periférico quedan fuera.
        center = ((w - 1) / 2, (h - 1) / 2)
        affine = cv2.getRotationMatrix2D(center, angle, 1.0)
        cosine, sine = abs(affine[0, 0]), abs(affine[0, 1])
        width = math.ceil(w * cosine + h * sine)
        height = math.ceil(h * cosine + w * sine)
        affine[0, 2] += (width - 1) / 2 - center[0]
        affine[1, 2] += (height - 1) / 2 - center[1]
        rotated = cv2.warpAffine(gray, affine, (width, height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        inverse = np.linalg.inv(np.vstack((affine, [0.0, 0.0, 1.0])))
    else:
        rotated, inverse = gray, np.eye(3)
    return ImageView(f"foto completa {angle}°", rotated, inverse)


def estimate_skew(gray):
    """Sólo propone un giro adicional; no reemplaza la foto sin transformar."""
    h, w = gray.shape
    scale = min(1.0, 1200 / max(h, w))
    # Esta miniatura se usa sólo para medir inclinación, nunca para reconocer.
    preview = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA) if scale < 1 else gray
    edges = cv2.Canny(preview, 60, 160)
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                               minLineLength=max(25, min(preview.shape) // 20),
                               maxLineGap=10)
    if segments is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        angle = (math.degrees(math.atan2(float(y2 - y1), float(x2 - x1))) + 45) % 90 - 45
        if .8 <= abs(angle) <= 20:
            angles.append(angle)
            weights.append(math.hypot(float(x2 - x1), float(y2 - y1)))
    if len(angles) < 4:
        return 0.0
    angles, weights = np.asarray(angles), np.asarray(weights)
    histogram, limits = np.histogram(angles, bins=40, range=(-20, 20), weights=weights)
    index = int(histogram.argmax())
    center = (limits[index] + limits[index + 1]) / 2
    aligned = np.abs(angles - center) <= 2
    if aligned.sum() < 4 or weights[aligned].sum() < weights.sum() * .55:
        return 0.0
    return round(float(np.median(angles[aligned])), 2)


def prepare_view(view, mode="gray"):
    """Contraste/binarización global sin extraer partes de la fotografía."""
    image = view.image
    mapping = view.to_source.copy()
    if mode in {"normalized", "otsu", "adaptive"}:
        image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    if mode == "otsu":
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif mode == "adaptive":
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 9)
    elif mode == "inverted":
        image = cv2.bitwise_not(image)
    elif mode not in {"gray", "normalized"}:
        raise ValueError(f"Preparación OCR desconocida: {mode}")
    # Únicamente se añade un borde externo. No se elimina ningún píxel.
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    mapping = mapping @ np.array([[1, 0, -10], [0, 1, -10], [0, 0, 1]], float)
    return ImageView(f"{view.name}/{mode}", image, mapping)


def source_box(view, box, source_shape):
    x, y, w, h = box
    points = np.float32([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]])
    mapped = cv2.perspectiveTransform(points, view.to_source)[0]
    mapped[:, 0] = np.clip(mapped[:, 0], 0, source_shape[1] - 1)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, source_shape[0] - 1)
    return mapped.astype(float).tolist()
