"""Tesseract por subprocess: TSV literal, límites de tiempo y cancelación."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2


class OcrCancelled(InterruptedError):
    pass


class OcrTimeout(TimeoutError):
    pass


@dataclass
class TsvResult:
    words: list[dict]
    malformed_rows: int = 0


def get_ocr():
    """Resuelve Windows/PATH sin depender del directorio de trabajo."""
    configured = os.environ.get("SEAI_TESSERACT_CMD", "").strip().strip('"')
    if configured:
        executable = shutil.which(configured)
        if executable:
            return executable
        raise RuntimeError(f"No se encuentra Tesseract en SEAI_TESSERACT_CMD: {configured}")
    executable = shutil.which("tesseract")
    if executable:
        return executable
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                 os.environ.get("LOCALAPPDATA")):
        if base:
            for folder in ("Tesseract-OCR", "Programs/Tesseract-OCR"):
                path = Path(base) / folder / "tesseract.exe"
                if path.is_file():
                    return str(path)
    raise RuntimeError(
        "No se encontró Tesseract. Instálalo y agrégalo a PATH o configura "
        "SEAI_TESSERACT_CMD con la ruta completa a tesseract.exe."
    )


def _run(command, payload=None, *, timeout=3.0, cancelled=None):
    deadline = time.monotonic() + timeout
    if cancelled and cancelled():
        raise OcrCancelled("Solicitud invalidada")
    environment = os.environ.copy()
    environment.setdefault("OMP_THREAD_LIMIT", "1")
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=environment,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as exc:
        raise RuntimeError(f"No se pudo iniciar Tesseract: {exc}") from exc
    try:
        pending = payload
        while True:
            if cancelled and cancelled():
                raise OcrCancelled("Solicitud invalidada")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OcrTimeout("Se agotó el tiempo de la llamada a Tesseract")
            try:
                output, errors = process.communicate(input=pending, timeout=min(.08, remaining))
                break
            except subprocess.TimeoutExpired:
                pending = None
        if process.returncode:
            message = errors.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Tesseract: {message or 'ejecución fallida'}")
        return output
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def _data_options():
    directory = os.environ.get("SEAI_TESSDATA_DIR", "").strip().strip('"')
    if not directory and not os.environ.get("TESSDATA_PREFIX"):
        bundled = Path(__file__).resolve().parents[1] / "models" / "tessdata_best"
        if os.environ.get("SEAI_TESSERACT_LANG", "eng") == "eng" and (bundled / "eng.traineddata").is_file():
            directory = str(bundled)
    return ["--tessdata-dir", directory] if directory else []


@lru_cache(maxsize=4)
def _check_languages(executable, language, directory, prefix):
    command = [executable, *_data_options(), "--list-langs"]
    output = _run(command, timeout=3.0).decode("utf-8", errors="replace")
    available = {line.strip() for line in output.splitlines()[1:] if line.strip()}
    missing = set(language.split("+")) - available
    if missing:
        raise RuntimeError(
            "Faltan idiomas de Tesseract: " + ", ".join(sorted(missing)) +
            ". Instala sus archivos .traineddata o ajusta SEAI_TESSERACT_LANG/"
            "SEAI_TESSDATA_DIR."
        )


def warmup_ocr():
    executable = get_ocr()
    _check_languages(executable, os.environ.get("SEAI_TESSERACT_LANG", "eng"),
                     os.environ.get("SEAI_TESSDATA_DIR", ""),
                     os.environ.get("TESSDATA_PREFIX", ""))
    if os.environ.get("SEAI_TEXT_EMPHASIS", "1").lower() not in {"0", "false", "no"}:
        from modulos.text_emphasis import warmup_detector
        warmup_detector()
    return executable


def parse_tsv(payload) -> TsvResult:
    """Tesseract TSV NO es CSV con comillas.

    Una palabra que sea una comilla literal nunca puede consumir las filas
    siguientes. Sólo se aceptan filas de palabra (nivel 5), con geometría y
    confianza válidas; las filas estructurales no son texto reconocido.
    """
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8-sig", errors="replace")
    rows = payload.splitlines()
    expected = ["level", "page_num", "block_num", "par_num", "line_num", "word_num",
                "left", "top", "width", "height", "conf", "text"]
    if not rows or rows[0].lstrip("\ufeff").split("\t") != expected:
        raise RuntimeError("Tesseract no devolvió un encabezado TSV válido")
    words, malformed = [], 0
    for row in rows[1:]:
        fields = row.split("\t", 11)
        if len(fields) != 12:
            malformed += bool(row)
            continue
        if fields[0] != "5":
            continue
        try:
            ids = tuple(int(value) for value in fields[1:6])
            x, y, width, height = (int(value) for value in fields[6:10])
            confidence = float(fields[10]) / 100.0
            if (not math.isfinite(confidence) or not 0 <= confidence <= 1 or
                    min(x, y) < 0 or min(width, height) <= 0 or min(ids) < 0):
                raise ValueError("Datos de palabra inválidos")
        except (ValueError, OverflowError):
            malformed += 1
            continue
        text = fields[11].strip()
        if text:
            words.append({"text": text, "confidence": confidence,
                          "box": [x, y, width, height], "line": list(ids[:4])})
    return TsvResult(words, malformed)


def read_words(image, *, executable=None, language=None, psm=11, timeout=3.0,
               cancelled=None) -> TsvResult:
    executable = executable or get_ocr()
    language = language or os.environ.get("SEAI_TESSERACT_LANG", "eng")
    if timeout <= 0:
        raise OcrTimeout("Presupuesto OCR agotado")
    if cancelled and cancelled():
        raise OcrCancelled("Solicitud invalidada")
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError("No se pudo preparar la imagen para OCR")
    # Sin diccionarios de códigos, whitelist ni autocorrección de identificadores.
    command = [executable, "stdin", "stdout", *_data_options(), "-l", language,
               "--oem", "1", "--psm", str(psm), "--dpi", "300",
               "-c", "load_system_dawg=0", "-c", "load_freq_dawg=0",
               "-c", "tessedit_create_tsv=1", "-c", "tessedit_create_txt=0"]
    output = _run(command, encoded.tobytes(), timeout=timeout, cancelled=cancelled)
    return parse_tsv(output)
