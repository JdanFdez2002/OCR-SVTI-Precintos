"""Reproducciones controladas. NO mide precisión sobre fotografías reales.

Ejecutar desde la raíz: .\venv\Scripts\python.exe -m ocr_tools.audit_ocr
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

BASELINE_COMMIT = "e35236e"
ROOT = Path(__file__).resolve().parents[1]


def load_baseline():
    """Carga la versión fijada sin tocar Git, índices ni archivos de trabajo."""
    import types

    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:modulos/ocr_engine.py"], cwd=ROOT
    )
    module = types.ModuleType("ocr_baseline")
    exec(compile(source, "git:baseline/ocr_engine.py", "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def reproduce(module):
    image = np.zeros((100, 300, 3), dtype=np.uint8)
    cases = []
    for name, response in (
        ("empty", ("", 0.0)),
        ("short", ("ABC", 0.99)),
        ("long_noise", ("BRAND ABC123 EXTRA", 0.99)),
        ("hyphens_only", ("------", 0.99)),
    ):
        with patch.object(module, "_ocr_imagen", return_value=response) as mocked:
            result = module.mejor_ocr(image)
        cases.append({"case": name, "ocr_calls": mocked.call_count, "result": result})
    # Segunda lectura apta detiene el flujo, pero devuelve el candidato previo
    # si tiene mayor score, aunque este NO cumpliera el umbral rápido.
    responses = [("BRAND ABC123 EXTRA", 0.87), ("ABC123", 0.99)]
    with patch.object(module, "_ocr_imagen", side_effect=responses) as mocked:
        cases.append({"case": "early_exit_selects_previous", "result": module.mejor_ocr(image),
                      "ocr_calls": mocked.call_count})
    return cases


def preprocessing_measurements(baseline, current, samples=12):
    image = np.random.default_rng(431).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    before = baseline.preprocesar(image)
    after = current.preprocesar(image)
    equivalent = len(before) == len(after) and all(
        n1 == n2 and np.array_equal(a, b)
        for (n1, a), (n2, b) in zip(before, after)
    )
    durations = {"baseline": [], "current": []}
    for _ in range(2):
        for module in (baseline, current):
            with patch.object(module, "_ocr_imagen", return_value=("ABC123", 0.7)) as mocked:
                module.mejor_ocr(image)
                # No salida temprana: asegura que el caso completo se conserva.
                assert mocked.call_count == 21
    for index in range(samples):
        order = [("baseline", baseline), ("current", current)]
        if index % 2:
            order.reverse()
        for name, module in order:
            # Sólo el coste necesario para alcanzar la primera variante gris;
            # un OCR simulado no constituye una medida de inferencia real.
            with patch.object(module, "_ocr_imagen", side_effect=[("ABC123", 0.7), ("ABC123", 0.99)]):
                start = time.perf_counter()
                module.mejor_ocr(image)
                durations[name].append((time.perf_counter() - start) * 1000)
    return {"kind": "synthetic_preprocessing_only_mocked_ocr", "samples": samples,
            "shape": list(image.shape), "variants_pixel_identical": equivalent,
            "ms": {name: {"mean": statistics.mean(values), "median": statistics.median(values),
                          "samples": values} for name, values in durations.items()}}


def environment(engine=None):
    packages = {}
    for name in ("paddlepaddle", "paddleocr", "numpy", "opencv-python", "PySide6", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
    if engine is not None:
        import paddle
        keys = ("use_gpu", "use_angle_cls", "rec_algorithm", "ocr_version", "det_limit_side_len",
                "drop_score", "cpu_threads", "enable_mkldnn")
        result.update(cuda_compiled=paddle.is_compiled_with_cuda(), device=paddle.device.get_device(),
                      paddle_config={key: getattr(engine.args, key, None) for key in keys})
        model_hashes = {}
        for key in ("det_model_dir", "rec_model_dir", "cls_model_dir"):
            directory = Path(getattr(engine.args, key))
            model_hashes[key] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(directory.iterdir()) if p.is_file()}
        result["model_sha256"] = model_hashes
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()
    from modulos import ocr_engine

    baseline, digest = load_baseline()
    start = time.perf_counter()
    engine = ocr_engine.get_ocr() if args.load_model else None
    result = {"baseline_commit": BASELINE_COMMIT, "baseline_source_sha256": digest,
              "environment": environment(engine), "model_load_s": time.perf_counter() - start,
              "controlled_reproductions_not_photographic_accuracy": {
                  "baseline": reproduce(baseline), "current": reproduce(ocr_engine)},
              "preprocessing": preprocessing_measurements(baseline, ocr_engine)}
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
