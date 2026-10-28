"""Integración real PaddleOCR sobre DOS imágenes sintéticas, no el banco SVTI.

No constituye una comparación de precisión fotográfica ni de generalización.
"""

import argparse
import json
import statistics
import time
from unittest.mock import patch

import cv2
import numpy as np

from modulos import ocr_engine
from modulos.ocr_analysis import AnalysisConfig, analyze
from ocr_tools.audit_ocr import environment, load_baseline
from ocr_tools.evaluate_ocr import Recorder, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", default=3, type=int)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Repeticiones debe ser positivo")
    baseline, digest = load_baseline()
    engine = ocr_engine.get_ocr()
    blank = np.full((100, 320, 3), 255, np.uint8)
    text = blank.copy()
    cv2.putText(text, "AB7-X90", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
    engine.ocr(text, cls=True)
    rows = []
    for name, image in (("synthetic_blank", blank), ("synthetic_text", text)):
        for repeat in range(args.repeats):
            order = [("baseline", baseline), ("current", ocr_engine)]
            if repeat % 2:
                order.reverse()
            for version, module in order:
                recorder = Recorder(engine)
                start = time.perf_counter()
                with patch.object(module, "get_ocr", return_value=recorder):
                    result = module.mejor_ocr(image)
                rows.append({"sample": name, "repeat": repeat, "engine": version,
                             "result": result, "ocr_calls": len(recorder.calls),
                             "elapsed_s": time.perf_counter() - start})
        result = analyze(image, input_mode="seal_roi", image_id=name, config=AnalysisConfig())
        rows.append({"sample": name, "repeat": 0, "engine": "experimental",
                     "result": result.to_dict(), "ocr_calls": result.ocr_calls, "elapsed_s": result.elapsed_s})
    equivalent = all(
        next(r for r in rows if r["sample"] == sample and r["repeat"] == repeat and r["engine"] == "baseline")["result"]
        == next(r for r in rows if r["sample"] == sample and r["repeat"] == repeat and r["engine"] == "current")["result"]
        for sample in ("synthetic_blank", "synthetic_text") for repeat in range(args.repeats)
    )
    summary = {}
    for sample in ("synthetic_blank", "synthetic_text"):
        summary[sample] = {}
        for version in ("baseline", "current", "experimental"):
            subset = [r for r in rows if r["sample"] == sample and r["engine"] == version]
            summary[sample][version] = {
                "repeats": len(subset), "mean_s": statistics.mean(r["elapsed_s"] for r in subset),
                "median_s": statistics.median(r["elapsed_s"] for r in subset),
                "ocr_calls": [r["ocr_calls"] for r in subset]}
    output = {"kind": "real_model_synthetic_inputs_only", "baseline_source_sha256": digest,
              "environment": environment(engine), "legacy_outputs_equal": equivalent,
              "summary": summary, "rows": rows}
    write_json(args.output, output)
    print(json.dumps({"legacy_outputs_equal": equivalent, "summary": summary}, indent=2))
    if not equivalent:
        raise SystemExit("Regresión en la salida legacy sobre la prueba de integración")


if __name__ == "__main__":
    main()
