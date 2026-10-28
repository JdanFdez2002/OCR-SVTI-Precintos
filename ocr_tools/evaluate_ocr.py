"""Inventario, evaluación pareada y registros JSON locales del OCR.

No usa referencias para predecir. No sobreescribe informes ni etiquetas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ocr_tools.audit_ocr import ROOT, environment, load_baseline

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
LABELS = {"readable", "unreadable", "no_seal", "irrelevant", "invalid", "unreviewed"}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def json_native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Tipo no serializable: {type(value).__name__}")


def inventory(directory: Path, expected: int):
    paths = sorted((p for p in directory.rglob("*") if p.is_file()
                    and p.suffix.lower() in EXTENSIONS), key=lambda p: p.as_posix().lower())
    records = []
    for path in paths:
        digest = file_hash(path)
        records.append({"path": path.relative_to(directory).as_posix(), "sha256": digest,
                        "group_id": digest, "split": "unassigned", "label": "unreviewed",
                        "reference_code": None, "verified": False, "reviewer": "", "notes": ""})
    return {"schema_version": 1, "image_root": str(directory.resolve()),
            "expected_images": expected, "observed_images": len(records),
            "records": records}


def distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = new
    return row[-1]


def character_errors(reference, observed):
    """Alineamiento de Levenshtein; nunca se emplea para corregir la lectura."""
    n, m = len(reference), len(observed)
    table = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        table[i][0] = i
    for j in range(m+1):
        table[0][j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            table[i][j] = min(table[i-1][j]+1,table[i][j-1]+1,
                              table[i-1][j-1]+(reference[i-1] != observed[j-1]))
    counts = dict(deletions=0,insertions=0,substitutions=0,initial_errors=0,final_errors=0)
    i,j = n,m
    while i or j:
        if i and j and reference[i-1] == observed[j-1] and table[i][j] == table[i-1][j-1]:
            i,j = i-1,j-1
            continue
        if i and j and table[i][j] == table[i-1][j-1]+1:
            counts['substitutions'] += 1
            counts['initial_errors'] += i == 1
            counts['final_errors'] += i == n
            i,j = i-1,j-1
        elif i and table[i][j] == table[i-1][j]+1:
            counts['deletions'] += 1
            counts['initial_errors'] += i == 1
            counts['final_errors'] += i == n
            i -= 1
        else:
            counts['insertions'] += 1
            counts['initial_errors'] += i == 0
            counts['final_errors'] += i == n
            j -= 1
    return counts


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(rows):
    """Denominadores explícitos. Las referencias sin verificar no generan precisión."""
    # Usa primera repetición para calidad; todas para latencia. No infla n.
    quality = [r for r in rows if r["repeat"] == 0]
    verified = [r for r in quality if r["verified"]]
    positives = [r for r in verified if r["label"] == "readable"]
    negatives = [r for r in verified if r["label"] in {"no_seal", "irrelevant"}]
    unreadable = [r for r in verified if r["label"] == "unreadable"]
    invalid = [r for r in verified if r["label"] == "invalid"]
    accepted = [r for r in verified if r["confirmed_identifier"] is not None]
    correct = [r for r in positives if r["confirmed_identifier"] == r["reference_code"]]
    # CER estricto: se conserva guion, espacios y ceros. Un rechazo cuenta como
    # eliminación completa, no desaparece del denominador.
    edits = sum(distance(r["reference_code"], r["confirmed_identifier"] or "") for r in positives)
    chars = sum(len(r["reference_code"]) for r in positives)
    candidate_edits = sum(distance(r["reference_code"], r["suggested_text"] or "") for r in positives)
    correct_negatives = [r for r in negatives if r["confirmed_identifier"] is None
                         and r["status"] != "error_tecnico"]
    safe_unreadable = [r for r in unreadable if r["confirmed_identifier"] is None
                       and r["status"] != "error_tecnico"]
    valid_invalid = [r for r in invalid if r["status"] == "imagen_invalida"]
    false_accepts = len(accepted) - len(correct)
    measurements = {r.get('measurement_id',i):r for i,r in enumerate(rows)}
    times = [r["elapsed_s"] for r in measurements.values()]
    error_counts = Counter()
    for row in positives:
        error_counts.update(character_errors(row['reference_code'], row['suggested_text'] or ''))
    return {"images": len(quality), "verified_images": len(verified),
            "unverified_images": len(quality) - len(verified), "readable_images": len(positives),
            "negative_images": len(negatives), "unreadable_images": len(unreadable),
            "invalid_images": len(invalid), "exact_code_accuracy": _ratio(len(correct), len(positives)),
            "cer_confirmed_including_rejections": _ratio(edits, chars),
            "cer_suggested_candidate_not_acceptance": _ratio(candidate_edits, chars),
            "accepted_verified": len(accepted), "false_accepts": false_accepts,
            "false_accept_rate_all_verified": _ratio(false_accepts, len(verified)),
            "error_rate_among_accepted": _ratio(false_accepts, len(accepted)),
            "coverage_verified": _ratio(len(accepted), len(verified)),
            "rejection_rate_verified": _ratio(len(verified) - len(accepted), len(verified)),
            "accuracy_among_accepted": _ratio(len(correct), len(accepted)),
            "negative_nonacceptance": _ratio(len(correct_negatives), len(negatives)),
            "unreadable_nonacceptance": _ratio(len(safe_unreadable), len(unreadable)),
            "operational_outcome_accuracy": _ratio(len(correct) + len(correct_negatives)
                                                   + len(safe_unreadable) + len(valid_invalid), len(verified)),
            "semantic_state_accuracy": None,
            "semantic_state_accuracy_note": "Sin texto no demuestra ausencia/ilegibilidad. No se equiparan estados.",
            "technical_errors": sum(r["status"] == "error_tecnico" for r in quality),
            "invalid_inputs": sum(r["status"] == "imagen_invalida" for r in quality),
            "status_counts": dict(Counter(r["status"] for r in quality)),
            "character_errors_suggested_verified_only": dict(error_counts) if positives else None,
            "latency_s": {"mean": statistics.mean(times) if times else None,
                          "actual_measurements": len(times),
                          "median": statistics.median(times) if times else None,
                          "p90": float(np.percentile(times, 90)) if times else None,
                          "p95": float(np.percentile(times, 95)) if times else None,
                          "p99": float(np.percentile(times, 99)) if times else None},
            "mean_ocr_calls": statistics.mean(r["ocr_calls"] for r in rows) if rows else None}


def cached_measurements(path, manifest, engines, crop, mode, max_calls):
    """Importa mediciones reales completas, conserva referencias y origen aparte."""
    records = {r['path']:r for r in manifest['records']}
    cached = {}
    for line in path.read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        reference = records.get(row['path'])
        if not reference or row['sha256'] != reference['sha256']:
            raise ValueError('Medición importada no pertenece al inventario exacto')
        if row['engine'] not in engines:
            continue
        if row['repeat'] != 0 or row['crop_policy'] != crop or row['input_mode'] != mode:
            raise ValueError('Protocolo de medición importada incompatible')
        if row['engine'] == 'experimental' and row.get('diagnostic'):
            diagnostic = row['diagnostic']
            if diagnostic.get('engine_version') != 'geometry-candidates-v2' or diagnostic['config']['max_calls'] != max_calls:
                raise ValueError('Versión/presupuesto del análisis importado incompatible')
        row['measurement_origin'] = str(path)
        row['source_measurement_path'] = row['path']
        cached.setdefault((row['engine'],row['sha256'],0),row)
    return cached


def validate_manifest(manifest):
    records = manifest["records"]
    if len(records) != manifest["expected_images"]:
        raise ValueError(f"Se esperaban {manifest['expected_images']} imágenes, hay {len(records)} registros")
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("Rutas duplicadas en el manifiesto")
    groups, hashes = {}, {}
    for record in records:
        if record["label"] not in LABELS:
            raise ValueError(f"Etiqueta desconocida: {record['path']}")
        if record["split"] not in {"unassigned", "development", "evaluation"}:
            raise ValueError("Partición desconocida")
        if not isinstance(record["verified"], bool):
            raise ValueError("verified debe ser booleano")
        if record["verified"]:
            if record["label"] == "unreviewed" or not record["reviewer"].strip():
                raise ValueError("Etiqueta verificada requiere estado y revisor")
            ref = record["reference_code"]
            if record["label"] == "readable" and (not isinstance(ref, str) or not ref):
                raise ValueError("Código legible necesita referencia textual no vacía")
            if record["label"] != "readable" and ref is not None:
                raise ValueError("Estados negativos/ilegibles no llevan un código inventado")
        for table, key in ((groups, record["group_id"]), (hashes, record["sha256"])):
            if not key:
                raise ValueError("Falta grupo/hash")
            if key in table and table[key] != record["split"]:
                raise ValueError("Filtración entre particiones: mismo grupo/hash en desarrollo y evaluación")
            table[key] = record["split"]


def paired_comparison(rows, engines):
    """Pareada por foto, sin tratar repeticiones como muestras independientes."""
    result = {}
    for engine in engines:
        if engine == "baseline":
            continue
        baseline = {r["path"]: r for r in rows if r["engine"] == "baseline" and r["repeat"] == 0}
        current = {r["path"]: r for r in rows if r["engine"] == engine and r["repeat"] == 0}
        if not baseline:
            continue
        wins = losses = same = unknown = 0
        latency_deltas = []
        for path in baseline.keys() & current.keys():
            b, c = baseline[path], current[path]
            bt = [r["elapsed_s"] for r in rows if r["path"] == path and r["engine"] == "baseline"]
            ct = [r["elapsed_s"] for r in rows if r["path"] == path and r["engine"] == engine]
            latency_deltas.append(statistics.median(ct) - statistics.median(bt))
            if not b["verified"] or b["label"] != "readable":
                unknown += 1
                continue
            bc = b["confirmed_identifier"] == b["reference_code"]
            cc = c["confirmed_identifier"] == c["reference_code"]
            wins += cc and not bc
            losses += bc and not cc
            same += bc == cc
        result[engine] = {"readable_gains": wins, "readable_regressions": losses,
                          "readable_same_outcome": same, "other_or_unverified_images": unknown,
                          "median_paired_latency_delta_s": statistics.median(latency_deltas) if latency_deltas else None}
    return result


def read_image(path):
    # imdecode permite rutas Unicode en Windows sin depender de imread.
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


class Recorder:
    def __init__(self, engine):
        self.engine, self.calls = engine, []

    def ocr(self, image, cls=True):
        call = {"shape": list(image.shape)}
        self.calls.append(call)
        start = time.perf_counter()
        try:
            raw = self.engine.ocr(image, cls=cls)
            call["detections"] = raw
            return raw
        finally:
            call["elapsed_s"] = time.perf_counter() - start


def evaluate(args):
    if args.deduplicate and args.repeats != 1:
        raise ValueError('La deduplicación requiere una repetición; no simula ejecuciones repetidas')
    if args.reuse_trace and not args.deduplicate:
        raise ValueError('Importar mediciones requiere --deduplicate')
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    validate_manifest(manifest)
    directory = Path(manifest["image_root"]).resolve()
    records = manifest["records"]
    # Nunca permite que fotografías nuevas o retiradas desaparezcan del informe.
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and p.suffix.lower() in EXTENSIONS}
    if actual != {r["path"] for r in records}:
        raise ValueError("El conjunto de archivos difiere del manifiesto. Rehacer inventario sin perder etiquetas.")
    for record in records:
        path = (directory / record["path"]).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != record["sha256"]:
            raise ValueError(f"Ruta/hash no coincide: {record['path']}")
    if args.split != "all":
        records = [r for r in records if r["split"] == args.split]
    if not records:
        raise ValueError("No hay imágenes en esta partición")
    cache = cached_measurements(args.reuse_trace,manifest,args.engines,args.crop,args.input_mode,args.max_calls) if args.reuse_trace else {}
    if args.output.exists():
        raise FileExistsError("La carpeta de resultados ya existe; usa una nueva para conservar evidencia")
    args.output.mkdir(parents=True)
    from modulos import ocr_engine
    from modulos.ocr_analysis import AnalysisConfig, analyze
    from config import recorte_reticula

    baseline, source_hash = load_baseline()
    load_start = time.perf_counter()
    engine = ocr_engine.get_ocr()
    model_load = time.perf_counter() - load_start
    # Warm-up explícito para TODOS; no se incluye en la latencia por fotografía.
    warm_start = time.perf_counter()
    engine.ocr(np.zeros((64, 192, 3), dtype=np.uint8), cls=True)
    warm_s = time.perf_counter() - warm_start
    config = AnalysisConfig(max_calls=args.max_calls)
    rows = []
    try:
        import psutil
        process = psutil.Process()
        rss_start = process.memory_info().rss
        cpu_start = sum(process.cpu_times()[:2])
    except ImportError:
        process = None
    with (args.output / "trace.jsonl").open("x", encoding="utf-8") as trace:
        for index, record in enumerate(records):
            read_start = time.perf_counter()
            try:
                image = read_image(directory / record["path"])
                if image is not None and args.crop == "reticle":
                    image = recorte_reticula(image)
                read_error = None
            except Exception as exc:
                image, read_error = None, f"{type(exc).__name__}: {exc}"
            read_s = time.perf_counter() - read_start
            for repeat in range(args.repeats):
                # Alterna el orden para reducir sesgo de calentamiento/caché.
                order = list(args.engines)
                if (index + repeat) % 2:
                    order.reverse()
                for name in order:
                    row = {"path": record["path"], "sha256": record["sha256"],
                           "label": record["label"], "reference_code": record["reference_code"],
                           "verified": record["verified"], "engine": name, "repeat": repeat,
                           "image_read_s": read_s, "crop_policy": args.crop,
                           "input_mode": args.input_mode, "confirmed_identifier": None,
                           "suggested_text": None, "ocr_calls": 0, "status": "imagen_invalida",
                           "error": read_error}
                    row.update(group_id=record['group_id'],split=record['split'])
                    cache_key = (name,record['sha256'],repeat)
                    if args.deduplicate and cache_key in cache:
                        source = cache[cache_key]
                        copied = {**source,**{k:row[k] for k in ('path','sha256','label','reference_code','verified','group_id','split','image_read_s')}}
                        copied.update(reused_measurement=True, measurement_id=f'{name}:{record["sha256"]}:{repeat}',
                                      source_measurement_path=source.get('source_measurement_path',source['path']))
                        rows.append(copied)
                        trace.write(json.dumps(copied,ensure_ascii=False,default=json_native)+'\n')
                        trace.flush()
                        continue
                    start = time.perf_counter()
                    diagnostic = None
                    if image is not None:
                        try:
                            if name == "experimental":
                                result = analyze(image, input_mode=args.input_mode,
                                                 image_id=record["sha256"], config=config)
                                diagnostic = result.to_dict()
                                for key in ("status", "confirmed_identifier", "suggested_text", "ocr_calls", "error"):
                                    row[key] = diagnostic[key]
                                row['candidates'] = [c.text for c in result.candidates]
                                best = result.candidates[0] if result.candidates else None
                                row['confidence'] = best.confidence_min if best else None
                                row['variant'] = sorted({d.variant for d in best.detections}) if best else []
                                row['region'] = best.detections[0].box if best else None
                            else:
                                module = baseline if name == "baseline" else ocr_engine
                                recorder = Recorder(engine)
                                try:
                                    with patch.object(module, "get_ocr", return_value=recorder):
                                        text, score, variant, confidence = module.mejor_ocr(image)
                                finally:
                                    row["ocr_calls"] = len(recorder.calls)
                                    diagnostic = {"calls": recorder.calls}
                                row.update(confirmed_identifier=text or None, suggested_text=text or None,
                                           status="reconocido" if text else "sin_texto", score=score,
                                           variant=variant, confidence=confidence)
                        except Exception as exc:
                            row.update(status="error_tecnico", confirmed_identifier=None,
                                       error=f"{type(exc).__name__}: {exc}")
                    row["elapsed_s"] = time.perf_counter() - start
                    verified_positive = row['verified'] and row['label'] == 'readable'
                    row['exact_match'] = (row['confirmed_identifier'] == row['reference_code']) if verified_positive else None
                    row['character_errors'] = character_errors(row['reference_code'],row['suggested_text'] or '') if verified_positive else None
                    row['false_accept'] = (row['confirmed_identifier'] is not None and
                        (not verified_positive or row['confirmed_identifier'] != row['reference_code'])) if row['verified'] else None
                    row.update(reused_measurement=False,measurement_origin='current_run',source_measurement_path=row['path'])
                    if args.deduplicate:
                        row['measurement_id'] = f'{name}:{record["sha256"]}:{repeat}'
                    rows.append(row)
                    detailed = {**row,'diagnostic':diagnostic}
                    if args.deduplicate:
                        cache[cache_key] = detailed
                    trace.write(json.dumps(detailed, ensure_ascii=False, default=json_native) + "\n")
                    trace.flush()
            print(f"{index + 1}/{len(records)} {record['path']}", flush=True)
    resources = None
    if process:
        resources = {"cpu_time_s": sum(process.cpu_times()[:2]) - cpu_start,
                     "rss_before_bytes": rss_start, "rss_after_bytes": process.memory_info().rss,
                     "scope": "whole_comparison_not_per_engine"}
    summary = {"baseline_commit": "e35236e", "baseline_source_sha256": source_hash,
               "manifest_sha256": file_hash(args.manifest), "expected_dataset": manifest["expected_images"],
               "evaluated_images": len(records), "split": args.split, "crop_policy": args.crop,
               "input_mode": args.input_mode, "repeats": args.repeats, "model_load_s": model_load,
               "warmup_s": warm_s, "environment": environment(engine), "resources": resources,
               "experimental_config": asdict(config),
               "engines": {name: metrics([r for r in rows if r["engine"] == name]) for name in args.engines},
               "paired_comparison": paired_comparison(rows, args.engines),
               "interpretation": "Exploratorio salvo partición protegida y protocolo previo. Confianza no calibrada."}
    unique_rows = {}
    for row in rows:
        unique_rows.setdefault((row['engine'],row['sha256'],row['repeat']),row)
    summary['unique_images_by_sha256'] = len({r['sha256'] for r in records})
    summary['deduplicated_inference'] = args.deduplicate
    summary['fresh_file_engine_inferences_this_run'] = sum(not r['reused_measurement'] for r in rows)
    summary['reuse_trace_sha256'] = file_hash(args.reuse_trace) if args.reuse_trace else None
    summary['reuse_trace_path'] = str(args.reuse_trace) if args.reuse_trace else None
    summary['latency_scope'] = 'One real measurement per engine/hash; reused copies do not create latency samples' if args.deduplicate else 'All measured executions'
    if args.reuse_trace and resources:
        resources['scope'] = 'Continuation only; prior imported measurements not included in CPU/RSS interval'
    summary['engines_unique_hashes'] = {name:metrics([r for r in unique_rows.values() if r['engine']==name]) for name in args.engines}
    try:
        summary["worktree_diff"] = subprocess.check_output(
            ["git", "diff", "--", "modulos/ocr_engine.py"], cwd=ROOT).decode("utf-8")
        summary["implementation_sha256"] = {p: file_hash(ROOT / p) for p in
                                             ("modulos/ocr_engine.py", "modulos/ocr_analysis.py", __file__)}
    except (subprocess.SubprocessError, OSError):
        summary["version_capture_error"] = True
    write_json(args.output / "summary.json", summary)
    columns = ["path", "engine", "repeat", "label", "verified", "reference_code",
               "status", "confirmed_identifier", "suggested_text", "exact_match", "confidence", "variant",
               "region", "candidates", "character_errors", "false_accept", "elapsed_s", "ocr_calls"]
    with (args.output / "results.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["engines"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("inventory")
    create.add_argument("--images", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--expected", type=int, default=104)
    run = commands.add_parser("evaluate")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--engines", nargs="+", choices=["baseline", "current", "experimental"],
                     default=["baseline", "current", "experimental"])
    run.add_argument("--crop", choices=["none", "reticle"], default="none")
    run.add_argument("--input-mode", choices=["full_photo", "seal_roi"], default="full_photo")
    run.add_argument("--split", choices=["all", "development", "evaluation"], default="all")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--max-calls", type=int, default=21)
    run.add_argument('--deduplicate',action='store_true')
    run.add_argument('--reuse-trace',type=Path)
    args = parser.parse_args()
    if args.command == "inventory":
        if not args.images.is_dir():
            parser.error("No existe la carpeta de imágenes")
        manifest = inventory(args.images.resolve(), args.expected)
        write_json(args.output, manifest)
        print(f"Inventariadas: {manifest['observed_images']}; esperadas: {args.expected}; verificadas: 0")
        if manifest["observed_images"] != args.expected:
            raise SystemExit(2)
    else:
        if args.repeats < 1 or len(args.engines) != len(set(args.engines)):
            parser.error("Repeticiones debe ser positivo y motores no deben repetirse")
        evaluate(args)


if __name__ == "__main__":
    main()
