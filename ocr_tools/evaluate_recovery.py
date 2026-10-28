"""Comparación visual sin pasar referencias a ningún predictor.

A/C se miden de nuevo en orden alternado; B usa la traza lenta real conservada
por SHA-256 y se identifica como histórica. Fast es un candidato acotado adicional.
No asigna etiquetas a partir de nombres de archivo.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import cv2
import numpy as np
import psutil

from modulos import ocr_engine
from modulos.ocr_reader import ReaderConfig, recognize_fast, recognize_original
from ocr_tools.audit_ocr import ROOT, environment, load_baseline
from ocr_tools.evaluate_ocr import Recorder, character_errors, file_hash, json_native, read_image, validate_manifest, write_json


def summary(rows):
    times = [r['elapsed_s'] for r in rows]
    cpu = [r['cpu_s'] for r in rows if r.get('cpu_s') is not None]
    rss = [r['rss_bytes'] for r in rows if r.get('rss_bytes')]
    return dict(unique_images=len(rows),nonempty=sum(bool(r['text']) for r in rows),
                technical_errors=sum(bool(r.get('error')) for r in rows),
                median_s=statistics.median(times),p95_s=float(np.percentile(times,95)),
                mean_calls=statistics.mean(r['ocr_calls'] for r in rows),
                cpu_s=sum(cpu) if cpu else None,
                max_rss_sampled_bytes=max(rss) if rss else None,
                verified_images=sum(r.get('verified',False) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'reports/dataset_evaluation_20261027.json')
    parser.add_argument('--slow-trace',type=Path,default=ROOT/'reports/photographic_complete_20261027/trace.jsonl')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf8'))
    validate_manifest(manifest)
    root = Path(manifest['image_root']).resolve()
    by_hash = {}
    for record in manifest['records']:
        path = (root/record['path']).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=record['sha256']:
            raise ValueError('El inventario cambió')
        by_hash.setdefault(record['sha256'],record)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source_files = ['modulos/ocr_reader.py','modulos/ocr_engine.py','modulos/ocr_analysis.py',
                    'modulos/ocr_worker.py','ui/main_window.py','modulos/camera_service.py','config.py',
                    'ocr_tools/evaluate_recovery.py']
    sources = {p:file_hash(ROOT/p) for p in source_files}
    write_json(args.output/'protocol.json',dict(manifest_sha256=file_hash(args.manifest),
               slow_trace_sha256=file_hash(args.slow_trace),source_sha256=sources,
               records=len(manifest['records']),unique_hashes=len(by_hash),
               references_passed_to_inference=False,fast_config=ReaderConfig().__dict__,
               note='Exploratory existing bank; no new independent real-world holdout or verified labels.'))
    slow = {}
    for line in args.slow_trace.read_text(encoding='utf8').splitlines():
        r = json.loads(line)
        if r['engine']=='experimental':
            slow.setdefault(r['sha256'],r)
    baseline,baseline_hash = load_baseline()
    engine = ocr_engine.get_ocr()
    engine.ocr(np.zeros((80,240,3),np.uint8),cls=True)
    process = psutil.Process()
    rows = []
    with (args.output/'trace.jsonl').open('x',encoding='utf8') as trace:
        for index,(digest,record) in enumerate(by_hash.items()):
            image = read_image(root/record['path'])
            base = dict(path=record['path'],sha256=digest,verified=record['verified'],
                        reference_code=record['reference_code'],label=record['label'])
            order = ['original','recovered_original','bounded_fast']
            if index%2:
                order.reverse()
            for name in order:
                row = dict(base,engine=name,text='',identifier=None,error=None)
                diagnostic = None
                cpu_start = sum(process.cpu_times()[:2])
                start = time.perf_counter()
                try:
                    if image is None:
                        raise ValueError('Imagen no decodificable')
                    if name=='original':
                        recorder = Recorder(engine)
                        with patch.object(baseline,'get_ocr',return_value=recorder):
                            text,score,variant,confidence = baseline.mejor_ocr(image)
                        row.update(text=text,ocr_calls=len(recorder.calls),confidence=confidence,variant=variant,status='legacy_text')
                        diagnostic = dict(calls=recorder.calls)
                    else:
                        result = (recognize_original if name=='recovered_original' else recognize_fast)(image,image_id=digest)
                        diagnostic = result.to_dict()
                        row.update(text=result.recognized_text,identifier=result.confirmed_identifier,
                                   ocr_calls=result.ocr_calls,confidence=result.confidence,
                                   variant=result.selected_variant,status=result.status,error=result.error,
                                   candidates=[c.text for c in result.candidates],stop_reason=result.stop_reason)
                except Exception as exc:
                    row.update(error=f'{type(exc).__name__}: {exc}',ocr_calls=row.get('ocr_calls',0),status='error_tecnico')
                row.update(elapsed_s=time.perf_counter()-start,cpu_s=sum(process.cpu_times()[:2])-cpu_start,
                           rss_bytes=process.memory_info().rss,measurement_origin='current_run')
                # Las referencias se consultan después de cerrar la inferencia.
                verified = row['verified'] and row['label']=='readable'
                row['exact_raw_text'] = row['text']==row['reference_code'] if verified else None
                row['character_errors_raw_text'] = character_errors(row['reference_code'],row['text']) if verified else None
                rows.append(row)
                trace.write(json.dumps(dict(row,diagnostic=diagnostic),ensure_ascii=False,default=json_native)+'\n')
                trace.flush()
            previous = slow.get(digest)
            if previous:
                row = dict(base,engine='slow_experimental',text=previous['suggested_text'] or '',
                           identifier=previous['confirmed_identifier'],status=previous['status'],
                           elapsed_s=previous['elapsed_s'],ocr_calls=previous['ocr_calls'],error=previous['error'],
                           measurement_origin=str(args.slow_trace),cpu_s=None,rss_bytes=0)
                rows.append(row)
                trace.write(json.dumps(row,ensure_ascii=False)+'\n')
                trace.flush()
            print(f'{index+1}/{len(by_hash)} unique images',flush=True)
    grouped = {name:[r for r in rows if r['engine']==name] for name in ('original','recovered_original','bounded_fast','slow_experimental')}
    stats = {name:summary(rs) for name,rs in grouped.items()}
    original = {r['sha256']:r for r in grouped['original']}
    recovered = {r['sha256']:r for r in grouped['recovered_original']}
    restored = sum(original[h]['text']==recovered[h]['text'] for h in original)
    fast = {r['sha256']:r for r in grouped['bounded_fast']}
    results = dict(engines=stats,exact_original_output_restored=restored,compared_unique=len(original),
                   environment=environment(engine),baseline_source_sha256=baseline_hash,
                   source_sha256=sources,unique_images=list(by_hash),
                   warning='Historical B timings are not a contemporaneous paired benchmark. Zero verified labels; no accuracy improvement claim.')
    write_json(args.output/'summary.json',results)
    columns = ['path','sha256','engine','text','status','confidence','variant','candidates','ocr_calls','elapsed_s','cpu_s',
               'rss_bytes','measurement_origin','verified','reference_code','exact_raw_text','character_errors_raw_text','error']
    with (args.output/'results.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
