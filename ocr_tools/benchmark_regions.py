"""Medición del detector real sobre TODO el inventario y seguimiento SINTÉTICO.

No conecta cámaras, no mide tracking RTSP real ni acredita seis cámaras.
Ejecutar sin otras inferencias concurrentes para medir latencia útil.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
import psutil

from modulos.ocr_engine import detectar_texto, get_ocr
from modulos.region_tracker import RegionTracker
from ocr_tools.audit_ocr import environment
from ocr_tools.evaluate_ocr import read_image, validate_manifest, write_json


def stats(values):
    return {'n':len(values),'median_s':statistics.median(values) if values else None,
            'p95_s':float(np.percentile(values,95)) if values else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text(encoding='utf8'))
    validate_manifest(manifest)
    root = Path(manifest['image_root']).resolve()
    engine = get_ocr()
    detectar_texto(np.zeros((100,200,3),np.uint8))
    process = psutil.Process()
    cpu0,wall0 = sum(process.cpu_times()[:2]),time.perf_counter()
    peak_rss = process.memory_info().rss
    rows = []
    for i,record in enumerate(manifest['records']):
        path = (root/record['path']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Ruta/hash alterado')
        row = dict(path=record['path'],sha256=record['sha256'])
        try:
            image = read_image(path)
            if image is None:
                raise ValueError('Imagen ilegible')
            tracker = RegionTracker(detector=detectar_texto)
            packet = tracker.update(image)
            row.update(detector_s=packet['detector_s'],processing_s=packet['processing_s'],
                       regions=packet['regions'],shape=packet['shape'])
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
        peak_rss = max(peak_rss,process.memory_info().rss)
        print(f"{i+1}/{len(manifest['records'])} detector",flush=True)
    wall = time.perf_counter()-wall0
    cpu = sum(process.cpu_times()[:2])-cpu0
    # Secuencia artificial de textura conocida, NO vídeo real de precintos.
    image = np.zeros((540,960,3),np.uint8)
    image[200:260,300:650] = np.random.default_rng(17).integers(0,256,(60,350,3),np.uint8)
    clock = [0.]
    tracker = RegionTracker(30.,lambda _: [[[[300,200],[650,200],[650,260],[300,260]]]],lambda:clock[0])
    tracker.update(image)
    track_times,position_errors = [],[]
    for i in range(1,31):
        clock[0] = i*.1
        moved = cv2.warpAffine(image,np.float32([[1,0,i],[0,1,0]]),(960,540))
        packet = tracker.update(moved)
        track_times.append(packet['processing_s'])
        position_errors.append(abs(packet['regions'][0]['box'][0]-(300+i)) if packet['regions'] else None)
    result = {'kind':'real_model_photos_detector_and_synthetic_tracking_no_camera',
              'environment':environment(engine),'all_available_files':len(rows),
              'unique_sha256':len({r['sha256'] for r in rows}),
              'detector_latency':stats([r['detector_s'] for r in rows if 'detector_s' in r]),
              'synthetic_tracking_latency':stats(track_times),
              'synthetic_tracking_error_px':position_errors,'rows':rows,
              'resources':{'wall_s':wall,'cpu_s':cpu,'mean_cpu_one_core_percent':100*cpu/wall,
                           'peak_rss_sampled_bytes':peak_rss},
              'six_cameras_validated':False,
              'limitation':'No hay vídeo real con anotaciones de objetivo, oclusiones, zoom ni seis streams simultáneos.'}
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'rows','environment'}},indent=2))


if __name__=='__main__':
    main()
