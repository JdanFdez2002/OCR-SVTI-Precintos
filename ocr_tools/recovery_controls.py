"""Modelo real sobre controles gráficos de códigos ajenos al banco.

No demuestra generalización industrial: son glifos renderizados con OpenCV.
Las referencias se usan para dibujar entradas y se evalúan después de inferir.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from modulos.ocr_reader import ReaderConfig,recognize_fast,recognize_original
from ocr_tools.evaluate_ocr import character_errors,distance,write_json


def rotate(image,angle):
    return image if angle==0 else cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE if angle==90 else cv2.ROTATE_90_COUNTERCLOCKWISE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cases = []
    for index,text in enumerate(('ZX-591874Q','R8 01473-Z','0081759','Q-72','KLMNOP','FX20-9T')):
        image = np.full((180,640,3),255,np.uint8)
        cv2.putText(image,text,(35,110),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),2,cv2.LINE_AA)
        for angle in (0,-90,90):
            cases.append(dict(name=f'code_{index}_{angle}',image=rotate(image,angle),reference=text,source_rotation=angle))
    cases.append(dict(name='blank',image=np.full((180,640,3),255,np.uint8),reference=None,source_rotation=0))
    cases.append(dict(name='noise',image=np.random.default_rng(18).integers(0,256,(180,640,3),np.uint8),reference=None,source_rotation=0))
    rows = []
    with (args.output/'trace.jsonl').open('x',encoding='utf8') as stream:
        for case in cases:
            path = args.output/(case['name']+'.png')
            cv2.imencode('.png',case['image'])[1].tofile(str(path))
            for method in ('original','fast'):
                result = (recognize_original if method=='original' else recognize_fast)(case['image'])
                # Ningún valor esperado entró a recognize_*.
                reference = case['reference']
                row = dict(case=case['name'],source_rotation=case['source_rotation'],engine=method,
                           reference=reference,text=result.recognized_text,elapsed_s=result.elapsed_s,
                           ocr_calls=result.ocr_calls,status=result.status,candidates=[c.text for c in result.candidates],
                           exact_text=result.recognized_text==reference if reference else None,
                           character_errors=character_errors(reference,result.recognized_text) if reference else None,
                           exact_candidate=any(c.text==reference for c in result.candidates) if reference else None,
                           diagnostic=result.to_dict())
                rows.append(row)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                stream.flush()
            print(case['name'],flush=True)
    summary = dict(kind='real_model_synthetic_unknown_code_controls_no_industrial_generalization_claim',
                   cases=len(cases),positive_images=18,negative_images=2,
                   references_passed_to_inference=False,
                   engines={m:dict(exact_text=sum(r['exact_text'] is True for r in rows if r['engine']==m),
                                   exact_candidate=sum(r['exact_candidate'] is True for r in rows if r['engine']==m),
                                   empty_negatives=sum(not r['text'] for r in rows if r['engine']==m and r['reference'] is None))
                            for m in ('original','fast')})
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
