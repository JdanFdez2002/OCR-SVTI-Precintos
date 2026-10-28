"""Reproduce fallos de la revisión original y contrasta correcciones sin hardware."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import threading
import types
from unittest.mock import Mock

import numpy as np

from modulos import camera_service, ptz_worker
from modulos.excel_manifest import RegistroEsperado
from modulos.identifier_policy import decide_arrival
from ocr_tools.audit_ocr import BASELINE_COMMIT, ROOT


def revision(path):
    source = subprocess.check_output(['git','show',f'{BASELINE_COMMIT}:{path}'],cwd=ROOT)
    module = types.ModuleType('baseline_'+Path(path).stem)
    exec(compile(source,'git:'+path,'exec'),module.__dict__)
    return module,hashlib.sha256(source).hexdigest()


def camera_probes(module):
    def service(status=200,body=''):
        s = module.CameraService('fixture','fixture')
        s.session = Mock()
        s.session.put.return_value = Mock(status_code=status,text=body)
        return s
    s = service(403)
    result = {'http_rejection_misreported_success':s.mover_camara('fixture',pan=20)}
    s = service(200,'<ResponseStatus><statusCode>4</statusCode></ResponseStatus>')
    result['isapi_rejection_misreported_success'] = s.mover_camara('fixture',focus=5)
    s = service()
    s.session.put.side_effect = [TimeoutError('fixture'),Mock(status_code=200,text='')]
    s.mover_camara('fixture')
    result['stop_attempted_both_axes'] = s.session.put.call_count == 2
    s = service()
    s.ultimo_frame = np.zeros((10,20,3),np.uint8)
    s.conectado = False
    result['disconnected_image_exposed_as_live'] = s.copiar_frame() is not None
    return result


def queue_probes(module):
    calls,done = [],threading.Event()
    service = Mock(ultimo_error=None)
    def move(ip,**cmd):
        calls.append(dict(ip=ip,**cmd))
        if cmd['pan']==20:
            done.set()
        return True
    service.mover_camara.side_effect = move
    worker = module.PtzWorker()
    worker.configurar(service,'fixture')
    worker.enviar(pan=10)
    worker.enviar()
    worker.enviar(pan=20)
    worker.start()
    try:
        if not done.wait(3):
            raise RuntimeError('No se despachó movimiento simulado')
        result = {'stop_delivered_before_next_move':len(calls)>=2 and all(calls[0][k]==0 for k in ('pan','tilt','zoom','focus'))}
    finally:
        worker.detener_hilo()
        if not worker.wait(5000):
            raise RuntimeError('Worker simulado no cerró')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports'/'safety_comparison_20261027.json')
    args = parser.parse_args()
    baseline_camera,cam_hash = revision('modulos/camera_service.py')
    baseline_ptz,ptz_hash = revision('modulos/ptz_worker.py')
    baseline_ui,ui_hash = revision('ui/main_window.py')
    def baseline_manifest(text,state='Pendiente'):
        dummy = Mock()
        dummy._manifiesto = [RegistroEsperado(precinto_esperado='B 0097312',estado=state)]
        dummy._normalizar_lectura.side_effect = lambda x:baseline_ui.MainWindow._normalizar_lectura(dummy,x)
        baseline_ui.MainWindow._actualizar_manifiesto_por_ocr(dummy,text)
        return dummy._manifiesto[0].estado
    result = {'kind':'simulated_services_no_hardware','baseline_commit':BASELINE_COMMIT,
              'baseline_sha256':{'camera_service':cam_hash,'ptz_worker':ptz_hash,'main_window':ui_hash},
              'baseline':{**camera_probes(baseline_camera),**queue_probes(baseline_ptz),
                          'extra_characters_arrival_state':baseline_manifest('XB 0097312Y'),
                          'later_partial_read_from_arrived':baseline_manifest('0097312','Llegado')},
              'current':{**camera_probes(camera_service),**queue_probes(ptz_worker)}}
    candidate = types.SimpleNamespace(text='XB 0097312Y',confidence_min=.99,
                                      observations=2,syntax_compatible=True,warnings=[])
    reading = types.SimpleNamespace(confirmed_identifier=candidate.text,status='reconocido',input_mode='seal_roi',candidates=[candidate])
    record = RegistroEsperado(precinto_esperado='B 0097312')
    result['current']['extra_characters_accepted'] = decide_arrival([record],reading).accepted
    out = args.output
    with out.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
