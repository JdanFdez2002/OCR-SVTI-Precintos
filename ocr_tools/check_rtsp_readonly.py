"""Sondeo RTSP de sólo lectura; nunca configura PTZ, enfoque ni imagen.

No imprime URL, IP, usuario, contraseña ni guarda fotografías de la cámara.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

from modulos.camera_service import CameraService
from modulos.settings_store import load_camera_settings,settings_completos
from ocr_tools.evaluate_ocr import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = load_camera_settings()
    report = dict(kind='read_only_rtsp_probe_no_http_no_ptz_no_saved_camera_images',
                  configured=settings_completos(cfg),received_frames=0,status='not_configured')
    if report['configured']:
        service = CameraService(cfg.usuario,cfg.password)
        # FFmpeg puede escribir sus diagnósticos al descriptor nativo stderr.
        # Se silencia durante el sondeo para no exponer URLs autenticadas.
        original_stderr = os.dup(2)
        try:
            with open(os.devnull,'w') as sink:
                os.dup2(sink.fileno(),2)
                service.iniciar_lectura(CameraService.url_rtsp(cfg.usuario,cfg.password,cfg.ip,cfg.canal))
                start = time.monotonic()
                last = -1
                ages = []
                while time.monotonic()-start<8.:
                    sample = service.copiar_muestra(after_sequence=last)
                    if sample is not None and sample[0] is not None:
                        frame,last,captured_at,epoch = sample
                        report['received_frames'] += 1
                        report['last_shape'] = list(frame.shape)
                        ages.append(time.monotonic()-captured_at)
                    time.sleep(.02)
                report['duration_s'] = time.monotonic()-start
                report['max_sample_age_s'] = max(ages) if ages else None
                report['status'] = 'fresh_frames_received' if ages else 'no_fresh_frames_within_8s'
        except Exception as exc:
            report['status'],report['error_type'] = 'probe_error',type(exc).__name__
        finally:
            service.cerrar()
            os.dup2(original_stderr,2)
            os.close(original_stderr)
    write_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
