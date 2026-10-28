"""Qt offscreen con fotografía repetida o RTSP real de sólo lectura.

Compara carga anterior reactivada explícitamente con la configuración recuperada.
El modo --live-rtsp incluye decodificación real; nunca inicia el worker PTZ.
No certifica la presentación en el monitor Windows.
"""
import argparse
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
import numpy as np
import cv2
import psutil
from PySide6.QtCore import QEventLoop,QTimer
from PySide6.QtGui import QFontDatabase,QFont
from PySide6.QtWidgets import QApplication

from modulos import ocr_engine
from modulos.ptz_worker import PtzWorker
from ocr_tools.audit_ocr import ROOT
from ocr_tools.evaluate_ocr import read_image,write_json
from ui.main_window import MainWindow
from modulos.camera_service import CameraService
from modulos.settings_store import load_camera_settings,settings_completos


class ReplayedCamera:
    ultimo_error = None
    def __init__(self,frame):
        self.frame = frame
        self.start = time.monotonic()
    def copiar_muestra(self,*,after_sequence=None):
        now = time.monotonic()
        sequence = int((now-self.start)*15)
        return (None if after_sequence==sequence else self.frame.copy(),sequence,now,1)
    def detener_lectura(self):
        pass


def measure(app,frame,previous,duration,service=None,manual=False):
    with patch.object(PtzWorker,'start'),patch.object(MainWindow,'_inicio'):
        window = MainWindow()
    window._timer.stop()
    window._service = service if service is not None else ReplayedCamera(frame)
    window._ocr_listo = True
    window._ocr.live_detection_enabled = previous
    window._ocr.live_ocr_enabled = previous
    window.resize(1280,760)
    window.show()
    times,durations = [],[]
    counts = dict(conversions=0,video_jobs=0,detector_calls=0,ocr_calls=0)
    manual_result = dict(requested=0,completed=0,elapsed_s=None)
    def manual_read():
        sample = window._service.copiar_muestra()
        if sample is not None and sample[0] is not None:
            manual_result['requested'] += 1
            # Sólo se mide OCR; no guardar la imagen/texto de la cámara.
            window._ocr.pedir_ocr(sample[0])
    def manual_done(packet):
        if manual and packet['context'] is None:
            manual_result['completed'] += 1
            manual_result['elapsed_s'] = packet['analysis'].elapsed_s
    window._ocr.analyzed.connect(manual_done)
    display,video_job = window._video.mostrar_frame,window._ocr.pedir_video
    detect,ocr = ocr_engine.detectar_texto,ocr_engine.ejecutar_ocr
    def display_count(*args,**kwargs):
        counts['conversions'] += 1
        return display(*args,**kwargs)
    def job_count(*args,**kwargs):
        counts['video_jobs'] += 1
        return video_job(*args,**kwargs)
    def detect_count(*args,**kwargs):
        counts['detector_calls'] += 1
        return detect(*args,**kwargs)
    def ocr_count(*args,**kwargs):
        counts['ocr_calls'] += 1
        return ocr(*args,**kwargs)
    def tick():
        if previous:
            # Antes se copiaba y convertía el último frame incluso si era el mismo.
            window._last_display_sequence = -1
        start = time.perf_counter()
        times.append(start)
        window._tick_video()
        durations.append(time.perf_counter()-start)
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(tick)
    process = psutil.Process()
    cpu_start,wall_start = sum(process.cpu_times()[:2]),time.perf_counter()
    # Funciones simples: un MagicMock guardaría todos los arrays en call_args_list
    # y falsificaría la medición de memoria y de carga de la UI.
    with patch('ui.main_window.DETECCION_EN_VIVO',previous),patch.object(window._video,'mostrar_frame',display_count),patch.object(window._ocr,'pedir_video',job_count),patch.object(ocr_engine,'detectar_texto',detect_count),patch.object(ocr_engine,'ejecutar_ocr',ocr_count):
        timer.start(33)
        QTimer.singleShot(int(duration*1000),loop.quit)
        if manual:
            QTimer.singleShot(2000,manual_read)
        loop.exec()
        timer.stop()
        cpu_s,wall_s = sum(process.cpu_times()[:2])-cpu_start,time.perf_counter()-wall_start
        window._ocr.detener()
        if not window._ocr.wait(10000):
            raise RuntimeError('El predictor no terminó para cerrar el benchmark')
    intervals = np.diff(times)*1000
    result = dict(mode='previous_workload_reenabled' if previous else 'recovered_defaults',
                  duration_s=wall_s,cpu_s=cpu_s,mean_cpu_one_core_percent=100*cpu_s/wall_s,
                  rss_sampled_bytes=process.memory_info().rss,ticks=len(times),
                  timer_interval_median_ms=float(np.median(intervals)),timer_interval_p95_ms=float(np.percentile(intervals,95)),
                  intervals_over_66ms=int((intervals>66).sum()),tick_processing_p95_ms=float(np.percentile(durations,95)*1000),
                  manual_ocr=manual_result,**counts)
    app.removeEventFilter(window)
    window._service = None
    window._closing = True
    window._ptz.detener_hilo()
    window.close()
    app.processEvents()
    window.deleteLater()
    app.processEvents()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--duration',type=float,default=10.)
    parser.add_argument('--live-rtsp',action='store_true',help='RTSP real de sólo lectura; ningún worker PTZ se inicia')
    parser.add_argument('--manual-ocr',action='store_true',help='Solicita una lectura manual en cada fase, sin guardar imagen ni texto')
    parser.add_argument('--opencv-threads',type=int,help='Límite experimental de hilos OpenCV para medir competencia por CPU')
    args = parser.parse_args()
    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((ROOT/'reports/dataset_evaluation_20261027.json').read_text(encoding='utf8'))
    path = Path(manifest['image_root'])/manifest['records'][0]['path']
    frame = read_image(path)
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    font_path = Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/segoeui.ttf'
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    if font_id>=0:
        app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0],10))
    ocr_engine.get_ocr()
    ocr_engine.ejecutar_ocr(np.zeros((80,240,3),np.uint8))
    service,original_stderr = None,None
    try:
        if args.live_rtsp:
            cfg = load_camera_settings()
            if not settings_completos(cfg):
                raise ValueError('No hay configuración de cámara completa')
            original_stderr = os.dup(2)
            with open(os.devnull,'w') as sink:
                os.dup2(sink.fileno(),2)
            service = CameraService(cfg.usuario,cfg.password)
            service.iniciar_lectura(CameraService.url_rtsp(cfg.usuario,cfg.password,cfg.ip,cfg.canal))
            start = time.monotonic()
            sample = None
            while sample is None and time.monotonic()-start<5.:
                sample = service.copiar_muestra()
                time.sleep(.02)
            if sample is None:
                raise RuntimeError('No llegaron fotogramas recientes de la cámara')
            frame = sample[0]
        before = measure(app,frame,True,args.duration,service,args.manual_ocr)
        after = measure(app,frame,False,args.duration,service,args.manual_ocr)
    finally:
        if service is not None:
            service.cerrar()
        if original_stderr is not None:
            os.dup2(original_stderr,2)
            os.close(original_stderr)
    result = dict(kind='qt_offscreen_live_rtsp_real_model_no_ptz' if args.live_rtsp else 'qt_offscreen_real_model_replayed_photo_generated_15fps_no_rtsp',
                  photo_sha256=None if args.live_rtsp else manifest['records'][0]['sha256'],source_shape=list(frame.shape),
                  before=before,after=after,opencv_threads=cv2.getNumThreads(),
                  limitation='Previous workload is reenabled in current UI. Qt offscreen; no Windows monitor rendering or PTZ/focus validation. Sequential phases, not randomized.')
    write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
