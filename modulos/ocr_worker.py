"""Un predictor/hilo: captura manual prioritaria y una sola muestra de video pendiente."""
import threading
import time

from PySide6.QtCore import QThread, Signal


class OcrWorker(QThread):
    engine_ready = Signal()
    engine_failed = Signal(str)
    result = Signal(str, float, str, float)  # Compatibilidad con MotorWindow.
    analyzed = Signal(object)
    regions = Signal(object)
    live_analysis = Signal(object)
    failed = Signal(str)
    analysis_failed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._stopped = False
        self._warmup = False
        self._manual = self._live = None
        self._epoch = 0
        self._tracker = None
        self._stream_key = None
        self._last_live_ocr = float('-inf')
        self.dropped_frames = 0
        from config import DETECCION_EN_VIVO, OCR_EN_VIVO
        self.live_detection_enabled = DETECCION_EN_VIVO
        self.live_ocr_enabled = OCR_EN_VIVO

    def pedir_warmup(self):
        with self._condition:
            self._warmup = True
            self._condition.notify()

    def pedir_ocr(self, frame_bgr, *, input_mode='full_photo', image_id='', context=None):
        with self._condition:
            if not self._stopped:
                self._manual = (self._epoch, frame_bgr.copy(), input_mode, image_id, context)
                self._condition.notify()

    def pedir_video(self, frame, *, generation, sequence, captured_at, stream_epoch):
        with self._condition:
            if self._stopped or not self.live_detection_enabled:
                return
            if self._live is not None:
                self.dropped_frames += 1
            self._live = (self._epoch, frame.copy(), dict(generation=generation, sequence=sequence,
                          captured_at=captured_at, stream_epoch=stream_epoch))
            self._condition.notify()

    def invalidar(self):
        with self._condition:
            self._epoch += 1
            self._manual = self._live = None
            self._condition.notify()

    def detener(self):
        with self._condition:
            self._stopped = True
            self._epoch += 1
            self._manual = self._live = None
            self._condition.notify()

    def _valid(self, epoch):
        with self._condition:
            return not self._stopped and epoch == self._epoch

    def run(self):
        from config import DETECCION_INTERVALO_S, OCR_DINAMICO_INTERVALO_S, REGION_MAX_AGE_S, recorte_con_margen
        from modulos.ocr_analysis import AnalysisConfig, analyze
        from modulos.ocr_reader import recognize
        from modulos.region_tracker import RegionTracker

        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopped or self._warmup or self._manual is not None or self._live is not None)
                if self._stopped:
                    return
                if self._warmup:
                    self._warmup = False
                    kind, job = 'warmup', None
                elif self._manual is not None:
                    kind, job = 'manual', self._manual
                    self._manual = None
                else:
                    kind, job = 'video', self._live
                    self._live = None
            try:
                if kind == 'warmup':
                    from modulos.ocr_engine import warmup_ocr
                    warmup_ocr()
                    if not self._stopped:
                        self.engine_ready.emit()
                elif kind == 'manual':
                    epoch, frame, mode, image_id, context = job
                    if not self._valid(epoch):
                        continue
                    if context and context.get('save_input'):
                        import cv2
                        import hashlib
                        encoded, data = cv2.imencode('.png',frame)
                        if not encoded:
                            raise OSError('No se pudo codificar la evidencia visual')
                        data.tofile(context['evidence']['full_image'])
                        image_id = hashlib.sha256(frame.tobytes()).hexdigest()
                        context['evidence']['input_sha256'] = image_id
                    analysis = recognize(frame, input_mode=mode, image_id=image_id,
                                         cancelled=lambda: not self._valid(epoch))
                    if self._valid(epoch):
                        self.analyzed.emit(dict(analysis=analysis, context=context))
                        text = getattr(analysis,'recognized_text','') or analysis.confirmed_identifier or analysis.suggested_text or ''
                        best = next((c for c in analysis.candidates if c.text == text), None)
                        confidence = getattr(analysis,'confidence',best.confidence_min if best else 0.)
                        self.result.emit(text, getattr(analysis,'selection_score',best.score if best else confidence),
                                         getattr(analysis,'selected_variant',analysis.status), confidence)
                else:
                    epoch, frame, context = job
                    if not self._valid(epoch) or time.monotonic() - context['captured_at'] > REGION_MAX_AGE_S:
                        self.dropped_frames += 1
                        continue
                    key = (epoch, context['generation'], context['stream_epoch'], frame.shape)
                    if self._tracker is None or key != self._stream_key:
                        self._tracker = RegionTracker(DETECCION_INTERVALO_S)
                        self._stream_key = key
                    packet = self._tracker.update(frame)
                    packet.update(context, dropped_frames=self.dropped_frames)
                    if self._valid(epoch):
                        self.regions.emit(packet)
                    stable = [r for r in packet['regions'] if r['stable']]
                    with self._condition:
                        busy = self._manual is not None
                    if (self.live_ocr_enabled and stable and not busy and self._valid(epoch)
                            and time.monotonic() - context['captured_at'] <= REGION_MAX_AGE_S
                            and time.monotonic() - self._last_live_ocr >= OCR_DINAMICO_INTERVALO_S):
                        # Un candidato de texto jamás se declara ROI semántico de precinto.
                        crop, bounds = recorte_con_margen(frame, stable[0]['box'])
                        self._last_live_ocr = time.monotonic()
                        analysis = analyze(crop, input_mode='full_photo', config=AnalysisConfig(max_calls=2),
                                           cancelled=lambda: not self._valid(epoch) or self._manual is not None)
                        if self._valid(epoch):
                            self.live_analysis.emit(dict(analysis=analysis, context=context, bounds=bounds))
            except Exception as exc:
                if kind == 'warmup':
                    self.engine_failed.emit(str(exc))
                elif self._valid(job[0]):
                    # Los errores de detección no se presentan como capturas manuales fallidas.
                    if kind == 'video':
                        self._tracker = None
                        self.regions.emit(dict(**job[2], regions=[], shape=list(job[1].shape), error=str(exc)))
                    else:
                        self.analysis_failed.emit(dict(context=job[4], error=str(exc)))
                        self.failed.emit(str(exc))
