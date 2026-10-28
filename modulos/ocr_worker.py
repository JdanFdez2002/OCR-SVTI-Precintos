"""Worker Qt compatible con el motor OCR desactivado."""

import threading

from PySide6.QtCore import QThread, Signal


class OcrWorker(QThread):
    engine_ready = Signal()
    engine_failed = Signal(str)
    result = Signal(str, float, str, float)
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
        self.dropped_frames = 0
        self.live_detection_enabled = False
        self.live_ocr_enabled = False

    def pedir_warmup(self):
        with self._condition:
            self._warmup = True
            self._condition.notify()

    def pedir_ocr(self, frame_bgr, *, input_mode="full_photo", image_id="", context=None):
        with self._condition:
            if not self._stopped:
                self._manual = (
                    self._epoch,
                    frame_bgr.copy(),
                    input_mode,
                    image_id,
                    context,
                )
                self._condition.notify()

    def pedir_video(self, frame, *, generation, sequence, captured_at, stream_epoch):
        # El seguimiento de regiones OCR queda desactivado hasta implementar
        # el lector genérico. La captura y el video siguen en manos de la UI.
        return None

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
        from modulos.ocr_reader import recognize

        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stopped or self._warmup or self._manual is not None
                )
                if self._stopped:
                    return
                if self._warmup:
                    self._warmup = False
                    kind, job = "warmup", None
                else:
                    kind, job = "manual", self._manual
                    self._manual = None

            try:
                if kind == "warmup":
                    # El worker está disponible, pero no existe motor OCR.
                    if not self._stopped:
                        self.engine_ready.emit()
                    continue

                epoch, frame, mode, image_id, context = job
                if not self._valid(epoch):
                    continue
                if context and context.get("save_input"):
                    import hashlib
                    import cv2

                    encoded, data = cv2.imencode(".png", frame)
                    if not encoded:
                        raise OSError("No se pudo codificar la evidencia visual")
                    data.tofile(context["evidence"]["full_image"])
                    image_id = hashlib.sha256(frame.tobytes()).hexdigest()
                    context["evidence"]["input_sha256"] = image_id

                analysis = recognize(
                    frame,
                    input_mode=mode,
                    image_id=image_id,
                    cancelled=lambda: not self._valid(epoch),
                )
                if not self._valid(epoch):
                    continue
                self.analyzed.emit({"analysis": analysis, "context": context})
                self.result.emit("", 0.0, "no_implementado", 0.0)
            except Exception as exc:
                if kind == "manual" and self._valid(job[0]):
                    self.analysis_failed.emit({"context": job[4], "error": str(exc)})
                    self.failed.emit(str(exc))
