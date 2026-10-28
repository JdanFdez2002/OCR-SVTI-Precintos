"""Worker Qt para el OCR de capturas manuales."""

import threading

from PySide6.QtCore import QThread, Signal


class OcrWorker(QThread):
    engine_ready = Signal()
    engine_failed = Signal(str)
    result = Signal(str, float, str, float)
    analyzed = Signal(object)
    failed = Signal(str)
    analysis_failed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._stopped = False
        self._warmup = False
        self._manual = None
        self._epoch = 0

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

    def invalidar(self):
        with self._condition:
            self._epoch += 1
            self._manual = None
            self._condition.notify()

    def detener(self):
        with self._condition:
            self._stopped = True
            self._epoch += 1
            self._manual = None
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
                    from modulos.ocr_engine import warmup_ocr
                    warmup_ocr()
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
                self.result.emit(analysis.recognized_text, analysis.selection_score,
                                 analysis.selected_variant, analysis.confidence)
            except Exception as exc:
                if kind == "warmup":
                    self.engine_failed.emit(str(exc))
                elif self._valid(job[0]):
                    self.analysis_failed.emit({"context": job[4], "error": str(exc)})
                    self.failed.emit(str(exc))
