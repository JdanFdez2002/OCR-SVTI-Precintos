"""Compatibilidad para el seguimiento de regiones OCR desactivado."""


def iou(a, b):
    """Compatibilidad histórica; no se calculan regiones OCR."""
    return 0.0


class RegionTracker:
    """Mantiene la interfaz de video sin ejecutar detección ni tracking."""

    def __init__(self, detection_interval=2.0, detector=None, clock=None):
        self.interval = detection_interval
        self.detector = detector
        self.clock = clock
        self.reset()

    def reset(self):
        self.regions = []
        self.previous = None
        self.last_detection = float("-inf")
        self._next_id = 0

    def update(self, frame):
        shape = list(getattr(frame, "shape", ()))
        return {
            "regions": [],
            "shape": shape,
            "detection": False,
            "detector_s": 0.0,
            "processing_s": 0.0,
        }
