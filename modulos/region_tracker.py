"""Detector de texto existente + flujo óptico. No identifica semánticamente precintos."""
import time

import cv2
import numpy as np


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - intersection
    return intersection / union if union else 0


class RegionTracker:
    def __init__(self, detection_interval=2.0, detector=None, clock=time.monotonic):
        if not .25 <= detection_interval <= 30:
            raise ValueError('Intervalo de detección fuera de .25..30 segundos')
        self.interval, self.detector, self.clock = detection_interval, detector, clock
        self.reset()

    def reset(self):
        self.regions = []
        self.previous = None
        self.last_detection = float('-inf')
        self._next_id = 0

    def _features(self, gray, box):
        mask = np.zeros_like(gray)
        x1, y1, x2, y2 = map(int, box)
        mask[max(0,y1):max(0,y2), max(0,x1):max(0,x2)] = 255
        return cv2.goodFeaturesToTrack(gray, 30, .02, 3, mask=mask)

    def _track(self, gray):
        if self.previous is None or self.previous.shape != gray.shape:
            self.regions = []
            return
        retained = []
        for region in self.regions:
            points = region['points']
            if points is None or len(points) < 4:
                continue
            forward, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None)
            if forward is None:
                continue
            backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, forward, None)
            if backward is None:
                continue
            good = ((status.ravel() == 1) & (back_status.ravel() == 1) &
                    (np.linalg.norm(backward - points, axis=2).ravel() < 1.5))
            if good.sum() < 4:
                continue
            delta = (forward[good] - points[good]).reshape(-1, 2)
            motion = np.median(delta, axis=0)
            if np.median(np.linalg.norm(delta-motion, axis=1)) > 2.5:
                continue
            x1, y1, x2, y2 = np.asarray(region['box']) + np.tile(motion, 2)
            if x1 < 0 or y1 < 0 or x2 > gray.shape[1] or y2 > gray.shape[0]:
                continue
            if self.clock() - region['detected_at'] > self.interval * 1.75:
                continue
            region.update(box=[float(x1),float(y1),float(x2),float(y2)], points=forward[good])
            region['stable_steps'] = region['stable_steps'] + 1 if np.linalg.norm(motion) <= 2 else 0
            retained.append(region)
        self.regions = retained

    def update(self, frame):
        start, now = time.perf_counter(), self.clock()
        scale = min(1., 960 / max(frame.shape[:2]))
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self._track(gray)
        detected = now - self.last_detection >= self.interval
        detector_s = 0.
        if detected:
            if self.detector is None:
                from modulos.ocr_engine import detectar_texto
                self.detector = detectar_texto
            detect_start = time.perf_counter()
            raw = self.detector(small)
            detector_s = time.perf_counter() - detect_start
            boxes = []
            for polygon in (raw[0] or []) if raw else []:
                points = np.asarray(polygon, dtype=float)
                if points.shape != (4,2) or not np.isfinite(points).all():
                    continue
                x1,y1 = points.min(0)
                x2,y2 = points.max(0)
                # Orientaciones horizontales y verticales, sin imponer código o marca.
                width,height = x2-x1,y2-y1
                if min(width,height) < 5 or max(width,height) / min(width,height) < 1.2:
                    continue
                boxes.append([max(0.,x1),max(0.,y1),min(gray.shape[1],x2),min(gray.shape[0],y2)])
            regions, used = [], set()
            for box in boxes[:12]:
                matches = [(iou(box,r['box']),i,r) for i,r in enumerate(self.regions) if i not in used]
                overlap,index,old = max(matches, default=(0,-1,None), key=lambda item:item[0])
                if overlap >= .45:
                    used.add(index)
                    region = dict(old)
                    region['hits'] += 1
                else:
                    self._next_id += 1
                    region = dict(id=self._next_id,hits=1,stable_steps=0)
                region.update(box=box,points=self._features(gray,box),detected_at=now)
                regions.append(region)
            self.regions = regions
            self.last_detection = now
        self.previous = gray
        result = []
        for region in self.regions:
            result.append({k:v for k,v in region.items() if k != 'points'})
            result[-1]['box'] = [float(v/scale) for v in region['box']]
            result[-1]['stable'] = region['hits'] >= 2 and region['stable_steps'] >= 3
        return {'regions':result,'shape':list(frame.shape),'detection':detected,
                'detector_s':detector_s,'processing_s':time.perf_counter()-start}
