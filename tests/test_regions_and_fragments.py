import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from modulos.ocr_analysis import AnalysisConfig, Detection, analyze, join_fragments
from modulos.ocr_worker import OcrWorker
from modulos import ocr_engine
from modulos.region_tracker import RegionTracker


def box(x1,y1,x2,y2):
    return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]


def detection(text, bounds, observation='v1', conf=.97):
    return Detection(text,text,conf,bounds,'fixture',0,observation)


class FragmentTests(unittest.TestCase):
    def test_native_drop_filter_is_lowered_only_to_preserve_evidence_and_restored(self):
        class Predictor:
            drop_score = .5
            def ocr(self,array,cls):
                return [[[box(10,10,20,30),('B',.3)]]] if self.drop_score == 0 else [None]
        predictor = Predictor()
        with patch.object(ocr_engine,'get_ocr',return_value=predictor):
            self.assertEqual(ocr_engine.ejecutar_ocr(None),[None])
            raw = ocr_engine.ejecutar_ocr(None,conservar_fragmentos=True)
            self.assertEqual(raw[0][0][1],('B',.3))
            self.assertEqual(predictor.drop_score,.5)
            with patch.object(predictor,'ocr',side_effect=RuntimeError('fixture')):
                with self.assertRaises(RuntimeError):
                    ocr_engine.ejecutar_ocr(None,conservar_fragmentos=True)
            self.assertEqual(predictor.drop_score,.5)

    def test_short_prefix_is_preserved_as_geometric_hypothesis(self):
        ds = [detection('B',box(15,20,25,40)),detection('0097312',box(40,20,140,40))]
        joined = join_fragments(ds)
        self.assertEqual([d.normalized_text for d in joined],['B 0097312'])
        self.assertEqual(joined[0].fragments,['B','0097312'])
        self.assertEqual(ds[0].raw_text,'B')

    def test_brand_different_line_scale_and_distant_text_never_join(self):
        for other in (detection('BRAND',box(5,20,50,40)),detection('B',box(15,60,25,80)),
                      detection('B',box(15,5,25,15)),detection('B',box(180,20,190,40))):
            self.assertEqual(join_fragments([other,detection('0097312',box(65,20,145,40))]),[])

    def test_fragments_alone_do_not_prove_complete_identifier(self):
        raw = [[[box(15,20,25,40),('B',.97)],[box(40,20,140,40),('0097312',.97)]]]
        image = np.random.default_rng(94).integers(0,256,(100,200,3),np.uint8)
        result = analyze(image,input_mode='seal_roi',infer=lambda _:raw,config=AnalysisConfig(max_calls=2))
        self.assertEqual(result.status,'posiblemente_incompleto')
        self.assertIsNone(result.confirmed_identifier)
        self.assertIn('B 0097312',[c.text for c in result.candidates])
        self.assertEqual(len(result.detections),4)

    def test_complete_line_in_two_observations_can_support_fragments(self):
        raw = iter([[[[box(15,20,25,40),('B',.97)],[box(40,20,140,40),('0097312',.97)]]],
                    [[[box(15,20,140,40),('B 0097312',.98)]]]])
        image = np.random.default_rng(94).integers(0,256,(100,200,3),np.uint8)
        result = analyze(image,input_mode='seal_roi',infer=lambda _:next(raw),config=AnalysisConfig(max_calls=2))
        self.assertEqual(result.confirmed_identifier,'B 0097312')

    def test_edge_characters_and_late_rival_block_confirmation(self):
        image = np.random.default_rng(94).integers(0,256,(100,200,3),np.uint8)
        edge = [[[box(0,20,130,40),('B 0097312',.99)]]]
        result = analyze(image,input_mode='seal_roi',infer=lambda _:edge,config=AnalysisConfig(max_calls=2))
        self.assertEqual(result.status,'posiblemente_incompleto')
        raw = iter([[[[box(15,20,140,40),(text,.99)]]] for text in ('0097312','0097312','B 0097312')])
        result = analyze(image,input_mode='seal_roi',infer=lambda _:next(raw),config=AnalysisConfig(max_calls=3))
        self.assertEqual(result.ocr_calls,3)
        self.assertIsNone(result.confirmed_identifier)
        self.assertEqual({c.text for c in result.candidates},{'0097312','B 0097312'})

    def test_spaces_are_valid_without_changing_real_characters(self):
        image = np.random.default_rng(94).integers(0,256,(100,200,3),np.uint8)
        raw = [[[box(15,20,140,40),('B 0097312',.99)]]]
        result = analyze(image,input_mode='seal_roi',infer=lambda _:raw,config=AnalysisConfig(max_calls=2))
        self.assertEqual(result.confirmed_identifier,'B 0097312')


class RegionTests(unittest.TestCase):
    def test_native_detection_ndarray_does_not_use_broken_rec_false_wrapper(self):
        boxes = np.array([box(20,30,150,60)],dtype=np.float32)
        class Predictor:
            def text_detector(self,image):
                return boxes,.01
            def ocr(self,*args,**kwargs):
                raise AssertionError('El wrapper instalado falla sobre ndarray')
        with patch.object(ocr_engine,'get_ocr',return_value=Predictor()):
            self.assertEqual(ocr_engine.detectar_texto(None),[boxes.tolist()])

    def setUp(self):
        self.now = 0.
        self.image = np.zeros((180,320,3),np.uint8)
        rng = np.random.default_rng(4)
        self.image[50:90,60:200] = rng.integers(0,256,(40,140,3),np.uint8)
        self.calls = 0
        def detect(_):
            self.calls += 1
            return [[box(60,50,200,90)]]
        self.tracker = RegionTracker(2.,detect,lambda:self.now)

    def test_detector_is_periodic_and_stationary_region_becomes_stable(self):
        packet = self.tracker.update(self.image)
        self.assertFalse(packet['regions'][0]['stable'])
        for i in range(1,10):
            self.now = i*.25
            packet = self.tracker.update(self.image)
        self.assertEqual(self.calls,2)
        self.assertTrue(packet['regions'][0]['stable'])

    def test_optical_flow_follows_translation_and_invalidates_lost_target(self):
        self.tracker.update(self.image)
        self.now = .25
        moved = cv2.warpAffine(self.image,np.float32([[1,0,6],[0,1,3]]),(320,180))
        packet = self.tracker.update(moved)
        self.assertAlmostEqual(packet['regions'][0]['box'][0],66,delta=1)
        self.assertFalse(packet['regions'][0]['stable'])
        self.now = .5
        packet = self.tracker.update(np.zeros_like(self.image))
        self.assertEqual(packet['regions'],[])

    def test_shape_change_and_empty_detection_invalidate(self):
        self.tracker.update(self.image)
        self.now = .25
        self.assertEqual(self.tracker.update(np.zeros((90,160,3),np.uint8))['regions'],[])
        self.tracker.detector = lambda _: [None]
        self.now = 3.
        self.assertEqual(self.tracker.update(self.image)['regions'],[])

    def test_latest_frame_slot_and_invalidation_are_bounded(self):
        worker = OcrWorker()
        worker.live_detection_enabled = True  # Infraestructura experimental activada explícitamente.
        for seq in range(40):
            worker.pedir_video(self.image,generation=1,sequence=seq,captured_at=0.,stream_epoch=1)
        self.assertEqual(worker.dropped_frames,39)
        self.assertEqual(worker._live[2]['sequence'],39)
        worker.invalidar()
        self.assertIsNone(worker._live)
        worker.detener()
        worker.pedir_video(self.image,generation=1,sequence=40,captured_at=0.,stream_epoch=1)
        self.assertIsNone(worker._live)


if __name__=='__main__':
    unittest.main()
