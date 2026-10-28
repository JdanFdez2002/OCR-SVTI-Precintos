"""Contratos de lectura universal y regresiones de captura, sin cámara física."""
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from modulos.ocr_reader import ReaderConfig, recognize_fast, recognize_original
from modulos.ocr_worker import OcrWorker
from ocr_tools.audit_ocr import load_baseline


def raw(text='ZX-591874Q',confidence=.96):
    return [[[[[20,20],[190,20],[190,45],[20,45]],(text,confidence)]]]


class OriginalRecoveryTests(unittest.TestCase):
    def test_original_text_with_extra_lines_is_kept(self):
        image = np.zeros((100,240,3),np.uint8)
        response = raw('UNKNOWN ZX-591874Q EXTRA')
        result = recognize_original(image,infer=lambda _:response)
        self.assertEqual(result.recognized_text,'UNKNOWN ZX-591874Q EXTRA')
        self.assertEqual(result.ocr_calls,1)
        self.assertEqual(result.detections[0].raw_text,'UNKNOWN ZX-591874Q EXTRA')
        self.assertIsNone(result.confirmed_identifier)

    def test_no_manifest_or_expected_code_is_an_input(self):
        self.assertNotIn('manifest',inspect.signature(recognize_original).parameters)
        self.assertNotIn('expected',inspect.signature(recognize_original).parameters)
        image = np.zeros((100,240,3),np.uint8)
        for text in ('ZX-591874Q','Q 07125-Z','0197','RANDOM', '9'):
            result = recognize_original(image,infer=lambda _,t=text:raw(t))
            self.assertIn(text,result.recognized_text)

    def test_traced_original_preserves_legacy_exit_and_full_text(self):
        baseline,_ = load_baseline()
        image = np.zeros((100,240,3),np.uint8)
        values = [('BRAND ZX-591874Q EXTRA',.87),('ZX-591874Q',.99)]
        with patch.object(baseline,'_ocr_imagen',side_effect=values):
            original = baseline.mejor_ocr(image)
        counter = iter(values)
        result = recognize_original(image,infer=lambda _:raw(*next(counter)))
        self.assertEqual(result.recognized_text,original[0])
        self.assertEqual(result.confidence,original[3])
        self.assertEqual(result.selection_score,original[1])
        self.assertEqual(result.ocr_calls,2)
        self.assertEqual(len(result.raw_observations),2)

    def test_live_inference_disabled_by_default_at_worker_boundary(self):
        worker = OcrWorker()
        self.assertFalse(worker.live_detection_enabled)
        self.assertFalse(worker.live_ocr_enabled)
        worker.pedir_video(np.zeros((100,240,3),np.uint8),generation=1,sequence=1,captured_at=0.,stream_epoch=1)
        self.assertIsNone(worker._live)

    def test_original_low_confidence_keeps_text_with_uncertainty(self):
        image = np.zeros((100,240,3),np.uint8)
        result = recognize_original(image,infer=lambda _:raw('ZX-591874Q',.7))
        self.assertEqual(result.recognized_text,'ZX-591874Q')
        self.assertEqual(result.status,'lectura_incierta')


class FastReaderTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(95).integers(0,256,(100,240,3),np.uint8)

    def observations(self,texts,confidences=None):
        index = [0]
        def infer(array):
            i = index[0]
            index[0] += 1
            original = np.array([[20,20],[190,20],[190,45],[20,45]],dtype=float)
            if i==1:
                original = np.column_stack((99-original[:,1],original[:,0]))
            elif i==2:
                original = np.column_stack((original[:,1],239-original[:,0]))
            return [[[original.tolist(),(texts[min(i,len(texts)-1)],confidences[i] if confidences else .97)]]]
        return infer

    def test_two_orientations_can_exit_without_expected_identifier(self):
        result = recognize_fast(self.image,infer=self.observations(['ZX-591874Q']*3))
        self.assertEqual(result.recognized_text,'ZX-591874Q')
        self.assertEqual(result.ocr_calls,2)
        self.assertEqual(result.stop_reason,'visual_consensus')

    def test_complete_observation_not_replaced_by_higher_confidence_truncation(self):
        result = recognize_fast(self.image,infer=self.observations(['ZX-591874Q','ZX-5918'],[.96,.99]),
                                config=ReaderConfig(max_calls=2))
        self.assertEqual(result.recognized_text,'ZX-591874Q')
        self.assertEqual(result.ocr_calls,2)
        self.assertEqual({c.text for c in result.candidates},{'ZX-591874Q','ZX-5918'})
        self.assertTrue(result.budget_exhausted)

    def test_full_text_extra_words_not_rejected(self):
        result = recognize_fast(self.image,infer=self.observations(['LABEL ZX-591874Q EXTRA']*3))
        self.assertEqual(result.recognized_text,'LABEL ZX-591874Q EXTRA')

    def test_conflicting_real_characters_never_corrected_or_hidden(self):
        result = recognize_fast(self.image,infer=self.observations(['ZX-591870Q','ZX-59187OQ']),
                                config=ReaderConfig(max_calls=2))
        self.assertEqual(result.status,'lectura_incierta')
        self.assertEqual({c.text for c in result.candidates},{'ZX-591870Q','ZX-59187OQ'})

    def test_inference_and_time_budgets_preserve_partial_evidence(self):
        import time
        def slow(array):
            time.sleep(.02)
            return raw('QZ7',.75)
        result = recognize_fast(self.image,infer=slow,config=ReaderConfig(max_calls=6,max_seconds=.01))
        self.assertEqual(result.ocr_calls,1)
        self.assertEqual(result.stop_reason,'time_budget')
        self.assertEqual(result.recognized_text,'QZ7')

    def test_unknown_pure_letters_and_short_codes_remain_useful(self):
        for text in ('RANDOM','9','Q-02 Z','TR-846201X'):
            result = recognize_fast(self.image,infer=self.observations([text]*3))
            self.assertEqual(result.recognized_text,text)


if __name__=='__main__':
    unittest.main()
