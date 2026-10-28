import json
from pathlib import Path
import tempfile
import unittest

from ocr_tools.evaluate_ocr import cached_measurements, character_errors, metrics
from ocr_tools.evaluate_recovery import summary


class EvaluationProvenanceTests(unittest.TestCase):
    def test_historical_resources_are_unknown_instead_of_zero(self):
        row = dict(elapsed_s=1.,text='ABC-78',ocr_calls=1,cpu_s=None,rss_bytes=0,error=None,verified=False)
        report = summary([row])
        self.assertIsNone(report['cpu_s'])
        self.assertIsNone(report['max_rss_sampled_bytes'])
    def test_character_operations_and_extremes_are_counted(self):
        cases = [('B009', '009', 'deletions','initial_errors'),
                 ('B009', 'B00', 'deletions','final_errors'),
                 ('B009', 'X009', 'substitutions','initial_errors'),
                 ('B009', 'B009Z', 'insertions','final_errors')]
        for expected,observed,operation,edge in cases:
            counts = character_errors(expected,observed)
            self.assertEqual(counts[operation],1)
            self.assertEqual(counts[edge],1)

    def test_imported_measurement_must_match_exact_inventory_and_protocol(self):
        manifest = {'records':[{'path':'a.png','sha256':'hash-a'}]}
        row = dict(path='a.png',sha256='hash-a',engine='baseline',repeat=0,
                   crop_policy='none',input_mode='full_photo',elapsed_s=1.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'trace.jsonl'
            path.write_text(json.dumps(row)+'\n',encoding='utf8')
            cached = cached_measurements(path,manifest,['baseline'],'none','full_photo',21)
            self.assertEqual(len(cached),1)
            self.assertEqual(cached['baseline','hash-a',0]['source_measurement_path'],'a.png')
            with self.assertRaises(ValueError):
                cached_measurements(path,manifest,['baseline'],'reticle','full_photo',21)
            row['sha256'] = 'different'
            path.write_text(json.dumps(row)+'\n',encoding='utf8')
            with self.assertRaises(ValueError):
                cached_measurements(path,manifest,['baseline'],'none','full_photo',21)

    def test_reused_copies_are_not_independent_latency_samples(self):
        row = dict(repeat=0,verified=False,label='unreviewed',confirmed_identifier=None,
                   status='sin_texto',elapsed_s=2.,ocr_calls=1,measurement_id='one-real-inference')
        result = metrics([row.copy(),row.copy()])
        self.assertEqual(result['images'],2)
        self.assertEqual(result['latency_s']['actual_measurements'],1)
        self.assertIsNone(result['exact_code_accuracy'])


if __name__=='__main__':
    unittest.main()
