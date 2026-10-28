"""Regresiones de lógica con respuestas controladas; no precisión fotográfica."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from modulos import ocr_engine
from modulos.ocr_analysis import AnalysisConfig, Detection, analyze, candidates_from, _original_box
from ocr_tools.audit_ocr import load_baseline
from ocr_tools.evaluate_ocr import distance, inventory, metrics, validate_manifest


BOX = [[10, 10], [90, 10], [90, 30], [10, 30]]
BOX2 = [[110, 10], [190, 10], [190, 30], [110, 30]]


def response(text="AB7-X90", confidence=0.97, box=None):
    return [[[(box or BOX), (text, confidence)]]]


class LegacyRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline, _ = load_baseline()

    def test_every_preprocessed_pixel_is_preserved(self):
        for shape in ((32, 64, 3), (101, 137, 3), (240, 320, 3)):
            image = np.random.default_rng(52).integers(0, 256, shape, dtype=np.uint8)
            before = self.baseline.preprocesar(image)
            after = ocr_engine.preprocesar(image)
            self.assertEqual([n for n, _ in before], [n for n, _ in after])
            for (_, a), (_, b) in zip(before, after):
                np.testing.assert_array_equal(a, b)

    def test_sequence_and_selected_result_match_for_every_exit_position(self):
        image = np.random.default_rng(3).integers(0, 256, (64, 128, 3), dtype=np.uint8)
        for exit_call in range(1, 23):  # 22 => agotamiento sin salida temprana
            observations = []
            results = []
            for module in (self.baseline, ocr_engine):
                calls = []

                def infer(array):
                    calls.append(array.copy())
                    return ("AB7-X90", 0.97) if len(calls) == exit_call else ("AB7-X9O", 0.7)

                with patch.object(module, "_ocr_imagen", side_effect=infer):
                    results.append(module.mejor_ocr(image))
                observations.append(calls)
            self.assertEqual(results[0], results[1])
            self.assertEqual(len(observations[0]), min(exit_call, 21))
            self.assertEqual(len(observations[0]), len(observations[1]))
            for a, b in zip(*observations):
                np.testing.assert_array_equal(a, b)

    def test_only_needed_preprocessing_runs(self):
        image = np.zeros((64, 128, 3), np.uint8)
        with patch.object(ocr_engine, "_ocr_imagen", side_effect=[("AB7-X90", 0.7), ("AB7-X90", 0.99)]), \
                patch.object(ocr_engine.cv2, "bilateralFilter", side_effect=AssertionError("No debe ejecutarse")), \
                patch.object(ocr_engine.cv2, "createCLAHE", side_effect=AssertionError("No debe ejecutarse")):
            ocr_engine.mejor_ocr(image)

    def test_shared_predictor_is_serialized(self):
        active = 0
        maximum = 0

        class Predictor:
            def ocr(self, *_args, **_kwargs):
                nonlocal active, maximum
                active += 1
                maximum = max(maximum, active)
                time.sleep(0.01)
                active -= 1
                return [None]

        with patch.object(ocr_engine, "get_ocr", return_value=Predictor()):
            threads = [threading.Thread(target=ocr_engine.ejecutar_ocr, args=(None,)) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(maximum, 1)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(4).integers(0, 256, (100, 220, 3), np.uint8)

    def test_consistent_roi_can_return_unknown_code_without_dictionary(self):
        result = analyze(self.image, input_mode="seal_roi", infer=lambda _: response(), config=AnalysisConfig(max_calls=2))
        self.assertEqual(result.confirmed_identifier, "AB7-X90")
        self.assertEqual(result.ocr_calls, 2)
        json.dumps(result.to_dict())

    def test_full_photo_text_is_not_semantic_seal_detection(self):
        result = analyze(self.image, infer=lambda _: response(), config=AnalysisConfig(max_calls=2))
        self.assertIsNone(result.confirmed_identifier)
        self.assertEqual(result.status, "texto_no_identificado")
        self.assertEqual(result.suggested_text, "AB7-X90")

    def test_conflicting_zero_and_letter_o_are_not_corrected(self):
        values = iter([response("AB7-X90"), response("AB7-X9O"), response("AB7-X90")])
        result = analyze(self.image, input_mode="seal_roi", infer=lambda _: next(values),
                         config=AnalysisConfig(max_calls=3))
        self.assertIsNone(result.confirmed_identifier)
        self.assertEqual(result.status, "ambiguo")
        self.assertEqual({c.text for c in result.candidates}, {"AB7-X90", "AB7-X9O"})

    def test_low_confidence_is_never_confirmed(self):
        result = analyze(self.image, input_mode="seal_roi", infer=lambda _: response(confidence=0.7),
                         config=AnalysisConfig(max_calls=3))
        self.assertEqual(result.status, "ambiguo")
        self.assertIsNone(result.confirmed_identifier)

    def test_distant_boxes_are_never_joined(self):
        raw = [[response("AB7")[0][0], response("X90", box=BOX2)[0][0]]]
        result = analyze(self.image, input_mode="seal_roi", infer=lambda _: raw,
                         config=AnalysisConfig(max_calls=3))
        self.assertIsNone(result.confirmed_identifier)
        self.assertEqual({c.text for c in result.candidates}, {"AB7", "X90"})

    def test_same_text_in_different_regions_is_not_consensus(self):
        values = iter([response(), response(box=BOX2)])
        result = analyze(self.image, input_mode="seal_roi", infer=lambda _: next(values),
                         config=AnalysisConfig(max_calls=2))
        self.assertIsNone(result.confirmed_identifier)
        self.assertEqual(len(result.candidates), 2)
        self.assertTrue(all(c.observations == 1 for c in result.candidates))

    def test_only_hyphens_or_noise_cannot_confirm(self):
        for text in ("------", "AB/123"):
            result = analyze(self.image, input_mode="seal_roi", infer=lambda _: response(text),
                             config=AnalysisConfig(max_calls=2))
            self.assertIsNone(result.confirmed_identifier)
            self.assertEqual(result.candidates[0].detections[0].raw_text, text)

    def test_unusual_length_is_warning_not_hard_rejection(self):
        text = "AB1234567890123456"
        result = analyze(self.image, input_mode="seal_roi", infer=lambda _: response(text), config=AnalysisConfig(max_calls=2))
        self.assertEqual(result.confirmed_identifier, text)
        self.assertIn("longitud_inusual_no_descartada", result.candidates[0].warnings)

    def test_long_text_has_no_length_reward(self):
        ds = [Detection(t, t, .97, BOX, "v", 0, str(i))
              for i, t in enumerate(("AB123", "AB1234567890123456"))]
        self.assertEqual(candidates_from(ds, AnalysisConfig())[0].text, "AB123")

    def test_identical_arrays_are_not_independent_observations(self):
        image = np.zeros((100, 100, 3), np.uint8)
        with patch("modulos.ocr_analysis._stages", return_value=iter([
                ("a", 0, image), ("b", 0, image.copy())])):
            result = analyze(image, input_mode="seal_roi", infer=lambda _: response())
        self.assertEqual(result.ocr_calls, 1)
        self.assertIsNone(result.confirmed_identifier)
        self.assertEqual(result.stages[1]["skipped"], "identical_pixels")

    def test_empty_does_not_claim_absence(self):
        result = analyze(self.image, infer=lambda _: [None], config=AnalysisConfig(max_calls=2))
        self.assertEqual(result.status, "sin_texto")
        self.assertIsNone(result.confirmed_identifier)
        self.assertIn("NO demuestra ausencia", result.reason)

    def test_technical_error_preserves_partial_evidence(self):
        values = iter([response(), RuntimeError("fallo controlado")])

        def infer(_):
            value = next(values)
            if isinstance(value, Exception):
                raise value
            return value

        result = analyze(self.image, infer=infer)
        self.assertEqual(result.status, "error_tecnico")
        self.assertIn("fallo controlado", result.error)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.ocr_calls, 2)

    def test_invalid_confidence_is_technical_error(self):
        result = analyze(self.image, infer=lambda _: response(confidence=float("nan")))
        self.assertEqual(result.status, "error_tecnico")

    def test_invalid_input_never_calls_model(self):
        for image in (None, np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2)), np.zeros((2, 2, 4), np.uint8)):
            result = analyze(image, infer=lambda _: self.fail("Modelo llamado con imagen inválida"))
            self.assertEqual(result.status, "imagen_invalida")
            self.assertEqual(result.ocr_calls, 0)

    def test_inverse_rotation_coordinates(self):
        # Coordenada original (20,30), imagen ancho 220, alto 100.
        for rotation, point in ((-90, [30, 199]), (90, [69, 20])):
            mapped = _original_box([point] * 4, rotation, self.image.shape)
            self.assertEqual(mapped, [[20.0, 30.0]] * 4)


class EvaluationTests(unittest.TestCase):
    def row(self, **kwargs):
        return {"repeat": 0, "verified": True, "label": "readable", "reference_code": "AB-01",
                "confirmed_identifier": None, "suggested_text": None, "elapsed_s": 1,
                "ocr_calls": 2, "status": "ambiguo", **kwargs}

    def test_rejecting_everything_does_not_inflate_accuracy(self):
        result = metrics([self.row()])
        self.assertEqual(result["exact_code_accuracy"], 0)
        self.assertEqual(result["coverage_verified"], 0)
        self.assertEqual(result["cer_confirmed_including_rejections"], 1)
        self.assertIsNone(result["accuracy_among_accepted"])

    def test_negatives_unreadable_and_errors_are_counted(self):
        rows = [self.row(label="no_seal", reference_code=None, confirmed_identifier="XX"),
                self.row(label="unreadable", reference_code=None),
                self.row(status="error_tecnico")]
        result = metrics(rows)
        self.assertEqual(result["images"], 3)
        self.assertEqual(result["false_accepts"], 1)
        self.assertEqual(result["negative_nonacceptance"], 0)
        self.assertEqual(result["technical_errors"], 1)

    def test_unverified_references_do_not_become_accuracy(self):
        result = metrics([self.row(verified=False)])
        self.assertEqual(result["unverified_images"], 1)
        self.assertIsNone(result["exact_code_accuracy"])

    def test_repeats_do_not_inflate_number_of_images(self):
        rows = [self.row(repeat=r) for r in range(3)]
        result = metrics(rows)
        self.assertEqual(result["images"], 1)
        self.assertEqual(result["readable_images"], 1)

    def test_edit_distance_preserves_zeros_and_hyphens(self):
        self.assertEqual(distance("001-AB", "1AB"), 3)
        self.assertEqual(distance("AB0", "ABO"), 1)

    def test_inventory_never_guesses_labels_and_catches_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            # No imagen real: archivo inválido deliberado que NO debe desaparecer.
            (path / "a.jpg").write_bytes(b"deliberately invalid fixture")
            (path / "b.jpg").write_bytes(b"deliberately invalid fixture")
            manifest = inventory(path, expected=2)
            validate_manifest(manifest)
            self.assertTrue(all(r["reference_code"] is None and not r["verified"] for r in manifest["records"]))
            self.assertEqual(manifest["records"][0]["group_id"], manifest["records"][1]["group_id"])
            manifest["records"][0]["split"] = "development"
            manifest["records"][1]["split"] = "evaluation"
            with self.assertRaisesRegex(ValueError, "Filtración"):
                validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
