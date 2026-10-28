import unittest
from types import SimpleNamespace

from modulos.excel_manifest import RegistroEsperado
from modulos.identifier_policy import decide_arrival, normalize_identifier


def reading(text, status='reconocido', confidence=.97, mode='seal_roi'):
    return SimpleNamespace(confirmed_identifier=text, status=status, input_mode=mode,
                           candidates=[SimpleNamespace(text=text, confidence_min=confidence,
                                                       observations=2, syntax_compatible=True,warnings=[])])


class IdentifierPolicyTests(unittest.TestCase):
    def test_full_match_only(self):
        records = [RegistroEsperado(precinto_esperado='B 0097312')]
        for value in ('0097312', 'XB 0097312', 'B 0097312Z', 'B 009731', 'B 00973120', 'B OO97312'):
            self.assertFalse(decide_arrival(records, reading(value)).accepted, value)
        self.assertTrue(decide_arrival(records, reading('b  0097312')).accepted)

    def test_ambiguity_confidence_and_dynamic_region_do_not_confirm(self):
        records = [RegistroEsperado(precinto_esperado='B-0097312')]
        for data in (reading('B-0097312', status='ambiguo'), reading('B-0097312', confidence=.7),
                     reading('B-0097312', mode='full_photo')):
            self.assertFalse(decide_arrival(records, data).accepted)

    def test_no_silent_downgrade_and_no_mutation_before_audit(self):
        records = [RegistroEsperado(precinto_esperado='B-0097312', estado='Llegado')]
        self.assertFalse(decide_arrival(records, reading('B-009731')).accepted)
        self.assertEqual(records[0].estado, 'Llegado')
        self.assertTrue(decide_arrival(records, reading('B-0097312')).accepted)
        self.assertEqual(records[0].estado, 'Llegado')

    def test_duplicate_expected_identifier_is_ambiguous(self):
        records = [RegistroEsperado(precinto_esperado='B7'), RegistroEsperado(precinto_esperado='B7')]
        self.assertFalse(decide_arrival(records, reading('B7')).accepted)

    def test_no_identity_changing_normalization(self):
        for value in ('AB/12', 'A\nB12', 'ß12', 'AB_12'):
            self.assertEqual(normalize_identifier(value), '')
        self.assertNotEqual(normalize_identifier('AB-12'), normalize_identifier('AB12'))
        self.assertNotEqual(normalize_identifier('A B12'), normalize_identifier('AB12'))

    def test_nonfinite_confidence_and_single_observation_never_confirm(self):
        records = [RegistroEsperado(precinto_esperado='B-0097312')]
        for confidence in (float('nan'),float('inf'),-1.,1.01):
            self.assertFalse(decide_arrival(records,reading('B-0097312',confidence=confidence)).accepted)
        data = reading('B-0097312')
        data.candidates[0].observations = 1
        self.assertFalse(decide_arrival(records,data).accepted)


if __name__ == '__main__':
    unittest.main()
