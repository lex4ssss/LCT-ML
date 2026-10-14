import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_array_equal
import confirm_v18 as confirm


class ConfirmationTests(unittest.TestCase):
    def test_twenty_explicit_cache_path_integrity_smokes(self):
        with tempfile.TemporaryDirectory() as temporary:
            for seed in range(20):
                path = Path(temporary) / f'features-{seed}.joblib'
                path.write_bytes(bytes([seed, 1, 2, 3]))
                digest = confirm.adaptive.file_digest(path)
                self.assertEqual(confirm.verified_feature_cache(str(path), digest), path)
                path.write_bytes(bytes([seed, 1, 2, 4]))
                with self.assertRaisesRegex(ValueError, 'feature cache'):
                    confirm.verified_feature_cache(path, digest)

    def test_tail_mask_independent_calendar_oracle_twenty_scenarios(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            days = np.arange(np.datetime64('2026-01-01'), np.datetime64('2026-02-01'), dtype='datetime64[D]').astype('datetime64[us]')
            rng.shuffle(days)
            present = rng.random(len(days)) > 0.2
            present[0] = True
            for hours in (72, 168):
                expected = [seen and moment <= max(days[present]) - np.timedelta64(hours-24, 'h') for moment, seen in zip(days, present)]
                assert_array_equal(confirm.test_mask(days, present, hours), expected)
        for value in ('2025-12-31', 'NaT'):
            with self.assertRaises(ValueError):
                confirm.test_mask(np.array([value], dtype='datetime64[us]'), [True], 168)

    def test_reference_mismatch_twenty_gates(self):
        actual = dict(rows=100, base_rate=0.2, precision=0.8, recall=0.6, f1=0.6857, ap=0.84, threshold=0.56)
        names = dict(rows='rows', base_rate='prevalence', precision='precision', recall='recall', f1='f1', ap='average_precision', threshold='threshold')
        expected = {names[key]: value for key, value in actual.items()}
        confirm.verify_reference(actual, expected)
        for seed in range(20):
            changed = actual.copy()
            key = list(names)[seed % len(names)]
            changed[key] += 1 if key == 'rows' else 1e-4
            with self.assertRaisesRegex(ValueError, 'reference test'):
                confirm.verify_reference(changed, expected)

    def test_twenty_noninferiority_gate_smokes(self):
        reference = dict(precision=0.8, recall=0.6, f1=0.6857, ap=0.84, qualifies=True)
        self.assertTrue(confirm.accepted(reference.copy(), reference))
        for seed in range(20):
            candidate = reference.copy()
            key = ['precision', 'recall', 'f1', 'ap', 'qualifies'][seed % 5]
            candidate[key] = False if key == 'qualifies' else candidate[key] - 1e-6
            self.assertFalse(confirm.accepted(candidate, reference))

    def test_frozen_selection_twenty_mutations(self):
        reference = dict(ap=0.7, precision_at_recall_0_5=0.8, qualifies=True, f1=0.6, margin=0.1)
        candidate = dict(ap=0.72, precision_at_recall_0_5=0.81, qualifies=True, f1=0.61, margin=0.11, threshold=confirm.THRESHOLD)
        selection = dict(complete=True, test_evaluation=False, targets={confirm.TARGET: dict(winner=confirm.VARIANT,
            results={'R': dict(validation=reference, features=77, model_count=1),
                     confirm.VARIANT: dict(validation=candidate, features=224, model_count=3)})})
        self.assertEqual(confirm.freeze(selection, confirm.MODEL_SHA)['validation']['threshold'], confirm.THRESHOLD)
        for seed in range(20):
            changed = copy.deepcopy(selection)
            digest = confirm.MODEL_SHA
            if seed % 5 == 0:
                changed['targets'][confirm.TARGET]['results'][confirm.VARIANT]['validation']['threshold'] += 1e-6
            elif seed % 5 == 1:
                digest = 'changed'
            elif seed % 5 == 2:
                changed['targets'][confirm.TARGET]['winner'] = 'R'
            elif seed % 5 == 3:
                changed['test_evaluation'] = True
            else:
                changed['targets'][confirm.TARGET]['results'][confirm.VARIANT]['validation']['ap'] = 0.705
            with self.assertRaises(ValueError):
                confirm.freeze(changed, digest)

    def test_completed_confirmation_is_immutable_twenty_scenarios(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path, scores = output / 'confirmation.json', output / 'candidate-scores.npy'
            self.assertIsNone(confirm.saved_confirmation(path, {'source': 'same'}))
            for seed in range(20):
                np.save(scores, [0.1, 0.2, 0.3])
                report = dict(fingerprint={'source': 'same'}, complete=True, accepted=False,
                              score_digests={scores.name: confirm.adaptive.file_digest(scores)})
                path.write_text(json.dumps(report))
                self.assertEqual(confirm.saved_confirmation(path, report['fingerprint']), report)
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    confirm.saved_confirmation(path, {'source': str(seed)})
                np.save(scores, [0.1, 0.2, 0.4 + seed / 100])
                with self.assertRaisesRegex(ValueError, 'scores changed'):
                    confirm.saved_confirmation(path, report['fingerprint'])


if __name__ == '__main__':
    unittest.main()
