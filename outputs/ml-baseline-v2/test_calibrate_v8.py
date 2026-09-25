import unittest
import numpy as np
import calibrate_v8 as c8


class ReliabilityTests(unittest.TestCase):
    def test_bins_match_brute_force(self):
        rng = np.random.default_rng(2)
        probabilities = np.r_[rng.random(500), 0.0, 0.1, 0.9999, 1.0]
        labels = (rng.random(len(probabilities)) < probabilities).astype(np.int64)
        table = c8.reliability(labels, probabilities)
        self.assertEqual(sum(row['rows'] for row in table), len(probabilities))
        for row in table:
            inside = (probabilities >= row['low']) & ((probabilities < row['high']) | (row['high'] == 1.0))
            self.assertEqual(row['rows'], inside.sum())
            self.assertAlmostEqual(row['observed_rate'], labels[inside].mean())
            self.assertAlmostEqual(row['mean_probability'], probabilities[inside].mean())


if __name__ == '__main__':
    unittest.main()
