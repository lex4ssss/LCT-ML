import unittest

import numpy as np
import pandas as pd

import pumps_data
import pumps_v5_features as v5
from test_pumps import make_frame


def naive(values, profile, window):
    padded = np.vstack([np.tile(profile, (300, 1)), values])
    rows = padded[-window:]
    k = np.arange(window)
    slope = np.array([np.polyfit(k, rows[:, i], 1)[0] for i in range(rows.shape[1])])
    return rows.mean(axis=0), slope


class FeatureV5Test(unittest.TestCase):
    def setUp(self):
        frame = make_frame({'NPV_2_1': [96, 12, 12], 'NPV_2_2': [96]}, normals_by_pump=6)
        self.reconstructed = pumps_data.reconstruct(frame)
        self.stats = v5.normal_stats([self.reconstructed])
        self.features, self.labels, self.meta = v5.prediction_rows(self.reconstructed, self.stats, [30, 60])
        self.ordered = self.reconstructed.sort_values(['trajectory', 'step'], kind='stable')

    def history(self, trajectory, step):
        rows = self.ordered[(self.ordered['trajectory'] == trajectory) & (self.ordered['step'] <= step)]
        return v5.signals(rows[pumps_data.SENSORS].to_numpy(dtype=np.float64))

    def row(self, trajectory, step):
        return np.flatnonzero((self.meta['trajectory'].to_numpy() == trajectory) & (self.meta['step'].to_numpy() == step))[0]

    def test_rolling_means_and_slopes_match_padded_history(self):
        trajectory = int(self.meta.loc[self.meta['pump_id'] == 'NPV_2_2', 'trajectory'].max())
        profile = self.stats['median']['NPV_2_2']
        for step in (0, 5, 11, 12, 40, 94):
            history, i = self.history(trajectory, step), self.row(trajectory, step)
            for window in (12, 36, 96, 288):
                mean, slope = naive(history, profile, window)
                np.testing.assert_allclose([self.features[f'{n}_mean_{window}'].iloc[i] for n in v5.SIGNALS], mean, rtol=1e-5, atol=1e-5)
                if window in v5.SLOPE_WINDOWS:
                    np.testing.assert_allclose([self.features[f'{n}_slope_{window}'].iloc[i] for n in v5.SIGNALS], slope, rtol=1e-4, atol=1e-5)

    def test_deviation_uses_12_row_mean(self):
        trajectory = int(self.meta.loc[self.meta['pump_id'] == 'NPV_2_2', 'trajectory'].max())
        i = self.row(trajectory, 40)
        mean, _ = naive(self.history(trajectory, 40), self.stats['median']['NPV_2_2'], 12)
        expected = (mean - self.stats['median']['NPV_2_2']) / self.stats['std']['NPV_2_2']
        np.testing.assert_allclose([self.features[f'{n}_deviation_12'].iloc[i] for n in v5.SIGNALS], expected, rtol=1e-4, atol=1e-4)

    def test_meta_and_labels_match_pumps_data(self):
        _, labels, meta = pumps_data.prediction_rows(self.reconstructed, [30, 60], feature_set='a3')
        np.testing.assert_array_equal(meta['trajectory'].to_numpy(), self.meta['trajectory'].to_numpy())
        np.testing.assert_array_equal(meta['step'].to_numpy(), self.meta['step'].to_numpy())
        for horizon in (30, 60):
            np.testing.assert_array_equal(labels[horizon], self.labels[horizon])
        self.assertTrue((self.meta.loc[self.meta['trajectory'] < 0, 'fault_type'] == 'normal').all())

    def test_normal_row_is_a_one_row_history(self):
        i = int(np.flatnonzero(self.meta['trajectory'].to_numpy() < 0)[3])
        pump = self.meta['pump_id'].iloc[i]
        value = v5.signals(self.features[pumps_data.SENSORS].iloc[[i]].to_numpy(dtype=np.float64))[0]
        mean, _ = naive(value[None, :], self.stats['median'][pump], 12)
        np.testing.assert_allclose([self.features[f'{n}_mean_12'].iloc[i] for n in v5.SIGNALS], mean, rtol=1e-5)

    def test_onset_age(self):
        deviation = np.zeros((8, 2))
        deviation[3, 1] = 5.0
        deviation[6, 0] = -9.0
        group = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        position = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        self.assertEqual(v5.onset_age(deviation, group, position).tolist(), [-1, -1, -1, 0, 1, -1, 0, 1])

    def test_future_rows_do_not_change_past_features(self):
        changed = self.reconstructed.copy()
        changed.loc[(changed['step'] >= 50) & (changed['trajectory'] >= 0), pumps_data.SENSORS] *= 3
        after, _, _ = v5.prediction_rows(changed, self.stats, [30])
        early = self.meta['step'].to_numpy() < 50
        pd.testing.assert_frame_equal(self.features[early], after[early])

    def test_no_forbidden_columns(self):
        forbidden = ('anomaly', 'timestamp', 'fault', 'status', 'trajectory', 'step', 'truth')
        self.assertFalse([c for c in self.features.columns if c.startswith(forbidden)])


class TreeTest(unittest.TestCase):
    def test_probability_mixes_router_and_experts(self):
        import pumps_v5

        class Router:
            classes_ = np.array(['normal', 'Несоосность', 'Засорение фильтра'])

            def predict_proba(self, features):
                return np.array([[0.5, 0.3, 0.2], [0.0, 1.0, 0.0]])

        class Expert:
            def predict_proba(self, features):
                return np.column_stack([np.full(len(features), 0.2), np.full(len(features), 0.8)])

        tree = pumps_v5.Tree(Router(), {'Несоосность': Expert(), 'Засорение фильтра': 1.0, 'Неизвестная неисправность': 0.0})
        np.testing.assert_allclose(tree.predict_proba(np.zeros((2, 3)))[:, 1], [0.3 * 0.8 + 0.2 * 1.0, 0.8])

    def test_single_class_expert_is_a_constant(self):
        import pumps_v5
        self.assertEqual(pumps_v5.fit_expert(np.zeros((3, 2)), np.array([1, 1, 1])), 1.0)


class EndToEndTest(unittest.TestCase):
    def test_pipeline_runs_and_pickles(self):
        import json
        import tempfile
        from pathlib import Path
        import pumps_v5
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            paths = []
            for seed in range(5):
                lengths = {'NPV_2_1': [12, 96, 12, 12], 'NPV_2_2': [96, 12, 12], 'NPV_2_3': [12, 12], 'NPV_2_4': [12, 96]}
                frame = make_frame(lengths, normals_by_pump=6, seed=seed)
                frame.loc[frame['tag'] >= 0, pumps_data.SENSORS] += (frame.loc[frame['tag'] >= 0, 'tag'] % 1000).to_numpy()[:, None] / 50
                paths.append(root / f'{seed}.csv')
                frame.to_csv(paths[-1], index=False)
            report = pumps_v5.run(paths[:3], paths[3], paths[4], Path(__file__).parent / 'run-004', root / 'out')
            self.assertTrue((root / 'out' / 'pumps_v5_models.joblib').exists())
            self.assertEqual(set(report['horizons']), {'30min', '1h'})
            json.dumps(report)


if __name__ == '__main__':
    unittest.main()
