import unittest

import numpy as np
import pandas as pd

import pumps_data
import pumps_v1


def make_frame(lengths_by_pump, normals_by_pump=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pump_id, lengths in lengths_by_pump.items():
        for step in range(max(lengths)):
            for order, length in enumerate(lengths):
                if step < length:
                    row = dict(timestamp=pumps_data.ORIGIN + pd.Timedelta(minutes=5 * step), pump_id=pump_id,
                               fault_type_name='Несоосность' if step == length - 1 else None,
                               tag=order * 1000 + step)
                    rows.append(row)
        for k in range(normals_by_pump):
            rows.append(dict(timestamp=pumps_data.ORIGIN + pd.Timedelta(minutes=5 * (300 + k)), pump_id=pump_id,
                             fault_type_name=None, tag=-1))
    frame = pd.DataFrame(rows)
    for name in pumps_data.SENSORS:
        frame[name] = rng.uniform(1.0, 2.0, len(frame))
    frame['fault_type_name'] = frame['fault_type_name'].astype('string')
    return frame.sort_values(['pump_id', 'timestamp'], kind='stable').reset_index(drop=True)


class ReconstructTest(unittest.TestCase):
    def test_rows_follow_their_generation_order(self):
        frame = make_frame({'NPV_2_1': [12, 96, 12, 96], 'NPV_2_2': [96, 12]})
        result = pumps_data.reconstruct(frame)
        in_trajectory = result[result['trajectory'] >= 0]
        self.assertEqual(in_trajectory.groupby('trajectory')['tag'].apply(lambda t: (t // 1000).nunique()).max(), 1)
        self.assertEqual(sorted(in_trajectory.groupby('trajectory').size()), [12, 12, 12, 96, 96, 96])
        self.assertTrue((result.loc[result['tag'] < 0, 'trajectory'] == -1).all())

    def test_normal_row_inside_window_is_rejected(self):
        frame = make_frame({'NPV_2_1': [12, 96]})
        extra = frame.iloc[[5]].assign(fault_type_name=pd.NA)
        frame = pd.concat([frame, extra]).sort_values(['pump_id', 'timestamp'], kind='stable').reset_index(drop=True)
        with self.assertRaises(pumps_data.NotReconstructable):
            pumps_data.reconstruct(frame)

    def test_missing_fault_label_is_rejected(self):
        frame = make_frame({'NPV_2_1': [12, 96]})
        frame.loc[frame['fault_type_name'].notna().idxmax(), 'fault_type_name'] = pd.NA
        with self.assertRaises(pumps_data.NotReconstructable):
            pumps_data.reconstruct(frame)

    def test_length_outside_allowed_is_rejected(self):
        frame = make_frame({'NPV_2_1': [13]})
        with self.assertRaises(pumps_data.NotReconstructable):
            pumps_data.reconstruct(frame)


class LabelTest(unittest.TestCase):
    def setUp(self):
        self.frame = pumps_data.reconstruct(make_frame({'NPV_2_1': [12, 96], 'NPV_2_2': [12]}))
        self.features, self.labels, self.meta = pumps_data.prediction_rows(self.frame, [30, 60, 480])

    def test_fault_rows_are_not_predicted(self):
        self.assertEqual(len(self.features), len(self.frame) - 3)

    def test_horizon_counts_per_trajectory(self):
        self.assertEqual(int(self.labels[30].sum()), 6 * 3)
        self.assertEqual(int(self.labels[60].sum()), 11 + 12 + 11)
        self.assertEqual(int(self.labels[480].sum()), 11 + 95 + 11)

    def test_normal_rows_are_negative(self):
        normal = self.meta['trajectory'].to_numpy() < 0
        self.assertEqual(normal.sum(), 4)
        for horizon in (30, 60, 480):
            self.assertEqual(int(self.labels[horizon][normal].sum()), 0)

    def test_fault_type_is_carried_to_every_row(self):
        in_trajectory = self.meta['trajectory'] >= 0
        self.assertTrue((self.meta.loc[in_trajectory, 'fault_type'] == 'Несоосность').all())
        self.assertTrue(self.meta.loc[~in_trajectory, 'fault_type'].isna().all())


class FeatureTest(unittest.TestCase):
    def test_change_uses_row_twelve_steps_earlier(self):
        frame = pumps_data.reconstruct(make_frame({'NPV_2_1': [96]}, normals_by_pump=0))
        features, _, _ = pumps_data.prediction_rows(frame, [30])
        ordered = frame.sort_values('step')['motor_current'].to_numpy()
        self.assertAlmostEqual(features['motor_current_change_12'][40], ordered[40] - ordered[28])
        self.assertAlmostEqual(features['motor_current_change_12'][5], ordered[5] - ordered[0])
        self.assertAlmostEqual(features['motor_current_mean_step_12'][5], (ordered[5] - ordered[0]) / 5)
        self.assertEqual(features['motor_current_change_12'][0], 0.0)

    def test_future_rows_do_not_change_past_features(self):
        frame = pumps_data.reconstruct(make_frame({'NPV_2_1': [96, 12]}))
        before, _, meta = pumps_data.prediction_rows(frame, [30])
        changed = frame.copy()
        late = changed['step'] >= 50
        changed.loc[late, pumps_data.SENSORS] *= 10
        after, _, _ = pumps_data.prediction_rows(changed, [30])
        early = (meta['trajectory'].to_numpy() >= 0) & (meta['step'].to_numpy() < 50)
        self.assertGreater(early.sum(), 50)
        pd.testing.assert_frame_equal(before[early], after[early])
        self.assertFalse(before[~early & (meta['trajectory'].to_numpy() >= 0)].equals(after[~early & (meta['trajectory'].to_numpy() >= 0)]))

    def test_normal_rows_have_zero_changes(self):
        frame = pumps_data.reconstruct(make_frame({'NPV_2_1': [12]}, normals_by_pump=3))
        features, _, meta = pumps_data.prediction_rows(frame, [30])
        normal = meta['trajectory'].to_numpy() < 0
        change_columns = [c for c in features.columns if '_change_' in c or '_mean_step_' in c]
        self.assertTrue((features.loc[normal, change_columns] == 0).all().all())

    def test_rule(self):
        frame = pd.DataFrame({name: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] for name in pumps_data.SENSORS})
        frame['pressure_diff_filter'] = 0.05
        frame.loc[1, 'vibration_motor_bearing_nondrive'] = 2.9
        frame.loc[2, 'motor_current'] = 83
        frame.loc[3, 'temperature_pump_bearing_nondrive'] = 81
        frame.loc[4, 'pressure_diff_filter'] = 0.15
        frame.loc[5, 'vibration_pump_bearing_drive'] = 2.8
        self.assertEqual(pumps_data.rule_alarm(frame).tolist(), [False, True, True, True, False, False])


class SplitTest(unittest.TestCase):
    def test_trajectories_stay_whole_and_follow_generation_order(self):
        frame = pumps_data.reconstruct(make_frame({'NPV_2_1': [12] * 10, 'NPV_2_2': [12] * 5}, normals_by_pump=10))
        _, _, meta = pumps_data.prediction_rows(frame, [30])
        part = pumps_data.split(meta)
        per_trajectory = pd.Series(part).groupby(meta['trajectory'].to_numpy()).nunique()
        self.assertTrue((per_trajectory.drop(-1) == 1).all())
        first = pd.Series(part).groupby(meta['trajectory'].to_numpy()).first().drop(-1)
        self.assertEqual(first.iloc[:10].tolist(), ['train'] * 6 + ['validation'] * 2 + ['test'] * 2)
        self.assertEqual(first.iloc[10:].tolist(), ['train'] * 3 + ['validation'] + ['test'])
        normal_parts = pd.Series(part[meta['trajectory'].to_numpy() < 0])
        self.assertEqual(normal_parts.value_counts().to_dict(), {'train': 12, 'validation': 4, 'test': 4})


class MetricTest(unittest.TestCase):
    def test_threshold_takes_the_widest_margin(self):
        labels = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        self.assertEqual(pumps_v1.choose_threshold(labels, scores), 0.4)
        self.assertAlmostEqual(pumps_v1.best_precision_at_recall(labels, scores), 1.0)

    def test_no_threshold_when_target_unreachable(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        self.assertIsNone(pumps_v1.choose_threshold(labels, scores))

    def test_scores_at(self):
        result = pumps_v1.scores_at(np.array([1, 1, 0, 0]), np.array([True, False, True, False]))
        self.assertEqual((result['precision'], result['recall'], result['base_rate']), (0.5, 0.5, 0.5))


if __name__ == '__main__':
    unittest.main()
