import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import joblib
import numpy as np
import run_v3
from test_run_v2 import write_inputs

HERE = Path(__file__).resolve().parent
HOUR = np.timedelta64(3600 * 10**6, 'us')


def grid(start='2023-03-01', end='2026-02-01', seed=0):
    rng = np.random.default_rng(seed)
    times = np.arange(np.datetime64(start, 'us'), np.datetime64(end, 'us'), 3 * HOUR)
    return dict(datetime=times, temperature=rng.normal(2, 8, len(times)).round(1), pressure=rng.normal(747, 7, len(times)).round(1), wind_speed=rng.uniform(0, 6, len(times)).round(1))


def expected(weather, as_of):
    times, temperature = weather['datetime'], weather['temperature']
    cutoff = np.datetime64(as_of, 'us') - 3 * HOUR
    day = (times > cutoff - 24 * HOUR) & (times <= cutoff)
    three = temperature[(times > cutoff - 72 * HOUR) & (times <= cutoff)]
    crossings = sum(1 for a, b in zip(three, three[1:]) if (a <= 0) != (b <= 0))
    pressure_now = weather['pressure'][times == cutoff][0]
    pressure_before = weather['pressure'][times == cutoff - 24 * HOUR][0]
    return [temperature[day].mean(), temperature[day].min(), temperature[day].max(), crossings, pressure_now, pressure_now - pressure_before, weather['wind_speed'][day].max()]


def write_weather(path, weather):
    lines = ['datetime,weather_condition,temperature,precipitation_mm,pressure,wind_speed,wind_direction_deg,cloud_cover_percent']
    for i, moment in enumerate(weather['datetime']):
        lines.append(f"{str(moment)[:19].replace('T', ' ')},облачно,{weather['temperature'][i]},0,{weather['pressure'][i]},{weather['wind_speed'][i]},180,50")
    path.write_text('\n'.join(lines) + '\n')


def run(*args):
    return subprocess.run([sys.executable, str(HERE / 'run_v3.py'), *map(str, args)], capture_output=True, text=True, cwd=HERE, timeout=900)


class WeatherFeatureTests(unittest.TestCase):
    def test_values_match_direct_computation(self):
        weather = grid()
        for as_of in ('2024-01-15', '2024-07-01', '2025-03-30', '2025-11-02'):
            np.testing.assert_allclose(run_v3.features_at(weather, np.datetime64(as_of, 'us')), expected(weather, as_of))

    def test_readings_after_cutoff_do_not_change_features(self):
        weather, as_of = grid(), np.datetime64('2024-05-10', 'us')
        before = run_v3.features_at(weather, as_of)
        future = weather['datetime'] > as_of - 3 * HOUR
        changed = {key: value.copy() for key, value in weather.items()}
        for key in ('temperature', 'pressure', 'wind_speed'):
            changed[key][future] = 99.0
        np.testing.assert_array_equal(run_v3.features_at(changed, as_of), before)

    def test_reading_at_cutoff_is_used(self):
        weather, as_of = grid(), np.datetime64('2024-05-10', 'us')
        changed = {key: value.copy() for key, value in weather.items()}
        changed['temperature'][changed['datetime'] == as_of - 3 * HOUR] = 60.0
        self.assertEqual(run_v3.features_at(changed, as_of)[2], 60.0)
        changed['temperature'][changed['datetime'] == as_of] = 90.0
        self.assertEqual(run_v3.features_at(changed, as_of)[2], 60.0)

    def test_gap_gives_nan(self):
        weather, as_of = grid(), np.datetime64('2024-05-10', 'us')
        keep = weather['datetime'] != as_of - 15 * HOUR
        gapped = {key: value[keep] for key, value in weather.items()}
        values = run_v3.features_at(gapped, as_of)
        self.assertTrue(np.isnan(values[0]) and np.isnan(values[3]) and np.isnan(values[6]))
        self.assertFalse(np.isnan(values[5]))
        keep = weather['datetime'] != as_of - 27 * HOUR
        values = run_v3.features_at({key: value[keep] for key, value in weather.items()}, as_of)
        self.assertTrue(np.isnan(values[5]) and np.isnan(values[3]))
        self.assertFalse(np.isnan(values[0]))

    def test_matrix_maps_each_row_to_its_day(self):
        weather = grid()
        as_of = np.array(['2024-02-02', '2024-02-01', '2024-02-02'], dtype='datetime64[us]')
        matrix = run_v3.weather_matrix(weather, as_of)
        np.testing.assert_array_equal(matrix[0], matrix[2])
        np.testing.assert_allclose(matrix[1], expected(weather, '2024-02-01'))


class RunV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        _, cls.examples, cls.episodes = write_inputs(cls.root, 5)
        cls.weather = cls.root / 'weather.csv'
        write_weather(cls.weather, grid())
        cls.baseline = cls.root / 'baseline.json'
        digest = hashlib.file_digest(cls.examples.open('rb'), 'sha256').hexdigest()
        cls.baseline.write_text(json.dumps(dict(input_sha256=digest, candidates=dict(hgb=dict(marker=1)))))
        result = run(cls.examples, cls.episodes, cls.weather, cls.baseline, cls.root / 'out')
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.report = json.loads((cls.root / 'out' / 'report.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_report_structure_and_flags(self):
        self.assertEqual(self.report['flags']['test_evaluation'], False)
        self.assertEqual(self.report['flags']['precipitation'], False)
        self.assertEqual(self.report['baseline']['hgb'], dict(marker=1))
        self.assertEqual(self.report['rows_missing_weather'], 0)
        hgb = self.report['candidates']['hgb_weather']
        self.assertEqual(set(hgb), {'threshold_A', 'selection_2024', 'frozen', 'retrained', 'coverage', 'calibration'})
        self.assertTrue((self.root / 'out' / 'hgb_weather.joblib').exists())

    def test_model_sees_weather_columns(self):
        model = joblib.load(self.root / 'out' / 'hgb_weather.joblib')
        self.assertEqual(model.n_features_in_, len(run_v3.ex.V2_FEATURES) + 2 + len(run_v3.WEATHER_FEATURES))

    def test_existing_output_refused(self):
        result = run(self.examples, self.episodes, self.weather, self.baseline, self.root / 'out')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FileExistsError', result.stderr)

    def test_baseline_with_other_input_refused(self):
        other = self.root / 'other.json'
        other.write_text(json.dumps(dict(input_sha256='0' * 64, candidates=dict(hgb={}))))
        result = run(self.examples, self.episodes, self.weather, other, self.root / 'out-other')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('differ from the baseline', result.stderr)
        self.assertFalse((self.root / 'out-other').exists())


if __name__ == '__main__':
    unittest.main()
