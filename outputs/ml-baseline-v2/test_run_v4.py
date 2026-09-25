import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import duckdb
import joblib
import numpy as np
import run_v4
from test_run_v2 import write_inputs

HERE = Path(__file__).resolve().parent
HOUR = np.timedelta64(3600 * 10**6, 'us')


def hourly(start='2023-03-01', end='2026-02-01', seed=0):
    rng = np.random.default_rng(seed)
    times = np.arange(np.datetime64(start, 'us'), np.datetime64(end, 'us'), HOUR)
    rain = np.where(rng.random(len(times)) < 0.1, rng.exponential(1.0, len(times)), 0.0).round(1)
    return dict(datetime=times, temperature=rng.normal(2, 8, len(times)).round(1), pressure=rng.normal(747, 7, len(times)).round(1),
                wind_speed=rng.uniform(0, 6, len(times)).round(2), precipitation=rain)


def expected(weather, as_of):
    times, temperature = weather['datetime'], weather['temperature']
    cutoff = np.datetime64(as_of, 'us') - 3 * HOUR
    day = (times > cutoff - 24 * HOUR) & (times <= cutoff)
    three_mask = (times > cutoff - 72 * HOUR) & (times <= cutoff)
    three = temperature[three_mask]
    crossings = sum(1 for a, b in zip(three, three[1:]) if (a <= 0) != (b <= 0))
    now = weather['pressure'][times == cutoff][0]
    before = weather['pressure'][times == cutoff - 24 * HOUR][0]
    return [temperature[day].mean(), temperature[day].min(), temperature[day].max(), crossings, now, now - before, weather['wind_speed'][day].max(),
            weather['precipitation'][day].sum(), weather['precipitation'][three_mask].sum()]


def write_weather(path, weather):
    lines = ['time,temperature_2m,surface_pressure,precipitation,wind_speed_10m,cloud_cover']
    for i, moment in enumerate(weather['datetime']):
        lines.append(f"{str(moment)[:16]},{weather['temperature'][i]},{weather['pressure'][i] / run_v4.MMHG_PER_HPA},{weather['precipitation'][i]},{weather['wind_speed'][i]},50")
    path.write_text('\n'.join(lines) + '\n')


def run(*args):
    return subprocess.run([sys.executable, str(HERE / 'run_v4.py'), *map(str, args)], capture_output=True, text=True, cwd=HERE, timeout=900)


class HourlyFeatureTests(unittest.TestCase):
    def test_values_match_direct_computation(self):
        weather = hourly()
        for as_of in ('2024-01-15', '2024-07-01', '2025-03-30', '2025-11-02'):
            np.testing.assert_allclose(run_v4.features_at(weather, np.datetime64(as_of, 'us')), expected(weather, as_of))

    def test_readings_after_cutoff_do_not_change_features(self):
        weather, as_of = hourly(), np.datetime64('2024-05-10', 'us')
        before = run_v4.features_at(weather, as_of)
        future = weather['datetime'] > as_of - 3 * HOUR
        changed = {key: value.copy() for key, value in weather.items()}
        for key in ('temperature', 'pressure', 'wind_speed', 'precipitation'):
            changed[key][future] = 99.0
        np.testing.assert_array_equal(run_v4.features_at(changed, as_of), before)

    def test_precipitation_window_edges(self):
        weather, as_of = hourly(), np.datetime64('2024-05-10', 'us')
        cutoff = as_of - 3 * HOUR
        base = run_v4.features_at(weather, as_of)
        for moment, day_delta, three_delta in ((cutoff, 10, 10), (cutoff - 23 * HOUR, 10, 10), (cutoff - 24 * HOUR, 0, 10), (cutoff - 71 * HOUR, 0, 10), (cutoff - 72 * HOUR, 0, 0)):
            changed = {key: value.copy() for key, value in weather.items()}
            changed['precipitation'][changed['datetime'] == moment] += 10
            values = run_v4.features_at(changed, as_of)
            self.assertAlmostEqual(values[7] - base[7], day_delta)
            self.assertAlmostEqual(values[8] - base[8], three_delta)

    def test_gap_gives_nan(self):
        weather, as_of = hourly(), np.datetime64('2024-05-10', 'us')
        keep = weather['datetime'] != as_of - 40 * HOUR
        values = run_v4.features_at({key: value[keep] for key, value in weather.items()}, as_of)
        self.assertTrue(np.isnan(values[8]) and np.isnan(values[3]))
        self.assertFalse(np.isnan(values[7]) or np.isnan(values[0]))


class RunV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        _, cls.examples, cls.episodes = write_inputs(cls.root, 5)
        cls.weather_data = hourly()
        cls.weather = cls.root / 'weather.csv'
        write_weather(cls.weather, cls.weather_data)
        cls.baseline = cls.root / 'baseline.json'
        with cls.examples.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        cls.baseline.write_text(json.dumps(dict(input_sha256=digest, candidates=dict(hgb=dict(marker=1)))))
        result = run(cls.examples, cls.episodes, cls.weather, cls.baseline, cls.root / 'out')
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.report = json.loads((cls.root / 'out' / 'report.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_report_structure_and_flags(self):
        flags = self.report['flags']
        self.assertEqual((flags['test_evaluation'], flags['precipitation'], flags['weather_source']), (False, True, 'open-meteo-era5-hourly'))
        self.assertEqual(self.report['weather_features'], run_v4.WEATHER_FEATURES)
        self.assertEqual(self.report['baseline']['hgb'], dict(marker=1))
        self.assertEqual(self.report['rows_missing_weather'], 0)
        self.assertEqual(set(self.report['candidates']), {'hgb_weather_openmeteo'})

    def test_reader_converts_pressure_and_maps_columns(self):
        with duckdb.connect() as connection:
            weather = run_v4.read_weather(connection, self.weather)
        np.testing.assert_allclose(weather['pressure'], self.weather_data['pressure'], atol=1e-6)
        np.testing.assert_array_equal(weather['precipitation'], self.weather_data['precipitation'])
        np.testing.assert_array_equal(weather['datetime'], self.weather_data['datetime'])

    def test_model_sees_all_weather_columns(self):
        model = joblib.load(self.root / 'out' / 'hgb_weather_openmeteo.joblib')
        self.assertEqual(model.n_features_in_, len(run_v4.v3.ex.V2_FEATURES) + 2 + len(run_v4.WEATHER_FEATURES))


if __name__ == '__main__':
    unittest.main()
