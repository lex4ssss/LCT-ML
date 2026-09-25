import json
from pathlib import Path
import sys
import duckdb
from feature_store import FeatureStore, InsufficientData

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ml-baseline-v2'))
import experiment_v2 as ex

STRATA = [('label IS NOT NULL', 120), ('label IS NOT NULL AND episode_starts_30d > 0', 100), ('label IS NOT NULL AND alarm_count_24h > 0', 60),
          ('label IS NOT NULL AND log_gap_hours_30d > 0', 60), ('exclusion_reason IS NOT NULL', 60), ("exclusion_reason = 'active_alarm'", 40)]
TOLERANCE = 1e-6


def sample(examples, seed):
    base = 'SELECT channel_id, as_of, ' + ', '.join(ex.V2_FEATURES) + ', exclusion_reason FROM read_parquet(?)'
    with duckdb.connect() as connection:
        return [row for where, size in STRATA
                for row in connection.execute(f'SELECT * FROM ({base} WHERE {where}) USING SAMPLE {size} ROWS (reservoir, {seed})', [str(examples)]).fetchall()]


def same(expected, served):
    if expected is None or served is None:
        return expected is None and served is None
    return abs(float(expected) - float(served)) < TOLERANCE


def main(prepared, config, examples, seed):
    store = FeatureStore(prepared, config)
    served, mismatches, outcomes = 0, {}, {}
    for channel, as_of, *values, exclusion in sample(examples, seed):
        try:
            _, features = store.features(channel, as_of)
        except InsufficientData as refusal:
            key = f'{exclusion}->{refusal.reason}'
            outcomes[key] = outcomes.get(key, 0) + 1
            continue
        served += 1
        key = f'{exclusion}->served'
        outcomes[key] = outcomes.get(key, 0) + 1
        for name, expected in zip(ex.V2_FEATURES, values):
            if not same(expected, features[name]):
                mismatches.setdefault(name, []).append(dict(channel_id=channel, as_of=str(as_of), training=expected, service=features[name]))
    print(json.dumps(dict(seed=seed, served=served, mismatched_features={name: len(rows) for name, rows in mismatches.items()},
                          first_mismatches={name: rows[:3] for name, rows in mismatches.items()}, outcomes=outcomes), indent=2, default=str))
    return 1 if mismatches else 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:4], int(sys.argv[4]) if len(sys.argv) > 4 else 7))
