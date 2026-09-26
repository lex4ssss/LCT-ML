import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import explore_v7 as e7
import explore_v8 as e8
import final_v2
import run_v2 as rv

BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def reliability(labels, probabilities):
    index = np.clip(np.digitize(probabilities, BINS[1:-1]), 0, len(BINS) - 2)
    return [dict(low=BINS[b], high=BINS[b + 1], rows=int((index == b).sum()), mean_probability=float(probabilities[index == b].mean()),
                 observed_rate=float(labels[index == b].mean())) for b in range(len(BINS) - 1) if (index == b).any()]


def summary(labels, raw, calibrated):
    return dict(rows=int(len(labels)), base_rate=float(labels.mean()), brier_raw=float(brier_score_loss(labels, raw)),
                brier_calibrated=float(brier_score_loss(labels, calibrated)), reliability_calibrated=reliability(labels, calibrated))


def main(examples, directory, search_report, run_dir, output, *targets):
    if output.exists():
        raise FileExistsError(str(output))
    chosen = e8.choose(json.loads(search_report.read_text())['results'], targets or None)
    hours, episodes, threshold = chosen['horizon_hours'], Path(chosen['episodes']), chosen['object_day']['threshold']
    model = joblib.load(run_dir / f"hgb_object_{chosen['target']}_{hours}h.joblib")
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        matrix, day, present, label, _ = e8.frame(connection, rv.load(connection, examples), directory, episodes, hours)
        _, evaluation = e8.masks(day, present, hours)
        scores_2025, labels_2025 = model.predict_proba(matrix[evaluation])[:, 1], label[evaluation]
        test_matrix, test_day, test_present, test_label, _ = e8.frame(connection, final_v2.load(connection, examples, 'test'), directory, episodes, hours)
    rows = test_present & (test_day <= test_day[test_present].max() - (hours - 24) * e7.HOUR)
    if not np.isclose(e8.margin_threshold(labels_2025, scores_2025)[0], threshold):
        raise ValueError('2025 scores do not reproduce the run threshold')
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(scores_2025, labels_2025)
    scores_test = model.predict_proba(test_matrix[rows])[:, 1]
    result = dict(target=chosen['target'], horizon_hours=hours, fit_on='2025 object-days, model trained before 2025',
                  calibrated_probability_at_threshold=float(calibrator.predict([threshold])[0]),
                  year_2025=summary(labels_2025, scores_2025, calibrator.predict(scores_2025)),
                  test_2026=summary(test_label[rows], scores_test, calibrator.predict(scores_test)),
                  protocol='isotonic fitted on 2025 only; test 2026 used once to report calibration, nothing refitted on it')
    output.mkdir(parents=True)
    joblib.dump(calibrator, output / 'isotonic.joblib')
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:6]), *sys.argv[6:])
