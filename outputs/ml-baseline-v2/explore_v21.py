import argparse
import json
from importlib.metadata import version as package_version
import os
from pathlib import Path
import time
import joblib
import numpy as np
import adaptive_v13 as adaptive
import explore_v8 as e8
import explore_v16 as prior
import explore_v18 as previous
from explore_v20 import accepted
import models_v12 as models
import shared_v21 as shared_hazard


VARIANTS = tuple(f"{family}_SHARED_{features}" for features in ("B", "TN") for family in ("H", "C", "E"))


def matched_variant(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown shared-hazard variant")
    return variant.replace("_SHARED_", "_TIME_")


def toggles(variant):
    original = matched_variant(variant) if variant in VARIANTS else variant
    flags = previous.toggles(original)
    flags.update(shared_interval_parameters=variant in VARIANTS, long_recurrence=False)
    return flags


def select(results):
    eligible = [name for name in VARIANTS if accepted(results[name]["validation"], results["R"]["validation"],
                                                     results["S"]["validation"])]
    if not eligible:
        return "S"
    return max(eligible, key=lambda name: (results[name]["validation"]["ap"], results[name]["validation"]["margin"],
                                          -results[name]["features"], -results[name]["model_count"]))


def reference_result(target, variant, report, path, data, evaluation):
    original = report["targets"][target]["results"][variant]
    scores_path = path.parent / f"{target}-{variant}-scores.npy"
    if adaptive.file_digest(scores_path) != original["score_digest"]:
        raise ValueError("Frozen independent-interval scores changed")
    scores = np.load(scores_path)
    validation = models.metrics(data["labels"][evaluation], scores, data["rule"][evaluation],
                                data["strong_rule"][evaluation], original["validation"]["threshold"])
    for name in ("rows", "base_rate", "ap", "precision", "recall", "f1", "threshold"):
        if not np.isclose(validation[name], original["validation"][name], rtol=1e-12, atol=1e-12):
            raise ValueError("Frozen independent-interval metrics changed")
    return scores, dict(validation=validation, source_variant=variant, toggles=toggles(variant),
                       source_toggles=original["toggles"], features=original["features"],
                       model_count=original["model_count"])


def source_state(cache, manifest, channel_path, time_path):
    source = Path(__file__).parent
    original = json.loads(manifest.read_text())
    channel_report = json.loads(channel_path.read_text())
    time_report = json.loads(time_path.read_text())
    source_identity = prior.fingerprint(source, cache, manifest, original)
    if not channel_report["complete"] or not time_report["complete"]:
        raise ValueError("Incomplete source experiment")
    if channel_report["test_evaluation"] or time_report["test_evaluation"]:
        raise ValueError("Source search includes test evaluation")
    if source_identity != channel_report["fingerprint"] or source_identity != time_report["fingerprint"]["source"]:
        raise ValueError("Source fingerprints differ")
    channel_features, time_labels = channel_path.parent / "channel_features.joblib", time_path.parent / "time_labels.joblib"
    if adaptive.file_digest(channel_features) != channel_report["feature_digest"]:
        raise ValueError("Channel feature cache changed")
    if adaptive.file_digest(time_labels) != time_report["label_digest"]:
        raise ValueError("Conditional labels changed")
    if adaptive.file_digest(channel_path) != time_report["fingerprint"]["channel_report"]:
        raise ValueError("Channel report changed")
    for name, digest in time_report["fingerprint"]["code"].items():
        if adaptive.file_digest(source / name) != digest:
            raise ValueError("Independent time implementation changed")
    packages = {name: package_version(name) for name in original["packages"]}
    threads = {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    if packages != original["packages"] or threads != channel_report["threads"]:
        raise ValueError("Runtime changed")
    identity = dict(source=source_identity, channel_report=adaptive.file_digest(channel_path),
                    time_report=adaptive.file_digest(time_path), channel_features=adaptive.file_digest(channel_features),
                    time_labels=adaptive.file_digest(time_labels),
                    code={name: adaptive.file_digest(source / name) for name in
                          ("shared_v21.py", "explore_v21.py", "spec_explore_v21.txt",
                           "explore_v20.py", "recurrence_v20.py")})
    return identity, packages, threads, channel_report, time_report, channel_features, time_labels


def search(cache, manifest, channel_path, time_path, output):
    identity, packages, threads, channel_report, time_report, feature_path, label_path = source_state(
        cache, manifest, channel_path, time_path)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "search.json"
    report = prior.restore_report(report_path, identity, packages)
    report.update(threads=threads, previous_test_looks=8, validation_reused=True,
                  interval_feature="zero_based_interval_code", hgb_parameters=channel_report["hgb_parameters"],
                  catboost_parameters=channel_report["catboost_parameters"])
    previous.persist(report, report_path)
    data_shared = joblib.load(cache)
    if np.any(data_shared["days"] >= np.datetime64("2026-01-01", "us")):
        raise ValueError("Test rows in search cache")
    built, labels = joblib.load(feature_path), joblib.load(label_path)
    for target, hours in (("incident", 72), ("neispraven", 168)):
        data = data_shared["targets"][target]
        previous.verify_labels(labels[target], data, data_shared["days"], hours)
        train, evaluation = e8.masks(data_shared["days"], data["present"], hours)
        own = report["targets"].setdefault(target, dict(horizon_hours=hours, results={}))
        prior.verify_scores(own["results"], output, target)
        for name, row in own["results"].items():
            if row.get("model_digest") and adaptive.file_digest(output / f"{target}-{name}.joblib") != row["model_digest"]:
                raise ValueError("Saved shared model changed")
        incumbent = "R" if target == "incident" else "C_TIME_TN"
        references = {}
        for variant in time_report["targets"][target]["results"]:
            scores, row = reference_result(target, variant, time_report, time_path, data, evaluation)
            references[variant] = row
            if variant == "R" or variant == incumbent:
                key = "R" if variant == "R" else "S"
                score_path = output / f"{target}-{key}-scores.npy"
                np.save(score_path, scores)
                own["results"][key] = dict(row, score_digest=adaptive.file_digest(score_path))
        if incumbent == "R":
            score_path = output / f"{target}-S-scores.npy"
            np.save(score_path, np.load(output / f"{target}-R-scores.npy"))
            own["results"]["S"] = dict(own["results"]["R"], score_digest=adaptive.file_digest(score_path))
        previous.persist(report, report_path)
        for variant in VARIANTS:
            if variant in own["results"]:
                continue
            flags, started = toggles(variant), time.monotonic()
            parts = [data_shared["base"], data["history"]]
            if flags["channel_trajectories"]:
                parts += [built["trajectories"], built["numeric"]]
            matrix = np.column_stack(parts)
            training, model_digest = None, None
            if flags["equal_ensemble"]:
                components = [variant.replace("E_", family + "_", 1) for family in ("H", "C")]
                scores = sum(np.load(output / f"{target}-{name}-scores.npy") for name in components) / 2
                model_count = 2
            else:
                prior.emit("training_started", target=target, variant=variant, toggles=flags)
                model = shared_hazard.fit(variant, matrix[train], labels[target]["next_hours"][train],
                    data_shared["object_ids"][train], data_shared["categories"], data_shared["days"][train], hours)
                scores = models.predict_model(variant, model, matrix[evaluation], data_shared["object_ids"][evaluation],
                                               data_shared["categories"])
                model_path = output / f"{target}-{variant}.joblib"
                models.save_verified(model, model_path, variant, matrix[evaluation][:37],
                                     data_shared["object_ids"][evaluation][:37], data_shared["categories"])
                model_digest, model_count, training = adaptive.file_digest(model_path), 1, model.training
                del model
            validation = models.metrics(data["labels"][evaluation], scores, data["rule"][evaluation],
                                        data["strong_rule"][evaluation])
            score_path = output / f"{target}-{variant}-scores.npy"
            np.save(score_path, scores)
            own["results"][variant] = dict(validation=validation, toggles=flags, model_count=model_count,
                model_digest=model_digest, interval_training=training, train_rows=int(train.sum()),
                expanded_train_rows=sum(row["rows"] for row in training) if training else None,
                train_last_as_of=str(data_shared["days"][train].max()),
                features=matrix.shape[1] + 1 + int(flags["object_identity"]),
                prediction_features=matrix.shape[1] + int(flags["object_identity"]),
                score_digest=adaptive.file_digest(score_path), matched_reference=references[matched_variant(variant)],
                accepted_for_further_study=accepted(validation, references["R"]["validation"], references[incumbent]["validation"]),
                elapsed_seconds=time.monotonic() - started)
            previous.persist(report, report_path)
            prior.emit("variant_complete", target=target, variant=variant, validation=validation, toggles=flags)
            del matrix
        own["winner_for_further_study"] = select(own["results"])
        prior.emit("target_selected", target=target, winner=own["winner_for_further_study"])
    report["complete"] = True
    previous.persist(report, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("cache", "manifest", "channel_report", "time_report", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    search(args.cache, args.manifest, args.channel_report, args.time_report, args.output)
