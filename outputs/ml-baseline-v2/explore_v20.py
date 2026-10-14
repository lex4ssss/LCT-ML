import argparse
import json
from importlib.metadata import version as package_version
import os
from pathlib import Path
import time
import duckdb
import joblib
import numpy as np
import adaptive_v13 as adaptive
import explore_v8 as e8
import explore_v16 as prior
import explore_v18 as previous_time
import models_v12 as models
import recurrence_v20 as recurrence
import telemetry_v12 as telemetry
import time_v18 as timing


VARIANTS = ("H_LONG", "C_LONG", "C_TIME_TN_LONG")


def toggles(variant):
    if variant not in ("R", "C_B", "C_TIME_TN", *VARIANTS):
        raise ValueError("Unknown recurrence variant")
    flags = previous_time.toggles("C_TIME_TN" if "TIME_TN" in variant else "R")
    flags.update(object_identity=variant.startswith("C_"), long_recurrence=variant in VARIANTS)
    return flags


def accepted(candidate, reference, incumbent):
    return bool(models.accepted(candidate, reference) and candidate["ap"] >= incumbent["ap"] + 0.005 and
                all(candidate[name] >= incumbent[name]
                    for name in ("precision", "recall", "f1", "precision_at_recall_0_5")))


def select(results):
    eligible = [name for name in VARIANTS if accepted(results[name]["validation"], results["R"]["validation"],
                                                     results["S"]["validation"])]
    if not eligible:
        return "S"
    return max(eligible, key=lambda name: (results[name]["validation"]["ap"], results[name]["validation"]["margin"],
                                          -results[name]["features"], -results[name]["model_count"]))


def reference_result(target, variant, cache_report, time_report, manifest, time_path, data, evaluation):
    source = cache_report if variant in ("R", "C_B") else time_report
    row = source["targets"][target]["results"][variant]
    path = (manifest.parent if variant in ("R", "C_B") else time_path.parent) / f"{target}-{variant}-scores.npy"
    if "score_digest" in row and adaptive.file_digest(path) != row["score_digest"]:
        raise ValueError("Frozen reference scores changed")
    scores = np.load(path)
    validation = models.metrics(data["labels"][evaluation], scores, data["rule"][evaluation],
                                data["strong_rule"][evaluation], row["validation"]["threshold"])
    for name in ("rows", "base_rate", "ap", "precision", "recall", "f1", "threshold"):
        if not np.isclose(validation[name], row["validation"][name], rtol=1e-12, atol=1e-12):
            raise ValueError("Frozen reference metrics changed")
    return scores, dict(validation=validation, toggles=toggles(variant), source_toggles=row["toggles"],
                       source_variant=variant, features=row["features"],
                       model_count=row.get("model_count", 1))


def search(cache, manifest, channel_path, time_path, output):
    source = Path(__file__).parent
    old = json.loads(manifest.read_text())
    channel_report = json.loads(channel_path.read_text())
    time_report = json.loads(time_path.read_text())
    source_identity = prior.fingerprint(source, cache, manifest, old)
    if not channel_report["complete"] or not time_report["complete"]:
        raise ValueError("Incomplete source experiment")
    if channel_report["test_evaluation"] or time_report["test_evaluation"]:
        raise ValueError("Search reference must exclude test inference")
    if source_identity != channel_report["fingerprint"] or source_identity != time_report["fingerprint"]["source"]:
        raise ValueError("Source fingerprints differ")
    channel_features = channel_path.parent / "channel_features.joblib"
    time_labels = time_path.parent / "time_labels.joblib"
    if adaptive.file_digest(channel_features) != channel_report["feature_digest"]:
        raise ValueError("Channel feature cache changed")
    if adaptive.file_digest(time_labels) != time_report["label_digest"]:
        raise ValueError("Time label cache changed")
    if adaptive.file_digest(channel_path) != time_report["fingerprint"]["channel_report"]:
        raise ValueError("Source channel report changed")
    for name, digest in time_report["fingerprint"]["code"].items():
        if adaptive.file_digest(source / name) != digest:
            raise ValueError("Time source implementation changed")
    packages = {name: package_version(name) for name in old["packages"]}
    threads = {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    if packages != old["packages"] or threads != channel_report["threads"]:
        raise ValueError("Runtime changed")
    identity = dict(source=source_identity, channel_report=adaptive.file_digest(channel_path),
                    time_report=adaptive.file_digest(time_path), channel_features=adaptive.file_digest(channel_features),
                    time_labels=adaptive.file_digest(time_labels),
                    flat_reference_scores={target: adaptive.file_digest(manifest.parent / f"{target}-C_B-scores.npy")
                                           for target in ("incident", "neispraven")},
                    code={name: adaptive.file_digest(source / name) for name in
                          ("recurrence_v20.py", "explore_v20.py", "spec_explore_v20.txt")})
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "search.json"
    report = prior.restore_report(report_path, identity, packages)
    report.update(threads=threads, previous_test_looks=8, validation_reused=True,
                  long_feature_names=recurrence.feature_names(), hgb_parameters=channel_report["hgb_parameters"],
                  catboost_parameters=channel_report["catboost_parameters"])
    previous_time.persist(report, report_path)
    shared = joblib.load(cache)
    if np.any(shared["days"] >= np.datetime64("2026-01-01", "us")):
        raise ValueError("Test rows in search cache")
    built = joblib.load(channel_features)
    labels = joblib.load(time_labels)
    long_path = output / "recurrence_features.joblib"
    if long_path.exists() and "long_feature_digest" in report:
        if adaptive.file_digest(long_path) != report["long_feature_digest"]:
            raise ValueError("Recurrence cache changed")
        long_features = joblib.load(long_path)
    else:
        long_features = {}
        paths = {Path(name).name: Path(name) for name in old["inputs"]}
        registry_path = next(Path(name) for name in old["inputs"] if Path(name).suffix == ".csv")
        mapping = telemetry.registry(registry_path)
        report["feature_audit"] = {}
        with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as connection:
            for target, hours in (("incident", 72), ("neispraven", 168)):
                prior.emit("recurrence_features_started", target=target, cutoff=str(shared["days"].max()))
                history = recurrence.read_history(connection, paths[f"episodes-900-{target}.parquet"],
                                                  mapping, shared["days"].max())
                matrix = recurrence.feature_matrix(history, shared["object_ids"], shared["days"], hours)
                early = np.flatnonzero(shared["days"] < np.datetime64("2025-01-01", "us"))
                selected = early[np.linspace(0, len(early) - 1, 37).astype(int)]
                cutoff = shared["days"][selected].max()
                prefix = history[history["first_alarm_at"] <= cutoff]
                rebuilt = recurrence.feature_matrix(prefix, shared["object_ids"][selected], shared["days"][selected], hours)
                np.testing.assert_allclose(matrix[selected], rebuilt, rtol=0, atol=0, equal_nan=True)
                long_features[target] = matrix
                report["feature_audit"][target] = dict(rows=len(matrix), features=matrix.shape[1],
                    unique_recorded_object_starts=len(history), prefix_append_invariance_rows=len(selected),
                    cutoff=str(shared["days"].max()))
                prior.emit("recurrence_features_ready", target=target, audit=report["feature_audit"][target])
        joblib.dump(long_features, long_path)
        report["long_feature_digest"] = adaptive.file_digest(long_path)
        previous_time.persist(report, report_path)
    for target, hours in (("incident", 72), ("neispraven", 168)):
        data = shared["targets"][target]
        previous_time.verify_labels(labels[target], data, shared["days"], hours)
        train, evaluation = e8.masks(shared["days"], data["present"], hours)
        own = report["targets"].setdefault(target, dict(horizon_hours=hours, results={}))
        prior.verify_scores(own["results"], output, target)
        incumbent = "R" if target == "incident" else "C_TIME_TN"
        references = {}
        for variant in ("R", "C_B", "C_TIME_TN"):
            scores, row = reference_result(target, variant, old, time_report, manifest, time_path, data, evaluation)
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
        previous_time.persist(report, report_path)
        for variant in VARIANTS:
            if variant in own["results"]:
                continue
            flags, started = toggles(variant), time.monotonic()
            prior.emit("training_started", target=target, variant=variant, toggles=flags)
            parts = [shared["base"], data["history"]]
            if flags["channel_trajectories"]:
                parts += [built["trajectories"], built["numeric"]]
            parts.append(long_features[target])
            matrix = np.column_stack(parts)
            if flags["conditional_time_heads"]:
                model = timing.fit(variant, matrix[train], labels[target]["next_hours"][train],
                                   shared["object_ids"][train], shared["categories"], shared["days"][train], hours)
                model_count, training = len(model.heads), model.training
            else:
                model = models.fit_model(variant, matrix[train], data["labels"][train],
                                         shared["object_ids"][train], shared["categories"], shared["days"][train])
                model_count, training = 1, None
            scores = models.predict_model(variant, model, matrix[evaluation], shared["object_ids"][evaluation],
                                           shared["categories"])
            models.save_verified(model, output / f"{target}-{variant}.joblib", variant, matrix[evaluation][:37],
                                 shared["object_ids"][evaluation][:37], shared["categories"])
            validation = models.metrics(data["labels"][evaluation], scores, data["rule"][evaluation],
                                        data["strong_rule"][evaluation])
            score_path = output / f"{target}-{variant}-scores.npy"
            np.save(score_path, scores)
            matched = "C_TIME_TN" if flags["conditional_time_heads"] else ("C_B" if flags["object_identity"] else "R")
            own["results"][variant] = dict(validation=validation, toggles=flags, model_count=model_count,
                head_training=training, train_rows=int(train.sum()), features=matrix.shape[1] + int(flags["object_identity"]),
                score_digest=adaptive.file_digest(score_path), matched_reference=references[matched],
                accepted_for_further_study=accepted(validation, references["R"]["validation"], references[incumbent]["validation"]),
                elapsed_seconds=time.monotonic() - started)
            previous_time.persist(report, report_path)
            prior.emit("variant_complete", target=target, variant=variant, validation=validation, toggles=flags)
            del model, matrix
        own["winner_for_further_study"] = select(own["results"])
        prior.emit("target_selected", target=target, winner=own["winner_for_further_study"])
    report["complete"] = True
    previous_time.persist(report, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("cache", "manifest", "channel_report", "time_report", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    search(args.cache, args.manifest, args.channel_report, args.time_report, args.output)
