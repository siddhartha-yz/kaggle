"""Validated, single-holdout CatBoost baseline. Validation needs only stdlib."""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass


ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    seed: int = 42
    validation_fraction: float = 0.2
    iterations: int = 1000
    learning_rate: float = 0.05
    depth: int = 6
    early_stopping_rounds: int = 50
    threads: int = 4
    device: str = "CPU"
    gpu_devices: str = "0"
    id_column: str = "id"
    positive_label: str | None = None

    def validate(self):
        for name in ("seed", "iterations", "depth", "early_stopping_rounds", "threads"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be an integer in its valid positive range")
        if self.depth > 16:
            raise ValueError("depth must be <= 16")
        if self.seed > 2**32 - 1:
            raise ValueError("seed must be <= 2**32 - 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.device not in ("CPU", "GPU"):
            raise ValueError("device must be CPU or GPU")
        if not isinstance(self.id_column, str) or not self.id_column.strip():
            raise ValueError("id_column must be a nonempty string")
        if self.positive_label is not None and not isinstance(self.positive_label, str):
            raise ValueError("positive_label must be a string (CSV labels are strings)")


def csv_rows(path):
    """Reject duplicate columns and malformed rows before pandas can alter them."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if not header or len(set(header)) != len(header) or any(not c.strip() for c in header):
            raise ValueError(f"{path}: missing, blank or duplicate column names")
        yield header
        count = 0
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"{path}: wrong number of fields at row {line}")
            count += 1
            yield dict(zip(header, row))
        if not count:
            raise ValueError(f"{path}: no data rows")


def read_ids_and_labels(path, id_column, target=None, sample=False):
    rows = csv_rows(path)
    header = next(rows)
    if id_column not in header or (target is not None and target not in header):
        raise ValueError(f"{path}: required ID/target column missing")
    ids, labels = [], []
    seen = set()
    for row in rows:
        identity = row[id_column]
        if not identity.strip() or identity in seen:
            raise ValueError(f"{path}: blank or duplicate ID")
        seen.add(identity)
        ids.append(identity)
        if target is not None:
            value = row[target]
            if not value.strip():
                raise ValueError(f"{path}: missing target")
            if sample:
                validate_probabilities([value], 1)
            else:
                labels.append(value)
    return header, ids, labels


def validate_probabilities(values, expected_count):
    if len(values) != expected_count:
        raise ValueError("Probability count does not match expected row count")
    result = []
    for value in values:
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("Probabilities must be finite and within [0, 1]")
        result.append(number)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(data_dir, config):
    config.validate()
    paths = {name: Path(data_dir) / f"{name}.csv"
             for name in ("train", "test", "sample_submission")}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Missing data file: {path}")
    sample_header = next(csv_rows(paths["sample_submission"]))
    if len(sample_header) != 2 or config.id_column not in sample_header:
        raise ValueError("sample_submission must contain exactly ID and one target column")
    target = next(c for c in sample_header if c != config.id_column)
    train_header, train_ids, labels = read_ids_and_labels(
        paths["train"], config.id_column, target)
    test_header, test_ids, _ = read_ids_and_labels(paths["test"], config.id_column)
    _, sample_ids, _ = read_ids_and_labels(
        paths["sample_submission"], config.id_column, target, sample=True)
    if set(train_header) != set(test_header) | {target} or target in test_header:
        raise ValueError("Train/test feature columns must match exactly; test must omit target")
    features = [c for c in train_header if c not in (target, config.id_column)]
    if not features:
        raise ValueError("No features remain after dropping ID and target")
    if set(train_ids) & set(test_ids):
        raise ValueError("Train and test IDs overlap")
    if len(test_ids) != len(sample_ids) or set(test_ids) != set(sample_ids):
        raise ValueError("Test and sample_submission IDs/row counts do not match")
    counts = Counter(labels)
    if len(counts) != 2 or min(counts.values()) < 2:
        raise ValueError("Target must have exactly two nonmissing classes, >=2 rows each")
    positive = config.positive_label
    if positive is None:
        if set(counts) != {"0", "1"}:
            raise ValueError("Non-0/1 target: explicitly set positive_label / --positive-label")
        positive = "1"
    if positive not in counts:
        raise ValueError("positive_label does not match a target class")
    n_valid = math.ceil(len(labels) * config.validation_fraction)
    if min(n_valid, len(labels) - n_valid) < 2:
        raise ValueError("Holdout and training partitions each need at least two rows")
    return {
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "hashes": {key: sha256(value) for key, value in paths.items()},
        "features": features, "target": target, "positive_label": positive,
        "class_counts": dict(counts), "labels": [int(v == positive) for v in labels],
        "train_ids": train_ids, "test_ids": test_ids, "sample_ids": sample_ids,
        "sample_header": sample_header,
    }


def dependency_versions():
    modules = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
               "catboost": "catboost"}
    missing = [dist for module, dist in modules.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise ValueError("Missing training dependencies: " + ", ".join(missing)
                         + ". Nothing installed automatically; see requirements.txt.")
    return {dist: importlib.metadata.version(dist) for dist in modules.values()}


def prepare_features(train_fold, other_frames):
    """Infer types using only the training fold; native CatBoost handles numeric NaN."""
    import numpy as np
    import pandas as pd

    frames = [train_fold.copy()] + [frame.copy() for frame in other_frames]
    categorical = []
    for name in train_fold.columns:
        observed = train_fold[name].replace("", np.nan)
        numeric = pd.to_numeric(observed, errors="coerce")
        is_numeric = numeric.notna().any() and numeric[observed.notna()].notna().all()
        if is_numeric:
            for frame in frames:
                values = pd.to_numeric(frame[name].replace("", np.nan), errors="raise").astype(float)
                if np.isinf(values).any():
                    raise ValueError(f"Infinite numeric feature: {name}")
                frame[name] = values
        else:
            categorical.append(name)
            for frame in frames:
                # Prefix prevents a real category from colliding with the missing marker.
                frame[name] = frame[name].map(lambda v: "M:" if v == "" else "V:" + v)
    return frames, categorical


def write_submission(path, info, probabilities):
    probabilities = validate_probabilities(probabilities, len(info["test_ids"]))
    by_id = dict(zip(info["test_ids"], probabilities))
    id_column = next(c for c in info["sample_header"] if c != info["target"])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=info["sample_header"])
        writer.writeheader()
        for identity in info["sample_ids"]:
            writer.writerow({id_column: identity, info["target"]: by_id[identity]})
    header, ids, _ = read_ids_and_labels(
        path, id_column, info["target"], sample=True)
    if header != info["sample_header"] or ids != info["sample_ids"]:
        raise ValueError("Saved submission schema/order does not match sample_submission")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                          + "\n", encoding="utf-8")


def run(data_dir, output_dir, config):
    info = validate_inputs(data_dir, config)
    versions = dependency_versions()
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    # Never overwrite an earlier run. All runtime files stay under this ignored directory.
    output_dir = Path(output_dir).resolve()
    outputs_root = (ROOT / "outputs").resolve()
    if not output_dir.is_relative_to(outputs_root) or output_dir == outputs_root:
        raise ValueError(f"output_dir must be a new run directory inside {outputs_root}")
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}; choose a new run name")
    x = pd.read_csv(info["paths"]["train"], dtype=str, keep_default_na=False,
                    usecols=info["features"])[info["features"]]
    test = pd.read_csv(info["paths"]["test"], dtype=str, keep_default_na=False,
                       usecols=info["features"])[info["features"]]
    y = np.asarray(info["labels"])
    fit_indices, valid_indices = train_test_split(
        np.arange(len(y)), test_size=config.validation_fraction,
        stratify=y, random_state=config.seed)
    if any(len(np.unique(y[index])) != 2 for index in (fit_indices, valid_indices)):
        raise ValueError("Both partitions must contain both target classes")
    (fit_x, valid_x, test_x), categorical = prepare_features(
        x.iloc[fit_indices], [x.iloc[valid_indices], test])
    params = dict(iterations=config.iterations, learning_rate=config.learning_rate,
                  depth=config.depth, random_seed=config.seed, loss_function="Logloss",
                  eval_metric="AUC", thread_count=config.threads, task_type=config.device,
                  allow_writing_files=False)
    if config.device == "GPU":
        params["devices"] = config.gpu_devices
    model = CatBoostClassifier(**params)
    model.fit(fit_x, y[fit_indices], cat_features=categorical,
              eval_set=(valid_x, y[valid_indices]), use_best_model=True,
              early_stopping_rounds=config.early_stopping_rounds, verbose=False)
    class_order = list(model.classes_)
    if set(class_order) != {0, 1}:
        raise ValueError("Unexpected model class ordering")
    positive_index = class_order.index(1)
    valid_p = validate_probabilities(model.predict_proba(valid_x)[:, positive_index], len(valid_indices))
    test_p = validate_probabilities(model.predict_proba(test_x)[:, positive_index], len(test))
    auc = float(roc_auc_score(y[valid_indices], valid_p))
    if not math.isfinite(auc):
        raise ValueError("Nonfinite validation ROC-AUC")
    # Detect input changes between validation and fitting before publishing any artifacts.
    if any(sha256(info["paths"][name]) != value for name, value in info["hashes"].items()):
        raise ValueError("An input CSV changed during this run")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    save_json(output_dir / "config.json", {**asdict(config), "positive_label": info["positive_label"]})
    save_json(output_dir / "metadata.json", {
        "python": platform.python_version(), "packages": versions, "git_commit": revision,
        "input_paths": info["paths"], "output_dir": str(output_dir),
        "git_dirty": dirty, "input_sha256": info["hashes"], "feature_columns": info["features"],
        "categorical_columns": categorical, "target": info["target"],
        "class_counts": info["class_counts"], "model_parameters": params,
        "numeric_missing": "native CatBoost NaN", "category_encoding": "M: / V:<raw value>",
    })
    metrics = {"validation_roc_auc": auc, "constant_reference_roc_auc": 0.5,
               "fit_rows": len(fit_indices), "validation_rows": len(valid_indices),
               "test_rows": len(test), "best_iteration_zero_based": int(model.get_best_iteration()),
               "tree_count": int(model.tree_count_), "model_fit": "training partition only"}
    save_json(output_dir / "metrics.json", metrics)
    model.save_model(str(output_dir / "model.cbm"))
    with (output_dir / "validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([config.id_column, "label", "probability"])
        writer.writerows((info["train_ids"][int(index)], int(y[index]), value)
                         for index, value in zip(valid_indices, valid_p))
    write_submission(output_dir / "submission.csv", info, test_p)
    # Written last: consumers must require this marker rather than trust a partial run.
    save_json(output_dir / "SUCCESS.json", {"status": "completed", **metrics})
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--positive-label")
    parser.add_argument("--device", choices=["CPU", "GPU"])
    parser.add_argument("--validate-only", action="store_true", help="No training dependencies needed")
    args = parser.parse_args(argv)
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        if args.positive_label is not None:
            settings["positive_label"] = args.positive_label
        if args.device is not None:
            settings["device"] = args.device
        config = Config(**settings)
        if args.validate_only:
            info = validate_inputs(args.data_dir, config)
            print(json.dumps({"status": "validated_only", "train_rows": len(info["train_ids"]),
                  "test_rows": len(info["test_ids"]), "target": info["target"],
                  "positive_label": info["positive_label"], "features": info["features"]}))
        else:
            print(json.dumps(run(args.data_dir, args.output_dir, config)))
        return 0
    except (OSError, ValueError, TypeError, csv.Error, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
