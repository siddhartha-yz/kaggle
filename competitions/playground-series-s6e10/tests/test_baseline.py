"""Synthetic fixtures only: no network, real data, credentials or GPU usage."""

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline


class DataFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kaggle-synthetic-")
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)
        # 60 synthetic records, mixed types, empty values and both target classes.
        self.train = [[str(i), "" if i % 13 == 0 else str(i % 17),
                       "A" if i % 2 else "B", str(i % 2)] for i in range(60)]
        self.test = [["00100", "7", "UNSEEN"], ["00101", "", ""]]
        self.sample = [["00101", "0.5"], ["00100", "0.5"]]
        self.write("train", ["id", "age", "cabin", "satisfaction"], self.train)
        self.write("test", ["id", "age", "cabin"], self.test)
        self.write("sample_submission", ["id", "satisfaction"], self.sample)

    def write(self, name, header, rows):
        with (self.data / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def validate(self, **kwargs):
        return baseline.validate_inputs(self.data, baseline.Config(**kwargs))

    def test_valid_schema_and_id_exclusion(self):
        info = self.validate()
        self.assertEqual(info["features"], ["age", "cabin"])
        self.assertEqual(info["positive_label"], "1")
        self.assertEqual(info["class_counts"], {"0": 30, "1": 30})
        self.assertEqual(info["test_ids"], ["00100", "00101"])
        self.assertEqual(len(info["hashes"]["train"]), 64)

    def test_submission_realigns_by_id_and_preserves_leading_zeros(self):
        info = self.validate()
        output = self.data / "submission.csv"
        baseline.write_submission(output, info, [0.2, 0.8])
        with output.open() as handle:
            self.assertEqual(list(csv.reader(handle)), [
                ["id", "satisfaction"], ["00101", "0.8"], ["00100", "0.2"]])

    def test_submission_preserves_target_first_column_order(self):
        self.write("sample_submission", ["satisfaction", "id"], [["0.5", "00101"], ["0.5", "00100"]])
        info = self.validate()
        output = self.data / "submission.csv"
        baseline.write_submission(output, info, [0.2, 0.8])
        with output.open() as handle:
            self.assertEqual(next(csv.reader(handle)), ["satisfaction", "id"])

    def test_reordered_test_columns_accepted(self):
        self.write("test", ["cabin", "id", "age"], [[r[2], r[0], r[1]] for r in self.test])
        self.assertEqual(self.validate()["features"], ["age", "cabin"])

    def test_missing_file(self):
        (self.data / "train.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Missing data file"):
            self.validate()

    def test_bad_headers_and_row_width(self):
        for text in ("id,age,age,satisfaction\n0,1,2,0\n",
                     "id,,cabin,satisfaction\n0,1,A,0\n",
                     "id,age,cabin,satisfaction\n0,1,A\n"):
            with self.subTest(text=text):
                (self.data / "train.csv").write_text(text)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_empty_data(self):
        self.write("test", ["id", "age", "cabin"], [])
        with self.assertRaisesRegex(ValueError, "no data rows"):
            self.validate()

    def test_bad_train_ids(self):
        for bad_id in ("", "   ", "1", "00100"):
            with self.subTest(identity=bad_id):
                rows = [row.copy() for row in self.train]
                rows[0][0] = bad_id
                self.write("train", ["id", "age", "cabin", "satisfaction"], rows)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_duplicate_test_or_sample_ids(self):
        for name, header, rows in (
                ("test", ["id", "age", "cabin"], [self.test[0]] * 2),
                ("sample_submission", ["id", "satisfaction"], [self.sample[0]] * 2)):
            with self.subTest(name=name):
                self.setUp_files()
                self.write(name, header, rows)
                with self.assertRaisesRegex(ValueError, "duplicate ID"):
                    self.validate()

    def setUp_files(self):
        self.write("test", ["id", "age", "cabin"], self.test)
        self.write("sample_submission", ["id", "satisfaction"], self.sample)

    def test_mismatched_sample_ids_or_count(self):
        for rows in ([["00100", "0.5"]], [["00100", "0.5"], ["999", "0.5"]]):
            with self.subTest(rows=rows):
                self.write("sample_submission", ["id", "satisfaction"], rows)
                with self.assertRaisesRegex(ValueError, "IDs/row counts"):
                    self.validate()

    def test_extra_or_missing_test_features_and_target_leak(self):
        for header, rows in ((["id", "age"], [["00100", "7"]]),
                             (["id", "age", "cabin", "extra"], [["00100", "7", "A", "X"]]),
                             (["id", "age", "cabin", "satisfaction"], [["00100", "7", "A", "1"]])):
            with self.subTest(header=header):
                self.write("test", header, rows)
                with self.assertRaisesRegex(ValueError, "feature columns"):
                    self.validate()

    def test_bad_labels(self):
        for replacement in ("", "2", "0"):
            with self.subTest(replacement=replacement):
                rows = [row.copy() for row in self.train]
                if replacement == "0":
                    for row in rows:
                        row[-1] = "0"
                else:
                    rows[0][-1] = replacement
                self.write("train", ["id", "age", "cabin", "satisfaction"], rows)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_text_labels_require_explicit_positive(self):
        for row in self.train:
            row[-1] = "satisfied" if row[-1] == "1" else "dissatisfied"
        self.write("train", ["id", "age", "cabin", "satisfaction"], self.train)
        with self.assertRaisesRegex(ValueError, "explicitly"):
            self.validate()
        info = self.validate(positive_label="satisfied")
        self.assertEqual(info["labels"][:2], [0, 1])
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.validate(positive_label="unknown")

    def test_invalid_sample_probabilities(self):
        self.write("sample_submission", ["id", "satisfaction"], [["00100", "nan"], ["00101", "0"]])
        with self.assertRaisesRegex(ValueError, "finite"):
            self.validate()

    def test_invalid_config(self):
        for change in ({"seed": -1}, {"seed": 2**32}, {"threads": 0}, {"depth": 17},
                       {"iterations": 0}, {"validation_fraction": 1}, {"validation_fraction": 0.01},
                       {"learning_rate": float("nan")}, {"device": "TPU"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(**change)

    def test_cli_validation_and_missing_data_exit(self):
        command = [sys.executable, str(baseline.ROOT / "baseline.py"), "--validate-only", "--data-dir"]
        result = subprocess.run(command + [str(self.data)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "validated_only")
        result = subprocess.run(command + [str(self.data / "missing")], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Missing data file", result.stderr)

    def test_probability_validation(self):
        for values, size in (([float("nan")], 1), ([float("inf")], 1), ([-0.1], 1),
                             ([1.1], 1), ([], 1), (["invalid"], 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                baseline.validate_probabilities(values, size)
        self.assertEqual(baseline.validate_probabilities([0, 0.5, 1], 3), [0, 0.5, 1])

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy", "pandas")),
                         "numpy/pandas not installed; preprocessing integration unavailable")
    def test_preprocessing_uses_fit_fold_only(self):
        import pandas as pd
        fit = pd.DataFrame({"numeric": ["1", ""], "category": ["A", ""]})
        valid = pd.DataFrame({"numeric": ["9"], "category": ["UNSEEN"]})
        frames, categorical = baseline.prepare_features(fit, [valid])
        self.assertEqual(categorical, ["category"])
        self.assertEqual(frames[1]["category"].tolist(), ["V:UNSEEN"])
        self.assertTrue(pd.isna(frames[0]["numeric"].iloc[1]))
        # A numeric fit column must not silently become categorical because of holdout data.
        with self.assertRaises(ValueError):
            baseline.prepare_features(fit, [valid.assign(numeric="unexpected")])

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                             ("numpy", "pandas", "sklearn", "catboost")),
                         "Training dependencies absent; real CatBoost synthetic smoke test skipped")
    def test_real_catboost_cpu_smoke(self):
        output = baseline.ROOT / "outputs" / ("synthetic-test-" + uuid.uuid4().hex)
        self.addCleanup(lambda: shutil.rmtree(output, ignore_errors=True))
        config = baseline.Config(iterations=20, early_stopping_rounds=3, depth=2, threads=1)
        metrics = baseline.run(self.data, output, config)
        self.assertTrue(0 <= metrics["validation_roc_auc"] <= 1)
        for name in ("config.json", "metadata.json", "metrics.json", "model.cbm",
                     "validation.csv", "submission.csv", "SUCCESS.json"):
            self.assertTrue((output / name).is_file(), name)
        with self.assertRaisesRegex(ValueError, "already exists"):
            baseline.run(self.data, output, config)


if __name__ == "__main__":
    unittest.main()
