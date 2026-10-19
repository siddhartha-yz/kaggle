# S6E10 v12 artifact replay

The submitted v12 scored **0.96118 public ROC-AUC**. The authenticated historical snapshot at **2026-10-19 05:28 UTC** was **676/2302**; the top-500 goal was not achieved.

See [V12.md](V12.md) for the frozen recipe, validation evidence, exact CSV identity, dependencies and limitations. The supported entrypoint is [v12.py](v12.py):

```bash
python v12.py describe
python v12.py verify --artifact-root /absolute/path/to/experiment
python v12.py reproduce-csv --artifact-root /absolute/path/to/experiment --output /absolute/path/to/v12-replayed.csv
```

Verification requires the retained private artifact bundle with all 125 manifest entries, including the original source files. It uses one CPU math thread and reproduces the saved CSV without training or loading model pickles. Data, models, predictions and calibration parameters remain outside Git.

Complete training and experiment history is retained on the [research branch](https://github.com/siddhartha-yz/kaggle/tree/codex/s6e10-local-confirmation/competitions/playground-series-s6e10/optimization). The [former main backup](https://github.com/siddhartha-yz/kaggle/tree/archive/main-before-v12-split-20261019) is preserved. Main supports artifact replay; raw-data training is documented in the research archive.
