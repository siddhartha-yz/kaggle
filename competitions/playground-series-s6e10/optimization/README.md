# Frozen CatBoost + LightGBM optimization

This directory preserves the actual cloud experiment code and the subsequent local CPU verification. It differs from the earlier `../baseline.py` preparation scaffold. The historical official baseline achieved public ROC-AUC **0.95816**; the local workflow never logs in to Kaggle or submits files.

## Evidence and provenance

The cloud handoff contained eight completed development experiments on the original stratified seed42 80/20 split. The selected 50/50 probability blend of CatBoost depth8 (1308 trees) and LightGBM lgb63cats (639 trees) achieved development AUC **0.9581313490780615**, versus baseline **0.9575997720187789**. ID and target are excluded, True is the positive class, and only official competition data are used.

Verified source archives:

- Experiment package SHA-256: `f77689759f0994ec6866a6c9386b3ff37ff7dffb0d979a1034089894e4da86d1`.
- Official data archive SHA-256: `5227297a3cd4608c7e5f995624cc3526ae49352042772329f4da199b367cf253`.
- All 62 handoff file hashes and all three official CSV hashes matched before local use.
- The original seed42 split and all eight saved AUCs were reproduced. Each selected model passed 1003 cross-host CPU inference comparisons with zero maximum difference.

## Frozen verification

`frozen_recipe.json` records the recipe and decision rule before confirmation. The second stratified 80/20 split uses seed **20261017**; model seed remains **42**. Baseline 956, CatBoost 1308, and LightGBM 639 tree counts are fixed. No confirmation early stopping or blend-weight search is performed. Proceed to full refit only when the frozen blend's paired AUC difference has a positive lower conditional 95% bound.

The local CPU confirmation completed at 2026-10-17 18:17 UTC: baseline AUC **0.9583800202558467**, frozen blend AUC **0.959057856153056**, paired difference **+0.0006778358972094**, conditional 95% interval **[0.0004603604837175, 0.0008953113107013]**. The frozen decision rule passed; see `local_confirmation.json`. The two validation splits share **27914** rows.

The second split reuses official training samples and overlaps the development validation set. It checks stability and is **not an untouched independent test**. Conditional paired intervals exclude hyperparameter selection uncertainty and do not guarantee leaderboard improvement.

The LightGBM recipe adds categorical copies of Inflight wifi service, Online boarding, Ease of Online booking, and Gate location. Original zero ratings remain valid values.

## Reproduction

Use Python 3.12 in a dedicated venv and install `requirements.txt` from official PyPI. Keep the official CSV files in ignored `data/`, the original cloud baseline artifacts in ignored `artifacts/`, and cloud development artifacts in ignored `iteration_artifacts/`. They are intentionally excluded from Git.

```bash
python local_preflight.py
python local_bounded_run.py confirm
python local_confirm_compare.py
# Only when the frozen confirmation rule passes:
python local_bounded_run.py full
python local_finalize.py
```

`local_bounded_run.py` enforces the authorized 2026-10-17 18:47:49 UTC cutoff, runs serial CPU jobs with six threads, and refuses existing run directories. After that deadline it will refuse training. Any future work needs a separately authorized budget and an explicit update to this guard. Do not replay the old cloud experiment queue.

`local_finalize.py` accepts only complete full-data models at the frozen tree counts. It binds data and model hashes in `best_ensemble.json`, uses `predict_best.py` for inference, and audits the exact lexical test/sample ID order, 299844 rows, column names, finite [0,1] probabilities, CSV readback, and agreement with saved full-model component predictions.

The original cloud `iterate.py` is preserved, including its permissive run-directory creation; use the bounded wrapper to prevent accidental overwrites. The model artifacts and CSV remain local/Library deliverables, never Git content. Parent-session coordination retains the total cap of two additional official submissions.

## Final local result

Both frozen components completed full-data CPU refits on all **699635** training rows at 2026-10-17 18:22 UTC. `best_ensemble.json` binds the actual full-model hashes and tree counts. The final candidate CSV passed the independent audit at **18:23:15 UTC**: **299844** rows, columns `id,satisfaction`, exact official lexical ID order, unique IDs, finite probabilities in [0,1], and readback agreement with the saved component predictions (maximum difference **2.220446049250313e-16**).

- Filename: `submission_cb8_lgb63cats_frozen.csv` (excluded from Git).
- Size: **7984405 bytes**.
- SHA-256: `e5006e109f93ee8263698fecc64881ad8a4311d94daef1055a12a11e9475769f`.
- Probability range: **0.0028820971790049325** to **0.9927350678867439**.
- Local formal submissions: **0**. The parent session coordinates any official upload and scoring within the original total cap of two new submissions. At the initial local handoff there was no candidate public score; the later confirmed score is recorded below.

`full_refit_results.json` and `submission_audit.json` provide the compact supporting records. All local training completed before the authorized cutoff. No additional experiment search was performed.

## Official candidate score reported by parent

The parent cloud browser confirmed **Complete**, public AUC **0.95855**, rank **1209/2177**, for the frozen candidate CSV with SHA-256 `e5006e109f93ee8263698fecc64881ad8a4311d94daef1055a12a11e9475769f`. This improves the historical baseline public AUC 0.95816 by **0.00039**. One of the two authorized new formal submissions has been used; at most one remains. Ranking is a time-specific observation. See `public_result_20261017.json`.

## One bounded continuation test (2026-10-17 18:37 UTC)

After the parent confirmed public AUC **0.95855**, a single fixed complementary blend was evaluated: **40% cb8 / 40% lgb63cats / 20% lgb63allcats**. The added model uses the already-selected **415** trees. Weights and the acceptance criterion were recorded before evaluation in `continuation_plan.json`; no public-score weight search was performed.

| Same validation rows | Submitted 50/50 blend AUC | New fixed blend AUC | Paired difference | Conditional 95% interval |
| --- | ---: | ---: | ---: | --- |
| Development seed42 | 0.9581313490780615 | 0.9581626604818705 | +0.0000313114038090 | [-0.0000097812106292, +0.0000724040182471] |
| Stability seed20261017 | 0.9590578561530560 | 0.9590968889423398 | +0.0000390327892837 | [-0.0000013247608206, +0.0000793903393880] |

**Decision: retain the submitted best.** Both point estimates improved slightly, but the predeclared confirmation criterion required a positive lower paired conditional interval bound. It did not pass. The interval excludes selection uncertainty, and the reused confirmation samples are not an independent test. No new full refit or CSV was generated. The one six-thread CPU confirmation fit completed in **10.56 seconds**. No additional formal submission was made; the total remains **1 used, at most 1 remaining**, coordinated by the parent. All continuation training ended at **18:37:59 UTC**, before the original 18:47:49 UTC cutoff. See `continuation_result.json` and `continuation_model_result.json`.
