# Kaggle S6E10 optimization and validation record

This directory preserves the actual cloud experiment code and the subsequent local CPU verification. It differs from the earlier `../baseline.py` preparation scaffold. The historical official baseline achieved public ROC-AUC **0.95816**; the local workflow never logs in to Kaggle or submits files.

## Current research window — snapshot 2026-10-17 22:08 UTC

The current confirmed public best is **0.96000**, rank **1009/2182** at the parent's 19:52 UTC observation. The new window has used **2 of 4** authorized formal submissions; **2 remain**, including the v5 candidate now ready for the parent to submit. One further slot is unreserved. The deadline is **2026-10-18 01:02:51 UTC**, with training start cutoff 00:40 and hard training cutoff 00:50. All fits run serially on CPU with at most six training threads and no paid resources. The historical counts and deadlines in the sections below refer to their earlier windows.

| Delivered candidate | Fixed five-fold OOF AUC | Confirmed public AUC | Audit |
| --- | ---: | ---: | --- |
| v3: train-fold-only Distance group statistics | 0.9603367815 | 0.95977 | [v3 audit](submission_distance_stats_v3_audit.json) |
| v4: statistics plus original-only teacher and lookup | 0.9605951395 | **0.96000** | [v4 audit](submission_distance_original_v4_audit.json) |
| v5: frozen deeper CatBoost blend | **0.9608926464** | Pending parent submission | [v5 audit](submission_cb_deep_v5_audit.json) |

The v4 Library file is `libfile_a0f39f797ee08191821d362559ac5c47`; its CSV SHA-256 is `87a7d34b2dd788ba1623283e06e689bd8f33e1bef427bc473bccf26c6e2d6e43`. Both delivered CSVs have 299844 rows and passed official ID-order, finite-probability, saved-model reload and CSV-readback checks. Models, data and predictions remain outside Git. Only the parent cloud browser submits; there are no local Kaggle credentials or login actions.

The additional source is the public CC0 aviation satisfaction dataset, with **21 exact feature-overlap rows excluded globally** before any source-only model or lookup fitting. Of 17 exact training overlaps, 3 have conflicting labels. All synthetic frequency, category and group mappings fit outer-train only; target encoding uses inner OOF values for training rows. See [overlap audit](original_overlap_audit.json) and [label-conflict audit](original_exact_label_audit.json).

Completed follow-ups retained both positive and failed results. The best predeclared fixed heterogeneous blend reached OOF **0.9607860791**, a gain of **0.0001909396** over v4 with all five fold directions positive. It did **not** meet the frozen **0.000200** submission effect gate and was not exported. See [fixed-blend report](research_heterogeneous_oof_report.json). Rating-distribution, strict interaction-encoding, second original-teacher and wider sampled LightGBM contrasts all failed their primary screen; see [four-contrast results](research_stage8_selection.json). The neural candidate failed its confirmation interval gate and was excluded from OOF combination.

Strict nested stacking completed all five folds normally at **21:14 UTC**. Its shallow residual meta-model passed primary and confirmation, but final OOF AUC **0.9607592663** improved only **0.0001641268** over v4 (conditional 95% interval **[0.0001041732, 0.0002240804]**), below the frozen effect gate. All five fold deltas were positive. No full refit or CSV was produced. See [nested results](research_results/nested_stack_oof_result.json). Two source-only context representations then both failed their primary screen; see [context results](research_context_selection.json).

The deeper CatBoost route completed all validation and full refits. The frozen candidate uses **46.875% v4 LGB +9.375% original-aware CB8 +18.75% residual-prior LGB +25% CB10**. CB10 has **813** trees, depth10, learning rate .04, L2 10 and seed20261018; its iteration count and blend weight were frozen before confirmation. Primary AUC was **0.9600030291**, confirmation **0.9608201578**, and fixed five-fold OOF **0.9608926464**. The pooled OOF gain over v4 is **0.0002975069**, with conditional 95% interval **[0.0002535770, 0.0003414369]**; every fold improved. It also exceeds the prior fixed core by **0.0001065673**.

Full CPU refits and CSV checks completed at **2026-10-17 22:05:36 UTC**. The existing full v4 model was reused; all three newly fitted models passed reload comparison with zero prediction difference. The independent CSV component-sum comparison differed by at most **1.1102230246251565e-16**. The delivered `submission_cb_deep_v5.csv` has **299844** rows and **7986343 bytes**, Library ID **`libfile_ca36559c66908191817867f5880b4635`**, version **0**, and SHA-256 **`0f9a978051e4f0fa164928f3b975a71346b60477959ee5cf50abb580a706e872`**. Library linkage and the post-upload local checksum were verified. The parent has not yet reported a formal v5 submission or public score. See [Library delivery record](submission_cb_deep_v5_library.json) and [five-fold results](research_results/cb_deep_oof_result.json).

See [research protocol](research_protocol.json), [submission ledger](research_submission_ledger.json), [nested plan](research_nested_stack_plan.json), and [nested checkpoint](research_nested_validation_checkpoint.json). Development and confirmation samples have been reused for feature selection; paired intervals are conditional on chosen models and exclude selection uncertainty. Do not replay completed queues or interpret these scores as an untouched final test.

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

## Fixed stochastic LightGBM continuation (2026-10-17 18:41–18:43 UTC)

A second bounded continuation was explicitly authorized after the previous test. This single model kept the targeted category features, 63 leaves, learning rate 0.04, minimum child rows 150, L2 10, and **639 fixed trees**, but used seed **20261017**, row sampling **0.8** each iteration and feature sampling **0.9**. The proposed blend weights were frozen before evaluation: **0.4 cb8 / 0.4 original lgb63cats / 0.2 new bagged model**. There was no seed, tree-count, parameter-grid, weight or public-score search.

| Same validation rows | Current submitted blend AUC | New frozen blend AUC | Paired difference | Conditional 95% interval |
| --- | ---: | ---: | ---: | --- |
| Development seed42 | 0.9581313490780615 | 0.9582112054287225 | +0.0000798563506610 | [0.0000421990005524, 0.0001175137007695] |
| Stability seed20261017 | 0.9590578561530560 | 0.9591094233123446 | +0.0000515671592886 | [0.0000134199794242, 0.0000897143391529] |

The predeclared criterion passed. Both validation models and the full 699635-row refit completed, using at most six CPU threads. The full model completed at **18:42:26 UTC**. Original full components were reused without retraining. `bagging_ensemble.json` binds all three full-model hashes and tree counts. As before, these are reused training samples and conditional intervals exclude model selection uncertainty.

The new CSV **submission_cb8_lgb63cats_bagged_v2.csv** passed independent exact ID/column/count/range/readback/component-agreement checks at **18:42:59 UTC**, was saved to Library before the requested **18:46 UTC** delivery cutoff, and was handed to the parent for the remaining official submission decision. It has **299844 rows**, **7984326 bytes**, and SHA-256 **49ad4d7e282300289c9c9fc263313c6232a674b3846d48caa7f35ae8f89f4766**. The previous best CSV remains unchanged. No local Kaggle login or formal submission occurred; the parent-reported total is still **1 used, at most 1 remaining** until the parent reports otherwise. No new candidate public score is known.

See `bagging_plan.json`, `bagging_result.json`, `bagging_model_results.json` and `bagging_submission_audit.json` for the small records. The training guard stops children at **18:45 UTC**, reserving time for delivery before the original **18:47:49 UTC** hard cutoff. Model, validation and CSV files remain ignored and local.

## New authorized research window

The parent confirmed the bagged candidate **Complete**, public AUC **0.95857**, rank **1193/2179** at that observation, bringing the historical submission total to **3**. A new window is authorized until **2026-10-18 01:02:51 UTC**, with **zero paid resources** and **at most 4 new formal submissions**, all coordinated by the parent cloud browser. The displayed top-100 line was 0.96190, tied through rank 109; exceeding it is a target, not a guarantee. See `public_result_bagging_20261017.json`.
