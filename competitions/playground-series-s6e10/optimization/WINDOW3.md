# Window3: 2026-10-18 05:06–11:06 UTC

Parent confirmed v6 Complete, public AUC **0.96058**, rank **858/2208** at04:00 UTC. Historical prior-window quota is exhausted. This independent window permits at most **2 new** formal submissions, parent only; zero used/reserved. No local Kaggle access.

Frozen protocol: `research_w3_plan.json`. Training is serial, CPU6threads, GPU PyTorch allocator cap2GiB, startcut10:10/hardcut10:45/deliverytarget10:55 UTC. Preserve all prior models and records.

Exploratory OOF diagnostics: v6 AUC .961135462544149. Averaging its2 neural members gains .00027957 over the mean member AUC; worst disagreement quartile has AUC .91728. This motivates a distinct shared-weight TabM ensemble; source-only neural teacher separately tests an independent representation, not warm-start transfer.

Source-only neural teacher trained on129859 safe original rows after globally excluding21exact predictor matches; no synthetic targets used. Fixed8epochs,2members completed14.88seconds at05:16:39 UTC. Adding its logit to a new raw value-twin NN failed primary: delta vs v6 **−.00002709**, vs matched old-NN weight control **−.00009538**. Do not promote.

TabM uses installed core with a custom fixed20epoch loop because PyTabKit wrapper requires validation and restores the best validation epoch. Four raw categorical vocabularies, numeric imputation/quantile/PWL bins are fitted only on outer training. No outer validation targets enter fitting. One full559708row no-score probe:2.95seconds/epoch, peak allocated481897984bytes, reserved538968064bytes. Scores do not select hardware.

Advance only with primary delta≥.0001 and positive conditional lower bound vs v6, confirmation positive lower bound, everyOOFfold positive, pooleddelta≥.0002 and positive lower bound. Frozen blend .75v6+.25newmodel. Intervals are conditional and all partitions are reused; they exclude model-selection uncertainty. No leaderboard tuning or guaranteed top100 claim.

Raw TabM completed20epochs but failed the unchanged primary gate: delta **+.00009011449**, conditional95%CI **[−.0000032792,+.0001835082]**. No weight or gate adjustment. A separately frozen one-feature expansion adds categorical twins for all original numerical values to the same TabM; `research_w3_tabm_values_plan.json` records this adaptive next hypothesis and unchanged gates. No-score probe peak allocated557395456bytes, oneepoch8.385s. An initial missingmodule import failed before any fit and is retained in `research_w3_tabm_values_preflight_repair.json`.

A conditional DeepFM fallback is frozen before seeing the TabM-value primary score. It runs only if the value-view path rejects:4fixed independent members,3epochs each, learned value embeddings +linear+pairwiseFM +MLP. The same train-only features are used. It does not choose seeds/members/weights from validation.
