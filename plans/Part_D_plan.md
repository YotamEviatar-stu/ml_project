# Stage 4 plan — model evaluation & results analysis (Part D)

No skill file exists yet for this stage (only `feature-engineering` and
`model-training` are built). This plan stands alone; fold it into a SKILL.md
once the approach settles, same as Stages 1–3 did.

## Prerequisite — Part C artifacts already in `train.ipynb`

- `X_train`/`y_train` (43,013 rows) and `X_holdout`/`y_holdout` (10,754 rows,
  3,321 positives) — holdout untouched by any Part C fitting/tuning.
- `fitted_models = {"LogisticRegression": logreg_best, "RandomForest":
  rf_best, "XGBoost": xgb_best}` — each already refit on the train pool.
- `results`/`summary` — CV AUC ± std, train-fold AUC, gap per model (cell
  108–125 range). Part D's holdout metrics are a *separate*, independent
  check of this ranking, not a restatement of it.

## Section 1 — confusion matrix & metrics on the holdout

- For each of the three `fitted_models`, call `.predict_proba(X_holdout)`,
  threshold at 0.5 for hard labels, compute confusion matrix + precision,
  recall, F1, accuracy, and holdout ROC-AUC (a single point estimate, not
  CV — say so explicitly since Part C already covered fold variability).
- Explain each metric against what a false positive/negative *means* here:
  a false negative is a client who will drop out but isn't flagged — no
  chance for intervention; a false positive is a client flagged who
  wouldn't have dropped — wasted outreach, not catastrophic. This asymmetry
  is the reason to discuss recall, not just accuracy, as the metric that
  matters most for the business use case.
- One comparison table: model × TP/FP/FN/TN × precision × recall × F1 ×
  accuracy × holdout AUC.
- Markdown interpretation cell: metric meanings in context, model
  comparison, which model looks best and why (feeds Section 3).

## Section 2 — threshold check

- 0.5 is the default cut for the confusion matrix above, but given the
  recall-matters-more asymmetry from Section 1, briefly check the
  precision-recall tradeoff (PR curve or a couple of alternate thresholds)
  for the model that ends up chosen in Section 3. This doesn't change the
  submission file (that wants raw probabilities, not hard labels) — it's
  only to justify whether 0.5 was a reasonable choice for the confusion
  matrix, or worth flagging as a tunable business decision.

## Section 3 — choose one model for the deep-dive

- Decision must cite the **holdout** table from Section 1 (independent
  evidence), not just re-quote Part C's CV-AUC ranking. Expect XGBoost to
  hold up (CV AUC 0.930 vs RF 0.922 vs LR 0.854 in Part C) but confirm
  before locking it in — that's the whole point of having a holdout.

## Section 4 — deep-dive: where does the chosen model struggle?

- Confidence = `max(p, 1-p)` from `predict_proba` on `X_holdout`. Bucket
  into low/medium/high confidence (e.g. <0.6, 0.6–0.8, >0.8) and report
  accuracy per bucket — this directly answers the assignment's prompt
  ("scenarios where the model provides a lower confidence level").
- Compare feature distributions (or a cross-tab against key categoricals
  like `Client_Category`, `Payment_Terms`) between the low- and
  high-confidence groups to characterize *which* clients get uncertain
  predictions.
- Reliability/calibration curve (predicted-probability bucket vs. observed
  dropout rate) — the correct tool for "is the model's confidence
  trustworthy," and directly relevant since the final submission delivers
  raw probabilities, not hard labels.

## Section 5 — SHAP

- `shap.TreeExplainer` for RF/XGBoost (or `LinearExplainer` if LR is somehow
  chosen) on the model from Section 3. Background = a sample of `X_train`;
  explain a capped subsample of `X_holdout` (state the cap and why, e.g.
  2,000 rows for runtime — not a silent truncation).
- Global: beeswarm + mean |SHAP| bar plot → top drivers of dropout.
- Local: waterfall/force plots for a few representative cases pulled from
  Section 4's buckets — a high-confidence correct call, a low-confidence
  call, and a misclassified case (FP or FN) from Section 1's confusion
  matrix.
- Interpretation cell: do the SHAP top drivers match what Stage 1/2 EDA
  already flagged as strong single-variable signals (e.g. `had_prior_dropout`,
  `Payment_Terms`)? Corroboration is expected; a mismatch is worth flagging,
  not smoothing over.

## Section 6 — final submission file

**The real gap to close first:** Stage 2's transforms (medians for
imputation, top-N category lists for one-hot encoding, the
`Payment_Terms`/`Client_Category` row-exclusion masks) were computed inline
against `df`/`df_model` — not captured as a reusable fit-on-train,
apply-to-any-df function. `Test_Data_No_Target.csv` (15,866 rows, same raw
columns as `Train_Data.csv` plus `Client_ID`, no target) needs the *exact
same fitted constants* applied, never recomputed from the test file itself
— recomputing medians/top-N categories on the test set would silently
produce a mismatched column set and train/test skew.

- Refactor the Stage 2 transform steps into a function (or a clearly
  ordered sequence with the fitted constants — medians, top-N category
  lists — captured as named variables) so it runs once on `df_model`
  (already done → `X`/`y`) and once on the raw test file.
- The Stage 2 row-exclusion rules (`PREPAID NONREFUNDABLE` +
  non-affiliated, `Client_Category == UNKNOWN`) apply to **training rows
  only**. Do not drop rows from the test file — every one of the 15,866
  `Client_ID`s needs an output row regardless of how those exclusion rules
  would have classified it.
- Once Part D's evaluation is done, refit the chosen model on the **full**
  labeled pool (`X_train` + `X_holdout`, all 53,767 rows) before predicting
  on the test file. The holdout's only job was the Part D evaluation above;
  the real submission should use every labeled row available, not the
  43,013-row train-only `.best_estimator_` from Part C. State this as an
  explicit separate fit, not a reuse of the Part C object.
- `predict_proba` on the transformed test matrix → `Drop_Probability`.
  Assemble `Client_ID` (from the raw test file, never dropped there the way
  it was dropped from `X`) + `Drop_Probability`, write to
  `Group_XX_Submission.csv` (XX = actual group number — not yet filled in).
- Sanity checks before writing: row count == 15,866, `Client_ID` uniqueness
  matches the source file, no NaN probabilities, all probabilities in
  [0, 1].

## Out of scope / open questions

- Group number for the output filename.
- Probability calibration (Platt/isotonic) — only worth building if Section
  4's reliability curve shows the chosen model is meaningfully miscalibrated;
  not built preemptively.

## Verification

- Confusion matrix/metrics computed only on `X_holdout` — never `X_train`
  or a CV fold (Part C already owns those numbers).
- Model-selection markdown cell cites Section 1's holdout table, not just
  Part C's CV ranking.
- SHAP background/explain sample sizes stated with a reason.
- Submission model is refit on train+holdout combined, confirmed distinct
  from the Part C `.best_estimator_` object.
- Test-file transform reuses train-fitted medians/category lists — grep the
  test-transform code for any `.median()`/`.value_counts()` call against
  the test dataframe itself (there should be none).
- Output CSV: exactly 2 columns (`Client_ID`, `Drop_Probability`), 15,866
  rows, no NaNs, values in [0, 1].
