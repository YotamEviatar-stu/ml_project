---
name: model-training
description: Methodology and checklist for Stage 3 (Part G — model training and hyperparameter tuning) on the Nova Academy Dropped_Course project. Use when building a model, choosing a hyperparameter search space, running cross-validation, checking overfitting/bias-variance tradeoff, or writing up model comparisons in train.ipynb. Not for Stage 4 (Part D — evaluation/SHAP), which is a separate later skill.
---

# Model training & hyperparameter tuning — Nova Academy project (Stage 3 / Part G)

Stage 3 assumes Stage 1 (EDA/imputation) and Stage 2 (feature engineering/
encoding) already produced a clean, fully-numeric feature matrix in
`train.ipynb`. If Stage 2 hasn't assembled that final matrix yet (as of this
writing it hasn't — the notebook is still variable-by-variable Stage 1
passes), do that first; don't start fitting models against half-finished
columns. See `feature-engineering` skill for Stage 1/2.

**Rubric target (Part G, verbatim):** build at least 3 different models —
one must be from the XGBoost family — briefly explain each model and its
hyperparameters, tune each model's hyperparameters, watch the bias-variance
tradeoff to avoid overfitting, use regularization where appropriate, and
justify every decision made along the way.

**Model lineup for this project:** Logistic Regression, Random Forest,
XGBoost. This mixes three different inductive biases (linear/regularized,
bagged trees, boosted trees) so the model-comparison write-up in Part D has
something real to contrast, not three trees that behave alike.

**Primary metric: AUC** (ROC-AUC). Baseline drop rate is ~41.4%, so accuracy
alone is a weak metric — AUC is threshold-independent and is what Part D's
confusion-matrix/metric discussion will build on.

**`Test_Data_No_Target.csv` is the final holdout** — per the
`feature-engineering` skill's glossary note, never fit, tune, or select a
model against it. All CV and model comparison in this stage happens on
`Train_Data.csv` only.

## The non-negotiable rule for every model

Every model gets three things, written down in the notebook (markdown cell
next to the fitting code, not just said in chat):
1. **A brief explanation of the model** — one or two sentences on the
   mechanism (e.g. "logistic regression fits a linear decision boundary in
   log-odds space"), not a textbook chapter.
2. **Which hyperparameters were swept, and why those ones matter for
   bias-variance** — e.g. for a tree model, depth controls variance;
   for logistic regression, `C` controls the regularization strength.
3. **Train-score vs. CV-score gap** — the actual numbers from this
   dataset's run, not just "it doesn't look overfit." This is how the
   bias-variance tradeoff claim gets justified with evidence rather than
   asserted.

## Data leakage guardrails

- Any target encoding or smoothed-mean feature built in Stage 2 (see
  `feature-engineering` skill's smoothed target encoding section) must be
  refit **inside each CV fold** during Stage 3, not computed once on the
  full `Train_Data.csv` and then reused across folds — otherwise every fold
  peeks at its own validation rows' target values through the encoding,
  inflating CV AUC. Wrap it in a `sklearn` `Pipeline`/`ColumnTransformer` (or
  a custom `TransformerMixin`) so `cross_val_score`/`RandomizedSearchCV`
  refit it per fold automatically — don't precompute the encoded column
  once on the whole training set and feed that into CV.
- `Test_Data_No_Target.csv` is touched exactly once, at the very end, using
  the already-tuned model and any encoders fit on the **full** training set
  — never during CV or tuning.

## Validation strategy

- `StratifiedKFold` (5 folds is the standard default — enough to average
  out fold noise without becoming slow given the time constraint), stratified
  on `Dropped_Course` so each fold keeps the ~41.4% base rate. Use the same
  fold split (same `random_state`) across all three models so their CV AUCs
  are comparable apples-to-apples.
- Report **mean ± std AUC across folds** per model (per the decision to skip
  a formal paired test here — coursework-level rigor, not a paired
  Wilcoxon/t-test between models).
- Also report **train-fold AUC vs. validation-fold AUC** per model (not just
  the validation number) — this pair is what actually demonstrates the
  bias-variance point the rubric asks for. A model with train AUC 0.95 and
  CV AUC 0.71 is overfit even if 0.71 looks fine in isolation; a model with
  train AUC 0.74 and CV AUC 0.73 is well-calibrated even if the absolute
  number is lower.

## Hyperparameter search: RandomizedSearchCV

Given the time constraint, use `RandomizedSearchCV` (not `GridSearchCV`,
not Optuna) for all three models — it's the standard scikit-learn approach,
needs no new dependency beyond `scikit-learn`/`xgboost`, and is easy to
justify in a write-up ("sampled N random combinations from the search space
rather than exhaustively grid-searching, given the parameter space size and
time budget"). Fix `n_iter` (e.g. 20–30) and `cv=StratifiedKFold(5,
shuffle=True, random_state=...)`, scoring=`"roc_auc"`.

### Search space per model — the regularization/complexity knobs the rubric wants named

| Model | Key hyperparameters (bias-variance role) | Regularization knob |
|---|---|---|
| Logistic Regression | `C` (inverse regularization strength — low `C` = more regularization = simpler boundary), `penalty` (`l1`/`l2`) | `C`, `penalty` directly control the penalty term on coefficient size |
| Random Forest | `max_depth` (deeper = more variance), `min_samples_leaf` (higher = less variance), `n_estimators` (more trees reduces variance via averaging, doesn't add bias) | `max_depth`, `min_samples_leaf`, `max_features` act as implicit regularization by limiting how much any single tree can overfit |
| XGBoost | `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `reg_alpha`/`reg_lambda` | `reg_alpha` (L1) / `reg_lambda` (L2) are explicit penalty terms; `subsample`/`colsample_bytree` add stochastic regularization; low `learning_rate` + early stopping is the other standard overfitting guard |

For XGBoost specifically, use `eval_set` + `early_stopping_rounds` on a
held-out fold in addition to the `RandomizedSearchCV` sweep — this is the
standard, expected way to guard against overfitting for boosted trees and
directly demonstrates "avoided overfitting" for the write-up.

## Reading the bias-variance tradeoff from the numbers

Concrete pattern to watch for once results exist (fill in real numbers from
this dataset's run when writing the interpretation — don't leave this
table's example values in the actual write-up, they're illustrative only):

| Train AUC | CV AUC | Gap | Diagnosis |
|---|---|---|---|
| 0.99 | 0.72 | 0.27 | Overfit — model memorized training rows; increase regularization (lower `max_depth`, raise `min_samples_leaf`/`reg_lambda`, lower `C`) |
| 0.75 | 0.74 | 0.01 | Well-calibrated, possibly underfit if a competing model reaches materially higher CV AUC at a similar gap |
| 0.80 | 0.78 | 0.02 | Healthy — small expected gap from finite training data, not a red flag |

State the actual gap number for each of the 3 models in the write-up, not
just "looks fine."

## Interpretation write-up style

Same convention as Stage 1/2: one bullet per plot/table in plot order, each
bullet a single clipped fact, followed by a short **Decision** paragraph in
plain "we" language. This write-up is for the professor — don't explain
generic concepts (what AUC means, what overfitting is) unless a specific
number needs interpreting; state the number and its consequence directly.

## Dependencies

`xgboost` is not yet in `requirements.txt` — add it before this stage's code
runs. No other new dependency needed (`RandomizedSearchCV` and
`StratifiedKFold` are both in `scikit-learn`, already present).

## Reference

`Part_C_plan.md` — model build order, search spaces, and verification
checklist. `feature-engineering` skill — Stage 1/2 methodology this stage's
feature matrix depends on.
