# Stage 3 plan — model training & hyperparameter tuning (Part G)

Full methodology lives in `.claude/skills/model-training/SKILL.md`. This
file tracks build order and what's covered vs. pending.

## Prerequisite

Stage 2's final feature matrix (all Stage 1/2 decisions applied: cleaned
categoricals encoded, missing values imputed/flagged, outliers handled)
must exist as a single numeric `X`/`y` before any cell below runs. Not yet
assembled in `train.ipynb` as of this plan being written — check the
notebook before starting.

## Setup

- Add `xgboost` to `requirements.txt`.
- `X_train`/`y_train` = full `Train_Data.csv` after Stage 1/2 processing.
  `Test_Data_No_Target.csv` stays untouched until the very end.
- One shared `StratifiedKFold(n_splits=5, shuffle=True, random_state=...)`
  object reused across all three models' `RandomizedSearchCV` calls, so CV
  AUCs are comparable across models.

## Model build order

1. **Logistic Regression** — baseline. Search `C` (log-spaced,
   e.g. 1e-3–1e2) × `penalty` (`l1`/`l2`, with `solver="liblinear"` or
   `saga`). Needs feature scaling (`StandardScaler` in the pipeline) —
   the tree models don't.
2. **Random Forest** — bagged-tree comparison point. Search
   `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
3. **XGBoost** — required by the rubric, expected best performer on
   tabular data. Search `max_depth`, `learning_rate`, `n_estimators`,
   `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`. Use
   `early_stopping_rounds` against a held-out fold in addition to the
   random search.

Each model gets: 1 markdown cell (brief model explanation + hyperparameters
being tuned), 1 code cell (pipeline + `RandomizedSearchCV` fit + best
params + train/CV AUC printed), 1 markdown cell (interpretation — best
params found, train vs. CV AUC gap, bias-variance diagnosis, decision).

## Section — model comparison summary

One table: model × best hyperparameters × mean CV AUC ± std × train AUC ×
gap. This is the artifact Part D's evaluation section will build on, so
keep the fitted model objects (or their best params) accessible, not just
printed and discarded.

## Verification

- Run notebook top to bottom; every model cell executes without error and
  prints train AUC, mean±std CV AUC, and best hyperparameters.
- Confirm no cell reads `Test_Data_No_Target.csv` before the final summary
  section.
- Confirm any target-encoded Stage 2 feature is inside the CV
  pipeline/`ColumnTransformer`, not precomputed once on the full training
  set (see `model-training` skill's leakage guardrails section) — spot
  check by verifying the encoder's `fit` is called from within
  `cross_val_score`/`RandomizedSearchCV`, not before it.
