# Stage 3 plan — model training & hyperparameter tuning (Part G)

Full methodology lives in `.claude/skills/model-training/SKILL.md`. This
file tracks build order and what's covered vs. pending.

## Prerequisite

Stage 2's final feature matrix (all Stage 1/2 decisions applied: cleaned
categoricals encoded, missing values imputed/flagged, outliers handled)
must exist as a single numeric `X`/`y` before any cell below runs. ✅ Met —
`train.ipynb` builds `X`/`y` at the end of the Stage 2 section: (53767,
156), zero nulls, all-numeric dtypes, row count reconciled.

## Setup

- Add `xgboost` to `requirements.txt`.
- **Held-out test split (new, do this first, before anything else below).**
  Part D needs a confusion matrix + metrics, which require known labels —
  `Test_Data_No_Target.csv` has none, so it cannot serve that purpose.
  Carve a stratified train/test split out of `Train_Data.csv`'s `X`/`y`
  (`train_test_split(..., stratify=y, random_state=...)`) **before** any CV
  or hyperparameter search runs. Pick the split fraction by reasoning about
  what's actually in front of you — dataset size (53,767 rows) and the
  class balance of `Dropped_Course` — and justify the choice in a markdown
  cell (e.g. "20% holdout = ~10.7k rows, still gives a low-variance test
  AUC estimate at this N; smaller if the positive class is rare enough that
  20% would leave too few positive cases in the holdout"). This holdout is
  never touched again until Part D.
- `X_train`/`y_train` = the train-side portion of that split only. All CV
  folds and hyperparameter search below operate on this portion.
  `Test_Data_No_Target.csv` separately stays untouched until Stage 4 proper
  (final predictions on the ungraded file) — it is unrelated to the Part D
  holdout above.
- One shared `StratifiedKFold(n_splits=5, shuffle=True, random_state=...)`
  object, built on the train-side portion only, reused across all three
  models' `RandomizedSearchCV` calls, so CV AUCs are comparable across
  models.

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
- Confirm the Part D holdout split happens first, before any CV or search
  cell, and that no model-fitting or tuning cell below it ever reads the
  holdout rows.
- Confirm no cell reads `Test_Data_No_Target.csv` before the final summary
  section.
- No target-encoded features exist in Stage 2's `X` (the original plan for
  smoothed target encoding was replaced with count-based top-N/`OTHER`
  thresholding on `Origin_Country`/`Company_ID`/`Agent_ID`/
  `Requested_Lab_Config`, which never reads `Dropped_Course`) — so there is
  no CV-fold-refitting requirement to check here.
- AUC values get read against the standard interpretation scale (Hosmer &
  Lemeshow): 0.5 none, 0.6–0.7 poor, 0.7–0.8 acceptable, 0.8–0.9 excellent,
  0.9–1.0 outstanding — used as an interpretive anchor and as a leakage
  trip-wire (>0.90 on this kind of tabular business data warrants a
  leakage re-check, not an uncritical win), not as a hard pass/fail target.
  Random Forest and XGBoost should each beat the Logistic Regression
  baseline's CV AUC by a real margin, not just land in a higher bucket by
  chance.
