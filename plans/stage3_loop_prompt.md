# Stage 3 autonomous run — prompt to paste into a fresh session

You are running Stage 3 (course Part C: model training & hyperparameter
tuning) for the Nova Academy `Dropped_Course` prediction project, in
`train.ipynb`. Stage 1 (EDA) and Stage 2 (feature engineering) are already
done and stable in the notebook — do not re-review or second-guess them.
Budget: run autonomously for up to ~2 hours, iterating like a senior ML
engineer with visible reasoning at every major step, not black-box output.

## Before writing any code

Read, in full: `stage3_plan.md`, `.claude/skills/model-training/SKILL.md`,
`CLAUDE.md`. Skim `.claude/skills/feature-engineering/SKILL.md` if you need
to confirm how a Stage 2 column was built. Do not start coding until
you've read `stage3_plan.md` end to end — it has just been updated with a
held-out test split requirement that did not exist in earlier drafts of
this plan; make sure you're working from the current version on disk, not
from memory of an older version.

## Hard constraints

- No explanatory comments, docstrings, or per-entry justification strings
  in code cells — this project is WIP, per `CLAUDE.md`. Put reasoning in
  markdown cells instead.
- No kernel is attached while you edit — after each notebook edit, execute
  it for real (e.g. `jupyter nbconvert --to notebook --execute --inplace
  train.ipynb`) so printed numbers are genuine, not typed from memory.
  Every number in a markdown cell must come from a code cell's own printed
  output.
- Never touch `Test_Data_No_Target.csv`. It has no `Dropped_Course` column
  and plays no role in Stage 3 or in Part D's confusion matrix — it is
  strictly a final-predictions file for later, out of scope here.
- Add `xgboost` to `requirements.txt` if it's not already there.

## Step 0 — the held-out test split (do this first, in its own cell(s))

Part D needs a confusion matrix and metrics, which require known labels.
`Test_Data_No_Target.csv` has none, so whatever data Part D scores against
must be a labeled slice carved out of `Train_Data.csv`'s `X`/`y` — reserved
before any CV or hyperparameter search touches it.

- Do a stratified `train_test_split` on `X`/`y`. Reason about the split
  fraction yourself from what you actually observe: the row count after
  Stage 2 (~53,767) and the class balance of `Dropped_Course`. Justify the
  number you pick in a markdown cell with the actual counts (e.g. "X% →
  Y positive cases in the holdout, still a low-variance AUC estimate at
  this N"). Don't default to 80/20 without checking whether the minority
  class has enough rows in the holdout to be meaningful — adjust if not.
- Everything in the rest of Stage 3 (CV, `RandomizedSearchCV`, all three
  models) operates only on the train-side portion. The held-out portion is
  not opened again until Part D — if you're not implementing Part D in
  this run, still create the split and set the holdout aside untouched, so
  whoever runs Part D next has it available and knows it was never leaked
  into tuning.

## Step 1 — shared CV

One `StratifiedKFold(n_splits=5, shuffle=True, random_state=...)` object,
built on the train-side portion only, reused across all three models so
their CV AUCs are comparable.

## Step 2 — three models, in this order

1. **Logistic Regression** — baseline. `StandardScaler` in the pipeline
   (tree models below don't need it). Search `C` (log-spaced, ~1e-3–1e2) ×
   `penalty` (`l1`/`l2`, `solver="liblinear"` or `saga`).
2. **Random Forest** — bagged-tree comparison point. Search `n_estimators`,
   `max_depth`, `min_samples_leaf`, `max_features`.
3. **XGBoost** — expected best performer on tabular data. Search
   `max_depth`, `learning_rate`, `n_estimators`, `subsample`,
   `colsample_bytree`, `reg_alpha`, `reg_lambda`, plus
   `early_stopping_rounds` against a held-out fold in addition to the
   random search.

For each model: 1 markdown cell (what the model is, which hyperparameters
are being tuned and why), 1 code cell (pipeline + `RandomizedSearchCV` fit
+ best params + train AUC + mean±std CV AUC, all printed), 1 markdown cell
(interpretation: best params found, train-vs-CV AUC gap, explicit
bias-variance diagnosis, a decision on whether to iterate further).

## Iterate per model — self-derived stopping, not a fixed target

There is no fixed target accuracy or AUC to hit — none exists anywhere in
this project's plan, and picking an arbitrary one without a baseline to
justify it would not be statistically defensible. Instead:

- Read AUC values against the standard interpretation scale (Hosmer &
  Lemeshow): 0.5 none, 0.6–0.7 poor, 0.7–0.8 acceptable, 0.8–0.9 excellent,
  0.9–1.0 outstanding. Use this as an interpretive anchor in your write-up
  ("0.78 → acceptable"), not as a pass/fail gate.
- Treat >0.90 CV AUC on this kind of tabular business data as a trip-wire,
  not a win — re-check for leakage (e.g. re-verify the top-N/OTHER
  category thresholds were fit correctly, re-confirm no dropped-for-
  leakage column like `Physical_Course_Kits`/`Assigned_Lab_Config`
  resurfaced) before reporting a number that high uncritically.
  Random Forest and XGBoost should each beat the Logistic Regression
  baseline's CV AUC by a real, explainable margin — not just land in a
  higher bucket by chance.
- Stop tuning a given model when: (a) the train-vs-CV AUC gap has
  stabilized at a small, defensible value (not shrinking further with more
  search iterations) — that's your bias-variance read — and (b) further
  hyperparameter changes stop improving mean CV AUC beyond what the
  fold-to-fold std would explain as noise. State explicitly, in the
  interpretation cell, which of these two conditions triggered the stop.
- For XGBoost specifically: use `early_stopping_rounds` as a direct,
  built-in underfitting/overfitting control (stop boosting when validation
  loss stops improving), not just a random-search dimension — explain in
  the markdown cell how many rounds it actually stopped at and why that's
  evidence against overfitting.

## Section — model comparison summary

One table: model × best hyperparameters × mean CV AUC ± std × train AUC ×
gap. Keep each `RandomizedSearchCV`'s `.best_estimator_` (already refit on
the full train-side pool with the winning hyperparameters, since
`refit=True` is the default) as its own named variable — not just the
printed best-params text. Part D will call `.predict()`/`.predict_proba()`
on these directly; do not make it refit from params alone.

## What Part D actually needs from you — do not build it, just leave it ready

Part D (course Part D, this project's Stage 4) is **out of scope for this
run** — do not build the confusion matrix, metrics, or SHAP analysis
yourself, and do not add cells for them. But leave the right foundation so
that whoever runs Part D next doesn't have to re-do or work around
anything from Stage 3. Concretely, Part D will need, and you should make
sure exist as clean, named artifacts by the end of this run:

- The Step 0 holdout (`X`/`y`), completely untouched by any fitting or
  tuning above — this is what the confusion matrix gets built from.
- Each model's fitted `.best_estimator_` object (per the point above) —
  Part D scores these against the holdout and picks one for the SHAP
  deep-dive.
- `X` (both the train-side pool and the holdout) kept as a `pandas.DataFrame`
  with real column names all the way through — not converted to a bare
  numpy array inside a pipeline step. Part D's SHAP plots need to attribute
  importance to actual column names (e.g. `Origin_Country_PRT`,
  `has_prior_dropout`), not `feature_47`. If any pipeline step here would
  silently drop column names (e.g. `StandardScaler` inside a plain
  `Pipeline`), make sure the DataFrame version survives alongside it (e.g.
  via `set_output(transform="pandas")` or an equivalent) so Part D isn't
  stuck reverse-engineering the column mapping.

Do not go further than this — no confusion matrix, no precision/recall,
no SHAP, no "chosen model" decision. That's Part D's job, done in a
separate run.

## Tools/skills to use proactively

Use the `model-training` skill's guidance throughout — don't improvise a
different tuning methodology. Use any project dataviz skill if one exists
for plots (train/CV AUC bars, hyperparameter search traces). Only fall
back to WebSearch/WebFetch if you need to look up an XGBoost parameter's
exact semantics you're unsure of — don't use it to look up "what's a good
AUC," that's already answered above.

## Deliverable — two separate reports at the end

1. **To me (the user), directly in your final message.** Candid and
   technical: what you tried per model, what worked/didn't, actual
   numbers, any leakage trip-wires you checked and what you found, and
   your honest assessment of which model you'd recommend and why.
2. **A markdown cell (or cells) in the notebook, written for the
   professor.** Same project write-up conventions as the rest of
   `train.ipynb`: numbers only from printed cell output, no basic-stats
   explanations (the audience already knows what CV/AUC/regularization
   are), short clipped bullets over dense prose. This is the raw material
   Stage 4's "systematic solution" narrative will build on — write it so
   it stands on its own as the Part C section of that story, not as notes
   to yourself.
