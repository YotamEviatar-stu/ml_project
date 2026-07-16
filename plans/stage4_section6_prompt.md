# Stage 4, Section 6 — submission file — prompt for a fresh session

Scope: implement Section 6 of `stage4_plan.md` (lines 82–116) in `train.ipynb`
— the final `Group_XX_Submission.csv`. Sections 1–5 (Part D: holdout metrics,
threshold check, model selection, confidence deep-dive, SHAP) were done in
an earlier session per `stage4_implementation.md`; this session picks up
after them and does not redo them.

## Gate — confirm Part D is actually finished before starting

The notebook's last cell (id `e6d0e407`, a markdown cell) was left as a
placeholder ending in *"interpretation pending — filled in after full
notebook execution"*. Check it first:

```bash
python3 -c "
import json
nb = json.load(open('train.ipynb'))
print(''.join(nb['cells'][-1]['source'])[-300:])
"
```

- If it still contains "interpretation pending", the notebook has not been
  executed top-to-bottom yet. Run it first (see `stage4_implementation.md`'s
  Execution section: `jupyter nbconvert --to notebook --execute --inplace
  train.ipynb --ExecutePreprocessor.timeout=1800`, in the background), then
  backfill that placeholder and any other pending interpretation cells with
  the real printed output, before touching Section 6.
- Read the Section 3 (model-selection) markdown cell's actual text to get
  the **chosen model** — don't assume XGBoost. The plan expects it (CV AUC
  0.930 vs RF 0.922 vs LR 0.854) but Section 3 explicitly requires
  confirmation against the *holdout* table, and that confirmation only
  exists once real execution has happened.

## Read before writing anything

`stage4_plan.md` Section 6 (lines 82–116) and Verification (lines 130–137),
both `CLAUDE.md` files per the split noted in `stage4_implementation.md`
(the `ml` repo's own `CLAUDE.md`, not the parent one), `.claude/skills/
model-training/SKILL.md`.

## Same tooling constraint as last time

`train.ipynb` is ~14.8MB — `Read`/`NotebookEdit` will fail on it. Edit with
`nbformat` directly; inspect via `jupyter nbconvert --to script --stdout
train.ipynb > /tmp/dump.py` and grep that instead of reading the `.ipynb`.
Locate cells by `id`, not by line number in the dump (they don't correspond).

## Where the pieces you need to reuse already live (already located this session)

All of these are in `df`/`df_model`-scoped code cells, computed only from
training data — Section 6's whole job is capturing them as named constants
and replaying them against `Test_Data_No_Target.csv` verbatim, never
recomputing:

- **Cell id `99243cf0`** (idx 102) — the two row-exclusion masks: `is_prepaid
  & ~affiliated` (`PREPAID NONREFUNDABLE` + non-affiliated, n=9,695) and
  `exclude_client_category` (`Client_Category == UNKNOWN`, n=2). These apply
  to **training rows only** — do not drop any of the 15,866 test rows.
- **Cell id `9cd356e3`** (idx 103) — imputation medians: `df_model
  ["Registration_Days_Before"].median()`, `df_model["students_count_clean"]
  .median()`, `df_model["practical_hours_clean"].median()`,
  `daily_tuition_clean.median()`. Capture these four numbers as named
  constants; apply the same numbers (not test-recomputed medians) to fill
  the corresponding test columns.
- **Cell id `dabbae9e`** (idx 105) — all the top-N / threshold category
  lists, each computed from `df` (full training data), each needs capturing
  as a named constant and reusing on the test file's raw columns:
  - `top_lab_configs` — top 6 `Requested_Lab_Config` values by count.
  - `top_countries` — `country_clean` categories at ≥4% of rows.
  - `big_companies` — `Company_ID` values with count ≥30.
  - `big_agents` — `Agent_ID` values with count ≥30.
  Also in this cell: `one_hot_df = pd.get_dummies(...)` over
  `client_category`, `enrollment_type`, `submission_source`,
  `catering_package`, `payment_terms_bucket` (already-cleaned Stage 1
  categorical columns). Test data must produce dummy columns aligned to
  **train's** column set — after building the test one-hot frame, reindex it
  to `one_hot_df.columns` (fill missing with 0) rather than trusting
  `pd.get_dummies` on the test file alone to produce the same columns.
- **Cell id `7debea49`** (idx 106) — `X = pd.concat([numeric_df,
  one_hot_df], axis=1)`, `y = df_model[TARGET]`. The test feature matrix
  must match `X`'s exact column set and order before calling `predict_proba`.
- **Cell id `35a01bad`** (idx 111) — `RANDOM_STATE = 42`, reuse for the
  Section 6 refit (not a new search — see next section).
- **Cell id `afda20d9`** (idx 113) — `X_train, X_holdout, y_train,
  y_holdout = train_test_split(...)`. Section 6 needs `X_train` and
  `X_holdout` concatenated back into the full labeled pool (53,767 rows).
- **Cell id `c8c8167f`** (idx 126) — `fitted_models = {...}`, the Part C
  tuned estimators. Pull the chosen model's hyperparameters from here (e.g.
  `fitted_models["XGBoost"].get_params()`), not to reuse the object itself.

## What "refit on the full pool" means here

Do not reuse the Part C `fitted_models[...]` object for the submission — it
was only ever trained on the 43,013-row `X_train`, deliberately never shown
`X_holdout`. Clone a fresh, untrained estimator with the **same
hyperparameters** the chosen model landed on in Part C (same architecture,
same tuned settings — no new search), then call `.fit()` on it once, on
`pd.concat([X_train, X_holdout])` / `pd.concat([y_train, y_holdout])` (all
53,767 rows). This is a distinct object from the Part C one; keep both
visible in the notebook rather than overwriting `fitted_models[...]`.

## Build steps (mirrors plan lines 94–115)

1. Load `Test_Data_No_Target.csv` (expect 15,866 rows, columns = raw
   `Train_Data.csv` columns minus `Dropped_Course`, plus `Client_ID`).
2. Apply the *same* Stage 1 text-cleaning + Stage 2 transform steps used for
   `df`/`df_model`, but with every fitted constant (medians, top-N lists,
   `one_hot_df.columns`) taken from the training-side variables above — no
   `.median()`/`.value_counts()`/`pd.get_dummies()` computed fresh against
   the test file. Do **not** apply the two row-exclusion masks.
3. Assemble the test feature matrix with columns aligned to `X`.
4. Refit as described above; `predict_proba` on the test matrix →
   `Drop_Probability` (positive-class probability column, not a hard 0/1
   label).
5. Assemble `Client_ID` (from the raw test file — it was dropped from `X`,
   not from the loaded test dataframe) + `Drop_Probability`.
6. Write `Group_XX_Submission.csv` — leave `XX` as a literal placeholder in
   the filename/code (group number not yet assigned); flag this clearly
   rather than guessing a number.

## Sanity checks before finishing (plan lines 113–115, 136–137)

- Row count == 15,866.
- `Client_ID` uniqueness matches the source file.
- No NaN probabilities.
- All probabilities in [0, 1].
- Grep your own test-transform code for `.median()`, `.value_counts()`,
  `pd.get_dummies(` applied to the test dataframe itself — there should be
  none; every one of those should trace back to a training-side constant
  computed earlier in the notebook.

## Out of scope for this session

- Deciding the group number, or renaming the notebook to
  `Group_XX_Notebook.ipynb`.
- The header cell with group number + submitters' names/ID numbers.
- Probability calibration — only revisit if Section 4's reliability curve
  (already done in Part D) showed meaningful miscalibration.
