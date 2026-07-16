# Stage 4 implementation runbook

**Scope: Sections 1-5 only (through SHAP). Do not implement Section 6
(submission file) — stop after Section 5's interpretation cell.** Section 6
needs a group number that isn't decided yet and is a separate, larger
refactor; it'll be its own follow-up task.

Implement `stage4_plan.md` in `train.ipynb`. Read `stage4_plan.md` and both
CLAUDE.md files (`/Users/yotameviatar/vs_code/.claude/CLAUDE.md` doesn't apply
to this repo — use `/Users/yotameviatar/vs_code/ml/CLAUDE.md` and
`.claude/skills/model-training/SKILL.md` for style) before writing anything.

## Technical gotcha — read this first

`train.ipynb` is ~14.8MB. Cell index 2 alone (the `ydata_profiling`
`ProfileReport` output) is ~13MB of embedded HTML. The `Read` tool errors out
on this file (exceeds its 25k-token cap) — which means `NotebookEdit` cannot
be used (it requires a successful prior `Read` of the same file).

**Insert/edit cells with a Python script using `nbformat` directly** (already
installed, v5.10.4, in `/Users/yotameviatar/vs_code/.venv`), e.g.:

```python
import nbformat as nbf
nb = nbf.read("train.ipynb", as_version=4)
# nb.cells is a list; find the target cell by nb.cells[i].id or by scanning
# source text; insert with nb.cells.insert(idx, nbf.v4.new_code_cell(source))
# or nbf.v4.new_markdown_cell(source)
nbf.write(nb, "train.ipynb")
```

To inspect the notebook's current structure/content without hitting the Read
limit, use `jupyter nbconvert --to script --stdout train.ipynb > /tmp/train_full.py`
and `grep`/`Read` that instead of the `.ipynb` directly. Cell *line numbers* in
that dump won't match the `.ipynb`'s cell list — locate cells by id or by
`grep`-ing distinctive source text via a small `json.load` script instead.

`shap` is already installed in the venv (added this session) — no action
needed for Section 5.

## Insertion point

Append after the **last cell**, id `c83f21f2`, a markdown cell titled "Part C
write-up — model training & tuning" (verify this is still the last cell —
`python3 -c "import json; nb=json.load(open('train.ipynb')); print(len(nb['cells']), nb['cells'][-1]['id'])"`).

Already defined and available to every new cell below (no re-derivation
needed): `X_train`/`y_train` (43,013 rows), `X_holdout`/`y_holdout` (10,754
rows, 3,321 positives), `fitted_models = {"LogisticRegression": logreg_best,
"RandomForest": rf_best, "XGBoost": xgb_best}`, `results`/`summary` (CV AUC
table), `RANDOM_STATE = 42`, `skf`, `cv_report()`.

## Section order (deviates from `stage4_plan.md`'s numbering)

The plan's Section 2 (threshold check) explicitly operates on "the model that
ends up chosen in Section 3" — but Section 3 is written after it in the doc.
Implement in dependency order instead, keeping the same substance:

1. **Section 1 — holdout confusion matrix & metrics** (plan lines 17-32).
   Loop `fitted_models`, `.predict_proba(X_holdout)`, threshold 0.5,
   `confusion_matrix` + precision/recall/F1/accuracy/holdout
   `roc_auc_score`. Stash each model's holdout probabilities in a dict —
   reused in steps 3-5, don't recompute. One comparison table: model ×
   TP/FP/FN/TN (counts **and** pct of holdout n, per notebook convention) ×
   precision × recall × F1 × accuracy × holdout AUC. Markdown before: FN
   (dropout unflagged, no intervention) vs. FP (flagged client who wouldn't
   have dropped, wasted outreach) — why recall > accuracy here. Markdown
   after: **do not write this until after the execution step below** — it
   must cite the real printed table, not guessed numbers.

2. **Model selection** (plan's Section 3, lines 44-49). Markdown-only,
   citing Section 1's *holdout* table specifically (not Part C's CV
   ranking). Plan expects XGBoost (CV AUC 0.930 vs RF 0.922 vs LR 0.854) but
   this must be confirmed against the holdout numbers, not assumed.

3. **Threshold check** (plan's Section 2, lines 34-42). PR curve or a few
   alternate thresholds, for the chosen model only. States explicitly this
   doesn't change the submission file (raw probabilities needed, not hard
   labels) — it's only to justify/flag whether 0.5 was reasonable for
   Section 1's confusion matrix.

4. **Section 4 — confidence deep-dive** (plan lines 51-64). `confidence =
   max(p, 1-p)` on the chosen model's holdout probabilities. Buckets: <0.6 /
   0.6-0.8 / >0.8 (or justify different cuts). Accuracy per bucket. Cross-tab
   low- vs. high-confidence groups against `Client_Category`/`Payment_Terms`
   — these aren't columns on `X_holdout` directly (it's one-hot encoded); use
   `X_holdout.index` to slice back into `df_model["client_category_clean"]`
   / `df_model["payment_terms_bucket"]`. Reliability curve: predicted-prob
   bucket vs. observed dropout rate.

5. **Section 5 — SHAP** (plan lines 66-80). `shap.TreeExplainer` (chosen
   model is RF or XGBoost, both tree-based — no need for `LinearExplainer`
   unless LR somehow wins). Background = sample of `X_train`; explain a
   capped subsample of `X_holdout` (state the cap, e.g. 2,000 rows, and why
   — runtime, not silent truncation). Global: beeswarm + mean |SHAP| bar.
   Local: waterfall/force plots for 3 cases from step 4's buckets — high-
   confidence correct, low-confidence, and a misclassified case from step
   1's confusion matrix. Interpretation: do top SHAP drivers match Stage 1/2
   EDA-flagged signals (`had_prior_dropout`, `Payment_Terms`, etc.)?

**Stop here.** Section 6 (submission file, plan lines 82-116) is explicitly
out of scope for this task — do not refactor the Stage 2 transforms, do not
touch `Test_Data_No_Target.csv`, do not refit on train+holdout. That's a
separate follow-up once the group number is known.

## Style conventions to mirror (see cells around `logreg_pipe`/`rf`/`xgb` for
the pattern)

- Markdown header + 1-2 sentence plain-language framing *before* each code
  cell (no jargon-first), then the code, then a markdown interpretation cell
  *after* using the code's own printed output.
- No inline code comments/docstrings (WIP phase — see both CLAUDE.md files).
  A bare list/constant needs no per-entry rationale in the file.
- Every printed mean/count table shows pct of total n alongside the count.
- Interpretation cells: short clipped bullets, one fact per bullet, decision
  line can stay hedged if genuinely open.
- Every number in an interpretation cell must come from that cell's own
  printed/plotted output — never typed from memory or copied from this
  runbook's estimates.

## Execution (single pass covers everything)

No kernel is attached during editing and there's no cached
model/data artifact on disk (checked: no `.pkl`/`.joblib` in the repo) — Part
D's interpretation numbers only exist after the *entire* notebook runs
top-to-bottom (Stage 1 EDA → Stage 2 features → Stage 3 tuning → Part D).
Write all code cells first (leave the "after execution" markdown
interpretation cells as short placeholders, or skip creating them until
after this step), then:

```bash
cd /Users/yotameviatar/vs_code/ml
source .venv/bin/activate  # or the project's venv path
jupyter nbconvert --to notebook --execute --inplace train.ipynb --ExecutePreprocessor.timeout=1800
```

This re-runs the `ProfileReport` (large but was already produced once
successfully) and all three `RandomizedSearchCV` fits — expect several
minutes total. Run it in the background and monitor rather than blocking.

After it completes, extract each new cell's actual printed output via a
small `json.load` script (not the `Read` tool, same token-limit problem) and
backfill the interpretation markdown cells' source text with those real
numbers using the same `nbformat` approach — this is a text edit to already-
written markdown cells, not a second full execute.

## Verification (from `stage4_plan.md`, lines 125-137)

- Confusion matrix/metrics computed only on `X_holdout`, never `X_train`/CV.
- Model-selection markdown cites Section 1's holdout table, not Part C's CV
  ranking.
- SHAP background/explain sample sizes stated with a reason.
- Notebook ends after Section 5's SHAP interpretation cell — no Section 6
  content (no transform refactor, no test-file predictions, no CSV write).
