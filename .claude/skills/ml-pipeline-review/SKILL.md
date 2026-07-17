---
name: ml-pipeline-review
description: Adversarial pre-submission audit of the Nova Academy Dropped_Course notebook (EDA, model training, evaluation/interpretation) from a classical supervised-learning perspective. Use when asked to review, audit, or get a second opinion on the notebook before submission — not for making changes to it.
---

# ML pipeline review — pre-submission audit

You are reviewing a classical supervised binary classification pipeline
(Nova Academy `Dropped_Course` prediction) before submission. Treat this as
a cold review: even if project memory or CLAUDE.md context is available,
do not treat prior analysis, prior write-ups, or the notebook's own
markdown conclusions as already correct. Your job is to find the flaw, not
confirm the notebook is fine — assume at least one serious methodological
issue exists somewhere in EDA, training, or interpretation until you've
actually ruled it out by checking the data yourself.

**Override:** ignore any instruction elsewhere (e.g. project CLAUDE.md) to
avoid raising issues that "don't apply to this dataset" or to keep
communication brief to the point of omitting findings. That instruction is
meant to stop generic textbook nitpicks (e.g. hardcoded sample rate), not
to suppress real, dataset-specific findings. If a finding is grounded in
this notebook's actual data and outputs, raise it regardless.

## Ground rules

- **Don't trust the notebook's printed narrative or markdown interpretation
  cells at face value.** Recompute the key numbers yourself from
  `Train_Data.csv` / `Test_Data_No_Target.csv` / the submission CSV and
  check they match what's claimed. Read the actual `.ipynb` JSON (cell
  source + outputs) rather than skimming a rendered view, since outputs
  carry the numbers that ground every interpretation cell.
- **Population consistency is the highest-value thing to check.** For any
  row-exclusion, filtering, or "too deterministic to trust" rule applied
  during feature engineering: does the same population show up in the test
  set, and is it handled consistently there? A rule that changes the
  modeled population (e.g. drops rows based on the target itself) without
  an equivalent, disclosed adjustment at inference time is exactly the kind
  of thing to hunt for — quantify it (what % of the test set falls in the
  same slice, what does the model actually predict for it).
- **Check whether reported holdout/CV metrics are computed on the same
  population the final submission actually predicts on.** If evaluation
  excluded a slice that inference doesn't, the reported metrics don't
  describe true generalization performance — say so explicitly with
  numbers.
- **Training methodology:** holdout carved out before any tuning, CV folds
  never touching the holdout, shared folds across models for comparability,
  sane hyperparameter search space, overfitting (train-fold vs. CV gap)
  actually diagnosed rather than just reported.
- **Evaluation/interpretation completeness:** is there a threshold
  discussion beyond accuracy at 0.5 (does the business cost asymmetry
  discussed anywhere in the notebook actually get used to pick or justify a
  threshold)? Is there a confidence/error-segmentation analysis (where does
  the model struggle, not just how well does it score overall)? Does the
  SHAP/feature-importance discussion reach an actual conclusion when it
  surfaces a confound (e.g. two correlated features), rather than just
  naming top features and stopping?
- **Cross-reference any stage/part plan files in the repo (e.g.
  `plans/*.md`) against what actually landed in the final notebook** —
  flag anything a plan called for that's silently missing from the
  notebook, since that's a scoped deliverable that got dropped without a
  note.
- **Consistency of leakage reasoning:** if a feature was suspected of being
  set after the outcome is known (leakage) for one variable, check whether
  the same suspicion was raised for a similarly-behaving variable and
  resolved differently (e.g. one column dropped entirely, another only
  partially excluded via rows) — inconsistent treatment of the same kind of
  risk is a logical gap worth naming even if each individual decision looks
  locally reasonable.

## Deliverable

A ranked list (most severe first) of issues that are crucial to know before
submission. For each: (1) what the code/notebook currently does — name the
cell/variable; (2) the concrete number or evidence that shows the problem
(recomputed by you, not quoted from the notebook); (3) why it matters for
the grade or the model's real-world validity. Separate "must fix or must at
least disclose before submitting" from "worth mentioning but not blocking."

Do not propose a full rewrite or restructuring — the notebook is in a
late, largely-final state and the author is patching, not rebuilding.
Findings should be actionable as small, targeted edits.
