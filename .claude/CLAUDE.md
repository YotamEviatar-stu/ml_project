# CLAUDE.md

This file provides guidance to Claude Code when working in this repo (the Nova Academy `Dropped_Course` prediction project, Group 46). General communication style, cross-project skills, and memory conventions live in the global `~/.claude/CLAUDE.md` — this file covers only what's specific to this project's domain.

## Project

Predict `Dropped_Course` (binary) for Nova Academy's B2B enrollments so at-risk clients can be flagged before they cancel. `Train_Data.csv` — 63,464 rows × 29 columns, baseline drop rate ~31%. `Test_Data_No_Target.csv` (15,866 rows, ungraded) is scored only at the very end — never used for exploration, feature derivation, or tuning.

## Established findings — don't silently re-derive or contradict these

- **Data quality:** 6 categorical columns were corrupted by stray symbols/casing (`Origin_Country` collapsed 721 → 154 raw strings once cleaned). `Students_Count` and `Practical_Hours` carry sentinel values (`9999`, `-5`/`5000`/`10000`) disguised as real data — treat as missing, not as extreme-but-real.
- **Leakage:** `Physical_Course_Kits`, `Assigned_Lab_Config`, `Welcome_Gift_Type` are excluded — set after the outcome was known, or signal-free. Any new feature must be checked against this same leakage bar (was it knowable *before* the drop decision?) before being added.
- **Row exclusions:** 9,695 rows where `Payment_Terms`/`Enrollment_Type` combine to a 100%-deterministic drop pattern (too clean to trust as learnable signal) and 2 rows with `Client_Category = UNKNOWN` are dropped. Final modeling matrix: 53,767 rows × 156 numeric columns, zero nulls.
- **Encoding convention:** high-cardinality identity columns use "keep categories with n ≥ 30, fold the rest into `OTHER`" (`Company_ID` 184 → 17, `Agent_ID` 203 → 77, `Origin_Country` 154 → 8). Match this rule for any new categorical feature rather than inventing a different threshold.
- **Holdout discipline:** the stratified 80/20 holdout (10,754 rows) was carved out first and never touched during tuning; all CV (`RandomizedSearchCV`, 5-fold) runs on the remaining 43,013 rows only. Never report a training-fold or CV metric as if it were a holdout metric.
- **Current best model:** XGBoost, holdout AUC 0.933, precision 0.822, recall 0.739, F1 0.778. Recall is weighted over raw accuracy in model selection — a missed dropout (false negative) forfeits any chance to intervene; a false positive only costs unnecessary outreach.
- **Known fairness caveat:** `Origin_Country = PRT` is a top SHAP driver but is entangled with `Client_Category` (66% Big Tech/Traditional IT for PRT vs. 66% SaaS for non-PRT). Don't present PRT as a clean, independent driver without flagging this confound.

## Pipeline stages

Part A (EDA/cleaning) → Part B (feature engineering) → Part C (modeling/tuning) → Part D (evaluation/SHAP). See `plans/Part_A_plan.md` through `Part_D_plan.md` for the canonical per-stage checklist, and `extra/EXECUTIVE_SUMMARY.md` for the current end-to-end write-up. Work in `train.ipynb`.

## Skills

`/feature-engineering` (Stage 1–2 methodology/checklist), `/model-training` (Stage 3 — model training and hyperparameter tuning), `/ml-pipeline-review` (adversarial pre-submission audit — review only, not for making changes), `/ml-research` (domain literature/methodology research — checks `plans/`/`extra/` and established findings above before searching externally).

## Communication Style

Startle's numeric-example / p-value convention generalizes here as: any claim about a model or feature must cite the actual metric (AUC, precision/recall, SHAP value) — never a qualitative "this should help" without the number.
