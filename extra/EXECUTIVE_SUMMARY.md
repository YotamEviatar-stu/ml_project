# Executive Summary — Nova Academy Course-Dropout Prediction (Group 46)

**Goal:** predict `Dropped_Course` (binary) for Nova Academy's B2B enrollments, so at-risk clients can be flagged for intervention before they cancel.

**Data:** `Train_Data.csv` — 63,464 rows × 29 columns, baseline drop rate ~31%. A separate ungraded `Test_Data_No_Target.csv` (15,866 rows) is scored only at the end.

## Process

**1. EDA & cleaning (Part A).** Every one of the 29 variables was profiled individually (distribution, rate-vs-target, missingness) before any transform was applied. Key findings: 6 categorical columns were corrupted by stray symbols/casing (e.g. `Origin_Country` collapsed 721 → 154 raw strings once cleaned); `Students_Count` and `Practical_Hours` carried sentinel values (`9999`, `-5`/`5000`/`10000`) disguised as real data; `Physical_Course_Kits`, `Assigned_Lab_Config`, and `Welcome_Gift_Type` were flagged as either leakage (set after the outcome was known) or signal-free.

**2. Feature engineering (Part B).** Transforms were grouped by treatment type, not applied ad hoc per column: 8 binary presence/threshold flags (e.g. `had_prior_dropout`, `has_waited`), one-hot encoding with a "keep categories with n ≥ 30, fold the rest into `OTHER`" rule for high-cardinality identity columns (`Company_ID` 184 → 17 groups, `Agent_ID` 203 → 77, `Origin_Country` 154 → 8), median imputation for skewed numerics, and 5 dropped columns. Two row-level exclusions were applied (9,695 rows where `Payment_Terms`/`Enrollment_Type` combined to a 100%-deterministic drop pattern, too clean to trust as learnable signal; 2 rows with `Client_Category = UNKNOWN`). Final modeling matrix: 53,767 rows × 156 numeric columns, zero nulls.

**3. Modeling (Part C).** A stratified 80/20 holdout (10,754 rows) was carved out first and never touched during tuning. Three models were tuned via 5-fold CV `RandomizedSearchCV` on the remaining 43,013 rows:

| Model | CV AUC (±std) | Train AUC | Gap |
|---|---|---|---|
| Logistic Regression | 0.854 ± 0.005 | 0.857 | 0.003 |
| Random Forest | 0.922 ± 0.003 | 0.967 | 0.045 |
| XGBoost | 0.930 ± 0.003 | 0.987 | 0.057 |

XGBoost's gap was checked against `early_stopping_rounds` (independently converged to the same 0.930) and against a leakage audit (all leakage-risk columns confirmed absent from `X`) — the overfitting is real but bounded, not a sign of a data error.

**4. Evaluation (Part D).** On the untouched holdout, XGBoost confirmed its lead independently of the CV ranking:

| Model | Precision | Recall | F1 | Accuracy | Holdout AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.730 | 0.571 | 0.641 | 0.802 | 0.855 |
| Random Forest | 0.818 | 0.707 | 0.758 | 0.861 | 0.928 |
| **XGBoost** | **0.822** | **0.739** | **0.778** | **0.870** | **0.933** |

Recall was weighted more heavily than raw accuracy in model selection: a missed dropout (false negative) forfeits any chance to intervene, while a false positive only costs unnecessary outreach. XGBoost catches 74% of real dropouts (866 of 3,321 slip through, 8.1% of the holdout) at 82% precision on flagged cases.

**SHAP.** Top global drivers: `Origin_Country = PRT` (Portugal), `had_prior_dropout` (drop rate nearly triples after one prior dropout, corroborating the Part A finding), open support tickets, specific agent groups, and lead/booking time. A secondary check found `Origin_Country = PRT` is entangled with `Client_Category` (PRT students are 66% Big Tech/Traditional IT vs. 66% SaaS for non-PRT) — the model may be partly using country as a proxy for industry.


## Conclusions & what would improve the model further

- The central signal is real and interaction-driven, not a single leaked column: the linear baseline already reaches 0.854 AUC, and the ~0.08 lift from tree models reflects genuine feature interactions.
- Recall is the binding constraint for business value (8.1% of dropouts still missed) — worth revisiting the 0.5 threshold as a tunable business decision (Part D, Section 2) rather than a fixed default.
- The `Origin_Country`/`Client_Category` entanglement suggests an interaction feature or a fairness/robustness check before deployment, so the model isn't silently using nationality as a proxy.

