# PREPAID NONREFUNDABLE slice — impact audit

Sandbox re-run of the notebook's tuned XGBoost baseline (cell 123 best params) under two scenarios. The raw `course_start_days` index was replaced with seasonal components (`start_month`, `start_dayofweek`) in both scenarios to avoid tree extrapolation on the temporal split.

- Slice definition: `payment_terms_clean == 'PREPAID NONREFUNDABLE'` and `enrollment_type_clean != 'AFFILIATED ADMISSION'` — **9,695 train rows**, drop rate 100.00%.
- Same slice in `Test_Data_No_Target.csv`: **1,987 rows** (12.52% of the 15,866 test rows).
- Temporal validation window: last 3 months of `Course_Start_Date` (cutoff 2017-01-26).

| Scenario | Modeling rows | Drop rate | Split | Train n | Val n | Val drop rate | ROC-AUC | FN @0.5 | FP @0.5 |
|---|---|---|---|---|---|---|---|---|---|
| Iteration 0 (Baseline: slice excluded) | 53,767 | 30.88% | A — random 80/20 | 43,013 | 10,754 | 30.88% | 0.9284 | 905 | 528 |
| Iteration 0 (Baseline: slice excluded) | 53,767 | 30.88% | B — temporal | 45,615 | 8,152 | 31.26% | 0.8516 | 1,170 | 651 |
| Iteration 1 (Reinstated: slice included) | 63,462 | 41.44% | A — random 80/20 | 50,769 | 12,693 | 41.44% | 0.9543 | 942 | 532 |
| Iteration 1 (Reinstated: slice included) | 63,462 | 41.44% | B — temporal | 53,942 | 9,520 | 41.13% | 0.9042 | 1,192 | 648 |

## Deltas (Reinstated minus Baseline)

| Split | Δ ROC-AUC | Δ FN | Δ FP |
|---|---|---|---|
| A — random 80/20 | +0.0259 | +37 | +4 |
| B — temporal | +0.0526 | +22 | -3 |

*Note: the two scenarios are scored on different validation populations (the reinstated one contains the near-deterministic slice), so AUC/FN/FP shifts reflect both model change and population change.*
