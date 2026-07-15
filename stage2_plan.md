# Stage 2 plan — feature engineering & assembly (step-by-step guidance)

Methodology lives in `.claude/skills/feature-engineering/SKILL.md` (Stage 2
section). This file is the execution plan: what order to build in, the full
Stage 1 → Stage 2 traceability table, and the open items that need a call
before any code runs.

## Step 0 — resolved

Both previously-open items are now settled:

1. **`Payment_Terms` × `Enrollment_Type` interaction** (cell 71). Resolved
   as a **row-level exclusion**, not a column drop: rows where
   `Payment_Terms` = `PREPAID NONREFUNDABLE` **and** `Enrollment_Type` is
   not `AFFILIATED ADMISSION` (n=9,695, ~15% of rows, 100% drop rate) are
   excluded from the training data entirely. The rest of `Payment_Terms`
   (`PAY UPON START`, the affiliated slice of `PREPAID NONREFUNDABLE`,
   `OTHER`) stays in as a normal 3-way one-hot group. Trade-off worth a
   line in the Stage 2 write-up: if this pattern is genuine signal rather
   than leakage, this exclusion sacrifices a near-perfect predictor for
   15% of rows — Stage 1 chose caution over that upside, carried forward
   as-is here.
2. **`Prev_Course_Dropouts` / `Prev_Course_Attended` encoding shape** —
   confirmed **binary**: `had_prior_dropout` (0 vs. 1+) and
   `attended_before` (0 vs. 1+), discarding the sparse ≥2/≥61-distinct
   tails, consistent with `Waiting_List_Days`/`Observers_Count`/
   `Pre_Course_Supports_Tickets`.

## Step 1 — traceability table (write before any transform code)

One markdown cell at the top of the Stage 2 section in `train.ipynb`,
built from this table. Every Stage 2 transform must trace to a specific
Stage 1 "Decision:" cell — nothing here is a fresh judgment call.

| Variable | Stage 1 decision | Stage 2 transform |
|---|---|---|
| `Prev_Course_Dropouts` | informative only 0 vs. 1, n=196 tail too small | binary flag `had_prior_dropout` |
| `Prev_Course_Attended` | informative only 0 vs. 1+ | binary flag `attended_before` |
| `Registration_Days_Before` | keep numeric; median fill for 4.2% missing; tail noted, no action | numeric passthrough + median impute |
| `Waiting_List_Days` | keep numeric + add `has_waited` (0 vs. 1+) flag | numeric passthrough + binary flag |
| `Registration_Changes` | group `0` / `1` / `2+` | 3-level categorical → one-hot |
| `Pre_Course_Supports_Tickets` | split `0` vs. `1+` | binary flag |
| `Professionals_Count` | keep as-is, 5 values, non-monotonic signal | numeric passthrough (no bucketing) |
| `Students_Count` | recode `9999`→NaN, median-fill (median=0) | numeric passthrough + sentinel recode + median impute |
| `Observers_Count` | `0` vs. `1+` binary | binary flag |
| `Practical_Hours` | recode sentinels (`-5`/`-1`/`5000`/`10000`, n=117)→NaN; median fill in Stage 3, no missing-indicator needed | numeric passthrough + sentinel recode |
| `Theory_Hours` | keep numeric, expect weak signal, flag 41h tail (no action) | numeric passthrough |
| `Daily_Tuition_Cost` | median fill; drop `5400` outlier value; add `is_zero_cost` flag | numeric passthrough + outlier recode + binary flag |
| `Physical_Course_Kits` | **drop** — value set after outcome known (leakage) | excluded |
| `Returning_Client` | keep as-is | numeric/binary passthrough |
| `Welcome_Gift_Type` | **drop** — no signal | excluded |
| `Requested_Lab_Config` | one-hot 6 substantial categories + `OTHER` (tiny n) + `MISSING`; hold mismatch feature | one-hot, 8 columns |
| `Assigned_Lab_Config` | **drop** — mismatch reflects attendance, not predicts it (leakage) | excluded |
| `Course_Start_Date` | convert to numeric (days from start); two low-n extreme months not treated as outliers | numeric (days-from-min-date) |
| `Origin_Country` | one-hot top 6 by % (`PRT`,`FRA`,`DEU`,`ESP`,`GBR`,`ITA`) + `OTHER` + `MISSING` | one-hot, 8 columns |
| `Client_Category` | drop `UNKNOWN` (n=2); one-hot remaining 7 | one-hot, 7 columns |
| `Payment_Terms` | 3-way group `PAY UPON START`/`PREPAID NONREFUNDABLE`/`OTHER`; exclude the `PREPAID NONREFUNDABLE`+non-affiliated slice (n=9,695) as too deterministic to trust | one-hot 3 groups + row-level exclusion of the n=9,695 slice |
| `Enrollment_Type` | one-hot all 5 categories | one-hot, 5 columns |
| `Submission_Source` | one-hot all 6 categories | one-hot, 6 columns |
| `Catering_Package` | one-hot all 5 categories | one-hot, 5 columns |
| `Lanyard_Color` | **drop** — no signal | excluded |
| `Company_ID` | `has_company_id` flag; one-hot the 16 companies with n≥30, fold remaining 168 into `OTHER` | binary flag + one-hot (17 columns) |
| `Client_ID` | **drop** — pure row identifier | excluded |
| `Agent_ID` | `has_agent_id` flag; one-hot the 76 agents with n≥30, fold remaining 127 into `OTHER` | binary flag + one-hot (77 columns) |

## Step 2 — build transform code, grouped by treatment type (not by column)

One cell per group, not per variable — keeps the notebook systematic and
short enough to review as a pipeline rather than 26 near-duplicate cells:

1. **Drop list** — `Client_ID`, `Welcome_Gift_Type`, `Lanyard_Color`,
   `Physical_Course_Kits`, `Assigned_Lab_Config` (column drops), plus a
   **row-level exclusion**: `PREPAID NONREFUNDABLE` + non-affiliated
   `Payment_Terms`/`Enrollment_Type` rows (n=9,695, see Step 0.1).
2. **Sentinel/outlier recodes** — `Students_Count` (`9999`→NaN),
   `Practical_Hours` (implausible values→NaN), `Daily_Tuition_Cost`
   (`5400`→NaN or flagged).
3. **Median imputation** — `Registration_Days_Before`, `Students_Count`,
   `Practical_Hours`, `Daily_Tuition_Cost` (after their recodes above).
4. **Binary flags** — `has_waited`, `Observers_Count`,
   `Pre_Course_Supports_Tickets`, `is_zero_cost`, `has_company_id`,
   `has_agent_id`, and the two Step-0.2 flags once resolved.
5. **One-hot block** — `Registration_Changes` (3-level), `Requested_Lab_Config`,
   `Client_Category`, `Enrollment_Type`, `Submission_Source`,
   `Catering_Package`, `Payment_Terms` (3-way group, after the row-level
   exclusion above).
6. **Top-N + OTHER one-hot function** — one reusable
   `fit_topn_categories(train, col, target, min_n)` /
   `apply_topn_categories(df, col, categories)` pair (fit the qualifying
   category list — those clearing the `n` threshold — on train only, fold
   everything else to `OTHER`). This is the actual pattern now used for
   all three high-cardinality identity/text columns, not smoothed target
   encoding: `Origin_Country` (6 countries ≥4% + `OTHER` + `MISSING`),
   `Company_ID` (16 companies n≥30 + `OTHER`), `Agent_ID` (76 agents n≥30
   + `OTHER`). Fit on train only — this is what makes it safe to reuse
   fold-by-fold in Stage 3's CV per the `model-training` skill's leakage
   guardrail (same reasoning as smoothed encoding would have needed, just
   a different transform).
7. **Datetime conversion** — `Course_Start_Date` → days-from-min-date.
8. **Numeric passthrough** — `Professionals_Count`, `Theory_Hours`,
   `Returning_Client`, and the raw numeric side of any column that also
   got a flag (`Waiting_List_Days`, `Daily_Tuition_Cost`).

## Step 3 — assemble final matrix

One cell: concatenate all transformed columns into `X`, confirm `y =
df[TARGET]`, print final shape, dtype summary, and a null-count check (must
be zero after imputation). This is the artifact Stage 3 consumes.

## Step 4 — synthesis write-up (for the professor, not line-by-line)

One markdown cell narrating the whole pipeline by treatment group, each
claim backed by the printed output from Step 3 — e.g. "5 columns dropped
(2 pure identifiers/no-signal, 3 flagged as leaking post-outcome
information), plus a 9,695-row exclusion for the deterministic
`Payment_Terms`/`Enrollment_Type` slice; 5 columns collapsed to
binary presence/threshold flags because Stage 1 found the signal lived
entirely in the 0-vs-nonzero split; 3 identity columns (`Origin_Country`,
`Company_ID`, `Agent_ID`) used smoothed target encoding instead of one-hot
due to cardinality (150+/184/203 raw levels)..." — referencing the Step 1
table rather than repeating each variable's Stage 1 paragraph.

## Verification

- Rerun notebook top to bottom, no errors.
- Final `X` has zero nulls, all-numeric dtypes.
- Row count drops by exactly 9,695 from the Payment_Terms exclusion (Step
  0.1) and by nothing else — state this explicitly in Step 3's printed
  output, per the `feature-engineering` skill's outlier-handling rule
  ("state the rule and the row count it affects").
- Smoothed-encoding function is confirmed callable as `fit(train)` +
  `transform(any)` two separate steps, not one inline computation on the
  full frame — this is what Stage 3 depends on.
- `Test_Data_No_Target.csv` not referenced anywhere in this stage.
