# Final Report Foundation — Group 46, Nova Academy `Dropped_Course` Prediction

Status: **foundation only** — structure, content anchors, and narrative threads for you to write from. No section below is final prose; it's the skeleton plus the specific numbers/quotes/decisions each section must be built around, pulled directly from `Group_46_Notebook.ipynb`.

Target: 8–10 pages, CRISP-DM ordered, following the notebook's own Part A → B → C → D structure. Appendix does not count against the 8–10 page budget (standard convention — confirm with the professor if unstated).

---

## 0. Governing principles for whoever writes from this outline

1. **Don't transcribe notebook bullets.** The notebook's markdown cells are terse, clipped, and written for a reader who has the code in front of them. The report is a different artifact — it needs connective prose that explains *why one decision led to the next*. Where this outline says "connect X to Y," that connection does not already exist verbatim in the notebook — writing it is the actual authorship work.
2. **Every number must trace to a printed notebook cell.** Same discipline as the notebook itself. This outline cites the specific numbers already in the notebook so you don't have to re-derive them; don't introduce new ones without checking the notebook output first.
3. **Not every variable gets equal ink.** ~26 variables were examined in Part A; a report that gives each one a paragraph is a transcript, not a report. Section 2 below tags each variable as **[FULL]** (earned real narrative — a dilemma, a surprising result, a reversal) or **[TABLE]** (state the decision, one line, done). This is where "significant contribution per variable" should show up as *proportional depth*, not uniform coverage.
4. **The throughlines are what make this a report and not a log.** Section 2 flags two threads (Origin_Country, and the deterministic Payment_Terms slice) that must be picked back up explicitly in Part C/D. Reusing them is what turns four disconnected stages into one argument.
5. **First person plural throughout** ("we decided," "we excluded"), matching the notebook's own voice. The agent is referred to as a tool the group directed and reviewed — never as a co-author. See Appendix A framing.

---

## 1. Introduction — Business Understanding (~0.75 page)

CRISP-DM's Phase 1 is not written up anywhere in the notebook — the report has to supply it. This is itself worth a sentence: state plainly that the business framing was inferred, not handed to you, and say what you inferred it from.

Content anchors:
- The problem: predict `Dropped_Course` for Nova Academy enrollments before the fact, so at-risk clients can be flagged for intervention.
- Why asymmetric costs matter here, stated up front (this is the seed for the recall-over-accuracy decision in Part D — plant it early so it doesn't feel bolted on later): a missed dropout (false negative) forecloses any chance to intervene; a false alarm (false positive) costs a check-in call. State this as your own inference, not a given fact — a genuine Phase-1 judgment call.
- Baseline rate: 41.4% of all enrollments end in `Dropped_Course = 1` (`baseline_rate` in Setup) — this is the number every later rate-by-value comparison in Part A is measured against, and it's also a reminder this is not a rare-event problem — it's close to a coin flip, which is itself a reason accuracy alone is a weak metric later.
- One paragraph previewing structure: Part A ≈ Data Understanding, Part B ≈ Data Preparation, Part C ≈ Modeling, Part D ≈ Evaluation. Say explicitly that the report follows the notebook's own section order because that order *is* the CRISP-DM order here — not a coincidence, a deliberate structuring choice made when the notebook was planned.

---

## 2. Part A — Data Understanding (~2.5–3 pages, the load-bearing section)

Open with a one-paragraph frame: EDA here wasn't a distribution-plotting checklist — every variable got a rate-vs-baseline comparison, a missingness check, and (for categoricals) a chi-square test, and every one of those checks produced an explicit, logged decision. That discipline is the actual finding of Part A, before any individual variable's result.

Organize as **clusters with a shared pattern**, not 26 flat subsections — each cluster states the pattern once, then the variables that fit it.

### 2.1 The sparse-tail pattern **[FULL — pick one representative, table the rest]**
Recurring shape: a count variable where 90%+ of mass sits at 0/1 and the tail (2, 3, …, 21) is too thin (n in the single/double digits per value) to trust individually.
- `Prev_Course_Attended`: drop rate 42.1% → 7.4% once a client has attended even once. Strongest protective single effect found in Part A — worth the full narrative treatment (why binary-collapse was the right call, not a numeric variable: 98% of mass at exactly 0).
- `Prev_Course_Dropouts`: 36.6% → 97.4% between 0 and 1 (n=196 above value 1, excluded from effect estimates).
- Table the rest with one line each: `Waiting_List_Days` (0 vs 1+: <40% vs >67%, kept both the flag *and* the raw day count since non-zero waits vary meaningfully around a ~75-day mean), `Registration_Changes` (0/1/2+ bucket; non-monotonic — 1 drops to 14%, 2+ bounces back to 21%, still under baseline), `Pre_Course_Supports_Tickets` (0 vs 1+: 54.4% vs 19.5%), `Observers_Count` (0 vs 1+: 41.6% vs 16.8%, n=315 non-zero).

### 2.2 The non-monotonic surprise **[FULL]**
`Professionals_Count`: rates are 26.3% (0) → 33.8% (1) → **44.2% (2, the highest-risk group, and 73.1% of all rows)** → 35.4% (3) → 23.8% (4). This is the one count variable in the whole EDA pass that did *not* get bucketed to 0/1+, precisely because collapsing it would have erased the signal — the peak risk sitting in the *middle* value is the finding, not noise to smooth over. Worth stating explicitly as a case where the "default" cleaning move (bucket a sparse-tail count) was consciously rejected.

### 2.3 Sentinels vs. real data **[FULL]**
The report should name this as a distinct skill from missing-value handling: recognizing a value that is *technically present but not real data* — a system artifact, not a client's answer.
- `Students_Count = 9999` (n=55) — not a headcount, recoded to NaN.
- `Practical_Hours ∈ {-5, -1, 5000, 10000}` (n=117) — negative hours and 5-figure hour counts aren't physically possible; recoded to NaN.
- `Daily_Tuition_Cost = 5400` — outlier value, recoded to NaN (surfaces in Part B).
- One line connecting these: all three were caught the same way — a `.describe()`/`.value_counts()` pass that made an implausible cluster visually obvious, then a domain-plausibility judgment (not a statistical rule) that these values couldn't be real.

### 2.4 Text cleanup — the dirty-categorical problem **[FULL for Origin_Country and Payment_Terms, TABLE the rest]**
Frame: several "categorical" columns were actually free text with typos/case/whitespace noise inflating cardinality by 10–100x. Regex normalization (uppercase, strip non-letters, collapse whitespace) recovered the real category count in every case — state the before/after as evidence the cleanup was necessary, not cosmetic:
- `Origin_Country`: 721 raw → 154 real countries.
- `Client_Category`: 505 raw → 8.
- `Payment_Terms`: cleaned + bucketed to 3.
- `Enrollment_Type`: 298 raw → 5.
- `Submission_Source`: 328 raw → 6.
- `Catering_Package`: 321 raw → 5.
- `Lanyard_Color`: 240 raw → 5 (then dropped, see 2.5).

**`Origin_Country` — thread #1, plant it here, pay it off in Part D.** `PRT` is 41.6% of all rows *and* the highest-risk group at 64% drop rate against a 41.4% baseline (chi² ≈ 0, not coincidence), while every other major country sits at or below baseline (`FRA` 17%, `DEU` 17%, `GBR` 28%). The notebook's own Part A text already flags the risk explicitly: *"PRT alone could end up dominating the model's use of this feature."* The report must state this as a prediction made in Part A, then confirm in Part D whether it came true (it does — `origin_country__PRT` is the #1 SHAP driver). That callback is the single strongest piece of evidence the report is a coherent analysis and not four disconnected stages.

**`Payment_Terms` — thread #2, the central dilemma of Part A.** `PAY UPON START` (81% of rows) sits well below baseline at 29.4%; `PREPAID NONREFUNDABLE` (17%) sits at **almost 100%**. That's not a normal effect size — it's a red flag for something structural. The investigation that followed is the best "dilemma" material in the whole notebook and should be walked through step by step in the report, not summarized in one line:
1. The question raised: is a near-100% rate real signal, or a hidden deterministic rule?
2. The cross-reference: split `PREPAID NONREFUNDABLE` by `Enrollment_Type` (affiliated vs. not).
3. The finding: non-affiliated + prepaid = **100% drop rate, n=9,695** — and every single one of the 24 non-dropped prepaid rows turned out to be affiliated. Zero exceptions in either direction.
4. The judgment call: this slice was excluded from modeling entirely (not encoded, not kept as a feature) rather than let a model learn what looks like a deterministic bookkeeping artifact rather than a behavioral pattern. This is a genuine, arguable decision — the report should present it as one, including the alternative that was considered and rejected (keeping it in, since a perfect rule is still a legitimately useful predictor if it holds on unseen data) and why the group didn't go that way (no way to confirm from the data alone whether the determinism reflects real client behavior or a labeling/process artifact specific to this training set).

Table the rest: `Client_Category` (SaaS 46% at 36% below baseline; Big Tech 19% at 68.4% — 27 points above baseline, chi² ≈0; `UNKNOWN` n=2 excluded), `Enrollment_Type` (General 71%/Affiliated 24% dominate), `Submission_Source` (B2B Platforms 85.5% dominant; Direct Website / Dedicated Sales flagged as protective), `Catering_Package` (`ALL INCLUSIVE` 79.5% but n=44, noted as too small to trust).

### 2.5 The null results — recognizing when a variable has nothing to say **[TABLE, but state the principle in one sentence]**
`Welcome_Gift_Type` and `Lanyard_Color`: every category sits on the baseline line — dropped. Stating *why* a variable was dropped (not just that it was) is what separates informed feature selection from arbitrary pruning — say so once, explicitly, rather than letting four drop-decisions look like an afterthought.

### 2.6 The leakage-adjacent trap **[FULL — this is a genuine "question that arose"]**
`Assigned_Lab_Config`: naively, clients whose assigned config *didn't* match their request dropped *less* — a plot that reads as "give people what they didn't ask for and they're more likely to stay," which is backwards from intuition. The resolution: a mismatch can only be recorded for clients who *showed up* to be reassigned in the first place — so the variable is encoding attendance (an outcome) rather than predicting dropout (upstream of the outcome). This is worth a full paragraph because it's a subtle temporal-leakage trap that a naive rate-vs-baseline read would have missed, and catching it required reasoning about the data-generating process, not just the numbers.

### 2.7 ID variables — the rare-category problem **[FULL for the reasoning, TABLE the specific numbers]**
`Company_ID` and `Agent_ID` share a pattern worth stating once: most values (companies/agents) have too few enrollments to estimate a reliable drop rate individually. The report should name the statistical reasoning explicitly — a minimum-n threshold (n≥30) grounded in the Central Limit Theorem, not an arbitrary round number — before giving the resulting split:
- `Company_ID`: 95%+ missing; known-company rows drop at half the baseline rate (21% vs 42%); only 16 of 184 known companies clear n≥30, ranging 3.3%–100% among those. Decision: keep a `has_company_id` flag *and* one-hot the 16 qualifying companies, `OTHER` for the rest.
- `Agent_ID`: 17.6% missing, and — opposite direction from `Company_ID` — known-agent rows drop *more* (43.1% vs 33.5% missing). 76 of 203 agents clear n≥30, ranging 1.0%–100%. Same flag + top-N + `OTHER` treatment.
- Worth one sentence: these two ID variables produced opposite-signed missingness effects, which is itself evidence the group treated each variable on its own data rather than applying a template blindly.

### 2.8 Numeric variables with real continuous signal **[TABLE, two lines each]**
`Registration_Days_Before` (right-skewed, median a more honest center than mean 103 vs 65; dropped clients register further ahead — median 104 vs 43 days; missingness flat vs baseline, median-filled), `Theory_Hours` (stayed vs. dropped distributions nearly identical — flagged as a variable expected to contribute little signal, kept anyway for completeness), `Daily_Tuition_Cost` (zero-cost flag added alongside raw value, mirroring the `Waiting_List_Days` pattern), `Course_Start_Date` (converted to days-since-start; two extreme months noted but *not* treated as outliers given small sample sizes in those same months — a decision to not overreact to noise).

### 2.9 Close Part A
One paragraph: by the end of Part A, every one of ~26 variables has a stated, quantified, defensible decision attached to it — keep as-is, bucket, binary-flag, one-hot with thresholding, drop, or exclude specific rows. That decision log is what Part B mechanically executes — nothing in Part B introduces a new judgment call that wasn't already made here.

---

## 3. Part B — Data Preparation (~1–1.25 pages)

Frame Part B as execution, not decision-making — the report should make clear this is a deliberate CRISP-DM distinction (Data Understanding decides, Data Preparation implements), not just where the notebook happens to draw a line.

Content anchors:
- Group the transforms by type, not by variable (this table already exists in the notebook's own Part B summary — reuse its structure, don't re-derive):
  - Bucketed (0/1/2+): `Registration_Changes`, `Payment_Terms` (3-way).
  - Binary flags added alongside raw values: `had_prior_dropout`, `attended_before`, `has_waited`, `has_observers`, `has_support_ticket`, `is_zero_cost`, `has_company_id`, `has_agent_id`.
  - One-hot with `OTHER`/`MISSING` thresholding: `Origin_Country`, `Client_Category`, `Payment_Terms`, `Enrollment_Type`, `Submission_Source`, `Catering_Package`, `Requested_Lab_Config`, `Company_ID`, `Agent_ID`.
  - Median-filled (deliberately not mean — right-skewed distributions established in Part A): `Registration_Days_Before` → 53.0, `Students_Count` → 0.0, `Practical_Hours` → 1.0, `Daily_Tuition_Cost` → 95.2.
  - Dropped entirely: `Client_ID`, `Welcome_Gift_Type`, `Lanyard_Color`, `Physical_Course_Kits`, `Assigned_Lab_Config`.
- **Row exclusions — pay off the Payment_Terms thread here.** 9,695 rows (the deterministic prepaid/non-affiliated slice) + 2 (`Client_Category = UNKNOWN`) = 9,697 of 63,464 rows (≈15%) removed before modeling. This is a big enough cut that the report should state the tradeoff explicitly in one sentence: real training signal was given up specifically to avoid teaching the model a pattern the group could not confirm was genuine behavior rather than a data artifact.
- Dimensionality: name it as two combined reduction strategies — EDA-based (text cleanup collapsing junk cardinality, e.g. `Origin_Country` 721→154) and statistics-based (Part B's n-threshold folding the rest into `OTHER`, e.g. `Origin_Country` 154→8 final columns, `Company_ID` 184→17, `Agent_ID` 203→77).
- Close with the final `X` shape and the rows-per-feature ratio the notebook prints — a concrete, checkable statement that the feature space is defensible relative to the row count, not a curse-of-dimensionality risk.
- One flagged-but-unresolved item to carry forward honestly: `Theory_Hours` (max 41h, n=1) and `Registration_Days_Before` (max 629 days, n=16) right-tail values were left untouched — say this was a conscious "not yet" rather than an oversight.

---

## 4. Part C — Modeling (~1.5–1.75 pages)

Open by naming the CRISP-DM shift: Data Preparation produced one clean `X`/`y`; Modeling's job is choosing *and justifying* an algorithm family, not just running one. State the three-model choice as deliberate escalation in model complexity/assumptions, not an arbitrary list: Logistic Regression (linear, interpretable baseline), Random Forest (bagged trees, nonlinear + interaction terms, variance-reduction via averaging), XGBoost (boosted trees, expected strongest on structured/tabular data, explicit regularization).

Content anchors:
- Methodology, stated as an evaluation-design decision before any modeling result is discussed: 20% held out (stratified) *before* any tuning touched the data, reserved untouched until Part D; the remaining 80% split into a shared 5-fold `StratifiedKFold` reused across all three models so their CV scores are directly comparable (a fair-comparison design choice, not a default).
- Results table (already computed in the notebook — reuse verbatim):

| Model | Mean CV AUC ± std | Train-fold AUC | Gap |
|---|---|---|---|
| Logistic Regression | 0.854 ± 0.005 | 0.857 | 0.003 |
| Random Forest | 0.922 ± 0.003 | 0.967 | 0.045 |
| XGBoost | 0.930 ± 0.003 | 0.987 | 0.057 |

- **Dilemma #1 — reading the bias-variance tradeoff, not just the AUC ranking.** Walk through the actual reasoning the notebook did, don't just state the conclusion: LogReg's near-zero gap (0.003) means no overfitting but is also consistent with a model too simple to capture the data's real structure. XGBoost's largest gap (0.057, ≈18× the fold-to-fold std) looks like the most overfit model by that metric alone — but two independent checks argue the 0.930 score is real, not inflated: fold-to-fold stability (low std means the gap is a consistent, bounded effect, not one noisy fold), and a separate `early_stopping_rounds` run converging on the *same* 0.930 validation AUC by a completely different mechanism (watching validation loss directly rather than cross-validating hyperparameters). State explicitly: this is a case where the "safer-looking" metric (smallest gap) was *not* the deciding factor — trust in the number came from corroboration across two independent methods, not from picking the model with the prettiest gap.
- **Dilemma #2 — the trip-wire.** Both tree models crossed 0.90 CV AUC, treated in the loop-prompt design (see Appendix A) as a mandatory check, not a default trust: reconfirm the known leakage columns (`Physical_Course_Kits`, `Assigned_Lab_Config`, `Welcome_Gift_Type`, `Lanyard_Color`, `Client_ID`) are absent from `X`, and reconfirm the `OTHER`-bucket groupings were built from raw frequency counts only, never from the target. State the actual interpretive payoff: because the *linear* baseline already reaches 0.854 on the same features, the tree models' extra 0.07–0.08 AUC reads as a real interaction-effect lift on genuine signal, not a sudden jump attributable to one leaked column — a linear model can't exploit an interaction leak the way a tree can, so its already-strong baseline is itself part of the leakage argument, not just a comparison point.

---

## 5. Part D — Evaluation & Interpretation (~1.5–1.75 pages)

**Dilemma — the metric that actually decided model selection.** This is where the Introduction's business framing gets paid off — make that callback explicit. XGBoost and Random Forest are close on AUC (0.9334 vs 0.9278 on the untouched holdout — note this *independently* confirms the CV ranking, a second robustness check worth naming), but the deciding factor stated in the notebook is recall: XGBoost catches 107 more of the 3,321 actual holdout dropouts than Random Forest, for only 9 extra false positives. Say explicitly why that trade was taken deliberately: given the Introduction's asymmetric-cost framing (a missed dropout forecloses intervention; a false alarm costs a check-in call), a small precision cost for a real recall gain is the right trade, not a neutral tie-breaker.

Holdout numbers to cite: 87% accuracy, 0.933 AUC, 74% recall, 82% precision on flagged cases; 866 real dropouts (8.1% of holdout) still slip through undetected — state this last number honestly as the model's real limitation, not just its headline accuracy.

**SHAP — the payoff of thread #1.** Top drivers: `origin_country__PRT`, then `had_prior_dropout`, then open support tickets, agent group, and registration/course-start timing. State explicitly: `had_prior_dropout` ranking #2 is the model rediscovering, on its own, exactly the effect Part A already found by hand (drop rate nearly tripling between 0 and 1 prior dropouts) — a second independent confirmation of a human-found effect, not a new discovery.

**The fairness dilemma — leave this genuinely open, don't resolve it artificially.** `origin_country__PRT` being the #1 driver is the thing Part A explicitly predicted might happen. The report should walk through the follow-up check performed: `PRT` students are disproportionately `Big Tech`/`Traditional IT` (66% combined) while non-`PRT` students are disproportionately `SaaS` (66.1%) — so the model may be partly using nationality as a proxy for industry rather than (or in addition to) a genuine nationality effect. State plainly that this was **not resolved** — it's presented as an open question for further work (a fairness/entanglement check before any deployment), not swept under a confident conclusion. This is one of the strongest pieces of evidence in the whole report that the analysis was driven by a critical human reader rather than a tool optimizing for a clean-looking result.

Close Part D with the "what we'd still do" list from the notebook, framed as genuine next steps rather than boilerplate limitations: tune the decision threshold away from the default 0.5 (recall/precision were reported *at* 0.5, not at an optimized cutoff — worth flagging as a choice that wasn't actually tuned), add an explicit interaction feature or fairness check for the country/industry entanglement, revisit the untouched `Theory_Hours`/`Registration_Days_Before` tails flagged back in Part A/B and never resolved.

---

## 6. Conclusion (~0.5 page)

Recap the CRISP-DM arc as one connected argument, not four summaries stapled together: a business framing that set the recall-over-accuracy stakes → an EDA that produced a per-variable decision log, including two threads (Origin_Country, Payment_Terms) that resurface later → a preparation stage that mechanically executed that log, at a real cost (≈15% of rows excluded) taken deliberately → a modeling stage that chose XGBoost via corroborated evidence (not a single metric) → an evaluation that confirmed Part A's own findings via SHAP and surfaced one real, unresolved fairness question. End on the honest note, not a triumphant one: state what's genuinely unresolved (the country/industry entanglement, the untuned threshold) as the report's real contribution — knowing what you don't yet know is itself a finding.

---

## Appendix A — The Claude Workflow

This appendix exists to make an explicit case: the notebook's code was written with an LLM coding agent as a tool, but the judgment calls throughout this report were not the agent's. State the operating principle plainly, close to how it was actually decided (paraphrase, don't just repeat this outline's wording verbatim):

> We divided the work by how much genuine ambiguity a step carried, not by how much code it required. Where a step was long, mechanical, and low-dilemma — running a hyperparameter search, executing repeated cross-validation folds, formatting comparison tables — we wrote the agent a bounded plan up front and let it execute autonomously within it. Where a step required a judgment call with no single correct answer — whether a value is a sentinel or real data, whether a deterministic-looking slice of rows is signal or artifact, which model's overfit gap is still trustworthy, which metric should decide between two close models — we made that call ourselves, iterating with the agent cell-by-cell rather than handing off the reasoning.

Concrete evidence to cite (already on disk, don't paraphrase from memory — pull the actual constraints):
- `plans/Part_C_loop_prompt.md` — the autonomous-run brief used for Part C. Point to specific constraints as evidence the *design* was human even though the *execution* was delegated:
  - No fixed target AUC to hit — the agent was told to derive its own stopping rule (train/CV gap stabilizing + further search not beating fold-to-fold noise), because picking an arbitrary target without a baseline "would not be statistically defensible."
  - The >0.90 AUC trip-wire was written into the brief in advance, not applied as an afterthought once the number came in suspiciously high.
  - Every markdown number had to come from an actually-executed cell (`jupyter nbconvert --execute`), not typed from memory — a guardrail against the agent (or a human) fabricating a plausible-sounding statistic.
  - Explicit scope fence: Part D was declared out of scope for that run, with a named list of artifacts (fitted estimators, untouched holdout, real column names surviving the pipeline) the next stage would need — evidence of the plan being designed with the whole pipeline's downstream needs in mind, not just the immediate task.
- Contrast this with the Part A variable-by-variable narrative in this report (Section 2) — every dilemma flagged **[FULL]** there was a live, cell-by-cell human call, not something a bounded autonomous run was ever set loose on.

Close the appendix with the actual point: this division of labor is itself a modeling decision, made the same way the ones in Part A–D were — by identifying where ambiguity lived and routing accordingly — and it's why the report can state, honestly, that an agent wrote code under direction while the analysis in this report was not delegated.

---

## Page budget summary

| Section | Pages |
|---|---|
| 1. Introduction | 0.75 |
| 2. Part A | 2.5–3.0 |
| 3. Part B | 1.0–1.25 |
| 4. Part C | 1.5–1.75 |
| 5. Part D | 1.5–1.75 |
| 6. Conclusion | 0.5 |
| **Body total** | **8–9.5** |
| Appendix A (Claude Workflow) | not counted against body |
