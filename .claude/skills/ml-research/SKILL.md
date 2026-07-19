---
name: ml-research
description: Domain-oriented literature/methodology research for the Nova Academy Dropped_Course ML project (classical supervised learning — sklearn/XGBoost internals, encoding/imputation methodology, fairness/leakage literature). Use when a question needs grounding in primary sources — a library's actual documented behavior, a paper on an encoding/imputation technique, bias/fairness methodology — rather than general research.
---

Spin up a **background agent** to do the research, so you keep working while it reads. This is the same mechanism as the generic `research` skill, pre-loaded with this project's domain so the agent doesn't start from zero.

Its job:

1. **Check `plans/` and `extra/` first.** `plans/Part_A_plan.md` through `Part_D_plan.md` and `extra/EXECUTIVE_SUMMARY.md` already document this project's established methodology and findings (leakage columns excluded, sentinel-value handling, the `n >= 30` / `OTHER`-bucket encoding rule, the untouched 80/20 holdout, XGBoost as current best at 0.933 holdout AUC, and the `Origin_Country=PRT` / `Client_Category` confound). Don't re-derive or contradict an established finding without flagging it explicitly as a challenge to prior work.
2. Investigate against **primary sources** — official library docs (scikit-learn, XGBoost), the paper introducing a technique, not a secondary blog write-up of it — following every claim back to the source that owns it, exactly as the generic `research` skill does.
3. Write findings to a Markdown file, citing each claim's source. Match this project's existing convention (`plans/Part_*.md` style) rather than inventing a new location.
4. If the finding bears on model choice or evaluation, state explicitly whether it should change the model comparison table in `extra/EXECUTIVE_SUMMARY.md` or is a refinement for future work — don't silently imply a result is overturned.
