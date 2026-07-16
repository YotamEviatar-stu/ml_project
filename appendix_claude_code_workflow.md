# Appendix — Our Development Workflow with Claude Code

All artifacts referenced below (plans, skills, `CLAUDE.md`) are committed to
this repository's git history and can be inspected directly.

## Plan → constrain → execute → validate, per stage

Each analysis stage (EDA, feature engineering, model training, evaluation)
followed the same four-step cycle:

**1. Plan in writing before any code.** Every stage started as a standalone
markdown plan (`stage1_plan.md` … `stage4_plan.md`), agreed on before
touching `train.ipynb`. Each plan states the rubric requirement it
satisfies, prerequisites from prior stages, and the concrete deliverable —
`stage4_plan.md`'s prerequisite section, for example, names exact objects
(`X_holdout`, `fitted_models`, row counts) Stage 3 must produce first.

**2. Encode standing rules once — `CLAUDE.md`.** Project-wide constraints
(statistical rigor over cosmetic plots, no speculative comments in WIP
code, every printed statistic must trace to real cell output) live in
`CLAUDE.md` at the repo root, so the rule is set once and enforced
automatically in every session, not repeated per prompt.

**3. Turn recurring methodology into skills.** Stabilized approaches became
reusable skills under `.claude/skills/` — `feature-engineering` and
`model-training` — self-loading checklists that pull in the rubric target
and prior decisions instead of us re-explaining context by hand.

**4. Set boxed requirements per prompt, not vague asks.** Prompts specified
hard constraints up front (files that must never be touched, required
train/holdout split, rubric items to hit), so output was checkable against
a spec rather than "looks reasonable."

## Loop prompting and fresh-session validation

For Stage 3, we used a dedicated autonomous-run prompt
(`stage3_loop_prompt.md`) pasted into a **fresh session** with a fixed
budget, forcing the agent to re-read the plan and skill files from disk
(not from memory) and execute the notebook for real after every edit.

We also deliberately opened new sessions to check prior work rather than
letting one long session mark its own homework: a fresh session has no
memory of *why* a choice was made, so if a claim only holds up "because I
remember the context," re-deriving it from the notebook's own output
exposes that immediately. This caught drift between what a plan said and
what the notebook actually did.

## Why this is our edge

Other groups' prompts are one-off asks answered from a blank slate every
time. Ours compound: `CLAUDE.md` and the skills directory mean every new
session inherits the full standard of rigor already established, plans
make each stage's deliverable checkable rather than subjective, and
fresh-session validation catches inconsistency instead of trusting
momentum. The full trail is version controlled in git — auditable, not
just claimed.
