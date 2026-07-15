# CLAUDE.md

Project-specific guidance for Claude Code when working in this repo (Nova Academy `Dropped_Course` prediction, Stage 1 EDA in `train.ipynb`).

## Research context

This is coursework-style analysis, but treat it with the same rigor as academic research: every analytical choice (bucketing thresholds, missing-value strategy, encoding choice) must be statistically defensible, not just visually convincing. Report the test statistic and p-value alongside any visual summary, and don't invent theoretical code-review issues (e.g. "hardcoded sample rate") that don't actually apply to this dataset.

See `.claude/skills/feature-engineering/SKILL.md` for the full methodology (variable glossary, dirty-categorical patterns, missing-value framework, encoding decision tree, interpretation write-up style) and `stage1_plan.md` for the variable priority queue. Read both before making changes to `train.ipynb`.

## Code comments & docstrings (WIP phase)

The codebase is in active, iterative development — not yet finalized. Do NOT add explanatory comments, docstrings, or per-entry justification strings to code unless explicitly asked. A bare list/value is enough (e.g. a `SUBJECTS_EXCLUDE`-style list needs no inline rationale per entry). If the reasoning should be kept somewhere, say so in chat instead of writing it into the file — documentation gets added once a piece of work is settled.

## Communication style

Keep explanations concise and concrete.

### Explaining technical changes

Every non-trivial code-change explanation should follow this pattern:
1. **State what the current code does, in plain terms** — name the specific variable/line, say what value it produces.
2. **Show a concrete numerical example of why that's a problem** — realistic numbers, walk through the wrong outcome.
3. **Show the same example after the fix** — same numbers, correct outcome, visible contrast.
4. **State the practical consequence** — how big is the error, does it bias a result or risk reversing a conclusion.

Don't explain a change as "X is more robust" without a number that makes the difference visible.

### Explaining concepts and methodology

When explaining a concept or statistical method (not a code diff), build it up in layers instead of front-loading formulas:
1. Plain language first, grounded in this project's real variables (e.g. `Client_Category`, `Origin_Country`) — not abstract math terms.
2. Introduce one symbol/piece at a time, each with its own concrete tie-in.
3. Only after each piece has landed, show the formal notation and connect it back.
4. Check whether it landed before adding more — if not, rebuild slower and plainer with the same objects, don't just re-explain more densely.

Stay at the level of what's specific to the method being taught; don't derive generic statistical machinery (e.g. how chi-square or OLS mechanically works) from first principles unless asked.

### Orienting in the codebase

When referencing a function, variable, or cell, say which file/cell it's in, what role it plays (e.g. "this is the cleaning step, not the rate calculation"), and whether the notebook needs to be re-run to see the change (edits made via tooling don't execute automatically — no kernel is attached during editing).

## Notebook conventions

- Every printed mean/count table should also show pct of total n.
- Every plot needs a y-axis label.
- Every number cited in an interpretation markdown cell must come from the code cell's own printed/plotted output — no side-channel computation typed in from memory.
- Interpretation cells: short, clipped bullets — one distinct fact per bullet (category breakdown, dominant-group comparison, missingness, etc. each get their own line), not a single dense sentence bundling percentage + n + comparison together. Don't force a strict "one bullet per plot" mapping — split a plot's findings across multiple bullets if that's clearer. Decision can stay short and hedged (e.g. "perhaps revisit and change to N buckets") when a cleaner alternative is still open, rather than being fully committed prose. See `.claude/skills/feature-engineering/SKILL.md` for the worked example.
