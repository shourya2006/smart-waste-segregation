# Team workflow

Use `main` for working milestones. Create a branch for a focused task, make
small descriptive commits, and open a pull request for teammate review.
Examples: `feat/dataset-audit`, `feat/image-descriptors`, `feat/ml-baselines`.

Run `ruff check .` before committing Python changes. Once functional code is
added, introduce tests for meaningful behavior: duplicate isolation, feature
dimensions, train/inference consistency, and leakage prevention.

Do not commit credentials, source datasets, cached feature matrices, or model
binaries. Review `git diff --staged` before committing. Record deliberate
dataset exclusions and experiment changes rather than silently replacing them.

Update `docs/phase-1-plan.md` when completing a milestone and record individual
contributions in `docs/contributions.md`. Never invent experimental scores.
All five members should be able to reproduce a run and explain the whole pipeline.
