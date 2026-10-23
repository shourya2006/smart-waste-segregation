# Team workflow

Use `main` for working milestones. Create a branch for a focused task, make
small descriptive commits, and open a pull request for teammate review.
Examples: `feat/dataset-audit`, `feat/image-descriptors`, `feat/ml-baselines`.

Run `ruff check .` before committing Python changes. Once functional code is
added, introduce tests for meaningful behavior. Run existing data checks with
`python -m unittest discover -s tests -v`. Extend them for feature dimensions
and train/inference consistency as the modeling pipeline is implemented.

Do not commit credentials, source datasets, cached feature matrices, or model
binaries. Review `git diff --staged` before committing. Record deliberate
dataset exclusions and experiment changes rather than silently replacing them.

Update `docs/phase-1-plan.md` when completing a milestone and record individual
contributions in `docs/contributions.md`. Never invent experimental scores.
Every member should be able to reproduce a run and explain the whole pipeline.

## Individual commits for the six-person work division

See [the assignments](docs/team-assignments.md) for each member's implementation
ownership. Each member should use their own GitHub account and local clone.
Configure Git locally using their actual name and an email verified by GitHub
(a GitHub-provided noreply address also works):

```sh
git clone https://github.com/shourya2006/smart-waste-segregation.git
cd smart-waste-segregation
git config user.name "Your actual name"
git config user.email "Your GitHub-verified email"
git switch -c feat/your-assigned-task
```

After implementing a coherent change, run the relevant checks, stage only the
intended files, inspect the staged diff, commit, and push the branch:

```sh
ruff check .
python -m unittest discover -s tests -v
git diff --staged
git commit -m "Describe the implemented change"
git push -u origin HEAD
```

Stage the specific files with `git add` before inspecting and committing them.
Open a pull request targeting `main`, request the assigned partner's review, and
link it in `docs/contributions.md`. The repository is private: its owner needs
to invite collaborators before they can push. Five collaborators’ GitHub usernames are pending;
no invitations have been sent as part of the assignment.

If preserving every reviewed commit on `main` is important, use **Create a merge
commit** when merging. Squash merges combine branch commits. Attribution still
depends on correct commit author details and a GitHub-associated email; commit
counts alone do not establish a meaningful contribution.

References: [GitHub commit email setup](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address) and [pull request merge strategies](https://docs.github.com/en/pull-requests/reference/pull-request-merges).
