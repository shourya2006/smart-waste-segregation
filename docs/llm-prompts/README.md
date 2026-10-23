# Prompts for the six members

Each member should use a coding assistant with access to their authenticated
clone of this private repository. Read shared-rules.md with the relevant brief.
The repository owner reports that collaborator invitations have been added;
members need to accept their invitations before using authenticated Git access.

| Member | Brief | Branch |
| --- | --- | --- |
| Om Yadav | [Preprocessing](om-yadav.md) | `feat/preprocessing` |
| Aditya Bharadwaj | [Features](aditya-bharadwaj.md) | `feat/image-features` |
| Shourya Bafna | [SVM and shared training](shourya-bafna.md) | `feat/svm-baselines` |
| Daksh Batra | [Random Forest and comparison](daksh-batra.md) | `feat/random-forest` |
| Prajjwal Tripathi | [Calibration and evaluation](prajjwal-tripathi.md) | `feat/model-evaluation` |
| Saksham Mighlani | [Demo and integration](saksham-mighlani.md) | `feat/demo-integration` |

## Ready-to-paste starter

Replace the name, brief filename, and branch below using the table:

```text
I am [full name], a collaborator on this repository:
https://github.com/shourya2006/smart-waste-segregation

Clone it if needed, or inspect my existing checkout without overwriting local
work. Fetch the latest main. Read README.md, CONTRIBUTING.md,
docs/llm-prompts/shared-rules.md, and docs/llm-prompts/[brief filename].
Implement my assigned component on [branch], following that brief and existing
interfaces. Run the relevant tests and existing checks. Use my own configured
Git identity, make meaningful commits, push my branch, and open a pull request
to main. Do not push directly to main or merge. If dependencies are missing,
complete independent work, document what remains, and do not invent results.
Finish with the changes, test results, commit hashes, and PR link.
```

All members can begin development, fixtures, and documentation in parallel.
Real-data integration follows preprocessing → features → shared model experiments
→ calibration/final evaluation → full demo. Do not resolve missing dependencies
by creating competing implementations of another member's owned component.
