# Phase 1 IEEE report

The current submission report is compiled directly from `main.tex` with IEEEtran.

- PDF: [Phase 1 report](../output/pdf/phase-1-report.pdf)
- Editable source: [main.tex](main.tex)
- References: [references.bib](references.bib)

The five-page report covers the six contributors, literature, dataset audit,
features, model theory and comparisons, calibration, frozen test results,
demo implementation and limitations. Counts and settings match the saved
experiment records. Report preparation does not refit or evaluate any model.

## Compilation

From the repository root with Tectonic installed:

```sh
tectonic --outdir output/pdf report/main.tex
```

This writes `output/pdf/main.pdf`. The distributed copy has the descriptive name
`phase-1-report.pdf`. Alternatively, run `make` inside `report/` with TeX Live,
or upload the source, bibliography and `figures/` folder to Overleaf.

The earlier `report.html` is an archived preview and does not contain all
corrections in the current LaTeX report. Use the compiled PDF for submission.
