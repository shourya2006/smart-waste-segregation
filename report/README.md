# Phase 1 IEEE Technical Report

This directory contains the complete IEEE-format technical report for Phase 1 of the Smart Waste Segregation project.

## Authors & Responsibilities
- **Om Yadav**: Data Verification & Preprocessing
- **Aditya Bharadwaj**: Handcrafted Features & Caching
- **Shourya Bafna**: Training Protocol & SVM Baselines
- **Daksh Batra**: Random Forest & Feature Studies
- **Prajjwal Tripathi**: Confidence Calibration & Evaluation
- **Saksham Mighlani**: Demo Integration & Report Assembly

## Compilation Instructions

### Local LaTeX Installation (TeX Live / MacTeX)
Compile with `pdflatex` and `bibtex`:
```bash
make
# or manually:
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

### Overleaf / Online LaTeX
Upload `main.tex`, `references.bib`, and the `figures/` directory to Overleaf. Set the engine to `pdfLaTeX` or `LaTeX`.

### Local Environment Note & Compilation Limitation
On this local environment, no TeX distribution (such as MacTeX or TeX Live `pdflatex`/`xelatex`) is installed. Direct command-line compilation from `main.tex` to PDF is therefore unavailable in this environment.

To inspect the rendered report layout without installing TeX:
- Open `report/report.html` in any web browser (renders complete IEEE two-column layout with KaTeX formulas and figures).
- Inspect `report/report.pdf`, generated directly from the HTML source.
- For formal camera-ready compilation from LaTeX source, upload this folder to Overleaf or run `make` on a system with TeX Live installed.
