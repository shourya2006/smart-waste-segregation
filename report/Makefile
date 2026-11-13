# Makefile for compiling IEEE LaTeX report

MAIN = main
BIB = references

all: $(MAIN).pdf

phase2: phase2.pdf

$(MAIN).pdf: $(MAIN).tex $(BIB).bib
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

phase2.pdf: phase2.tex $(BIB).bib ablation-section.tex hybrid-section.tex evaluation-section.tex
	pdflatex phase2
	bibtex phase2
	pdflatex phase2
	pdflatex phase2

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz $(MAIN).pdf phase2.pdf

.PHONY: all phase2 clean
