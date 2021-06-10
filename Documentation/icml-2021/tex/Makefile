all: paper.pdf appendices.pdf ../paper.pdf

paper.pdf:
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

appendices.pdf:
	pdflatex appendices.tex
	bibtex appendices
	pdflatex appendices.tex
	pdflatex appendices.tex

../paper.pdf: paper.pdf appendices.pdf
	pdfunite $^ $@

.PHONY: paper.pdf appendices.pdf
