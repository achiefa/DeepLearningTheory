# Define output files
NTK_OUT = ntk.pdf
NNTRAINING_OUT = nntraining.pdf

all: nntraining

ntk:
		pdflatex ntk
		bibtex ntk
		pdflatex ntk
		pdflatex ntk

nntraining:
		pdflatex nntraining
		bibtex nntraining
		pdflatex nntraining
		pdflatex nntraining

help:
		@echo "Makefile for LaTeX"
		@echo "Usage: make [target]"
		@echo "Available targets:"
		@echo "  ntk	       - Build old notes"
		@echo "  nntraining   - Build new notes"

clean:
	rm -f *~ *.out *.toc *.log $(NTK_OUT) $(NNTRAINING_OUT) *.aux *.blg *.bbl *.idx
