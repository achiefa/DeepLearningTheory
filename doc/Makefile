# Define output files
NNTRAINING_OUT = nntraining.pdf

all: nntraining

nntraining:
		pdflatex nntraining
		bibtex nntraining
		pdflatex nntraining
		pdflatex nntraining

help:
		@echo "Makefile for LaTeX"
		@echo "Usage: make [target]"
		@echo "Available targets:"
		@echo "  nntraining   - Build paper (default)"

clean:
	rm -f *~ *.out *.toc *.log $(NNTRAINING_OUT) *.aux *.blg *.bbl *.idx
