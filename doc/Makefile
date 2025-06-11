all:
		@echo "Makefile for LaTeX"
		@echo "Usage: make [target]"
		@echo "Available targets:"
		@echo "  ntk	       - Build old notes"
		@echo "  nntraining   - Build new notes"

# Define output files
NTK_OUT = ntk.pdf
NNTRAINING_OUT = nntraining.pdf

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

clean:
	rm -r *~ *.out *.toc *.log $(NTK_OUT) $(NNTRAINING_OUT) *.aux *.blg *.bbl
