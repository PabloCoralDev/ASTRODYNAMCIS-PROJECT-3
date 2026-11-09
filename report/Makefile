PY ?= python3

all: main.pdf

main.pdf: main.tex sections/*.tex figures/*.tex figures/*.png outputs/*.txt ../*.py
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

# re-run the script (from the repo root) to refresh program output and figures
outputs:
	cd .. && MPLBACKEND=Agg $(PY) tasks1to3.py > report/outputs/main.txt

clean:
	rm -f main.aux main.log main.out main.pdf

.PHONY: all outputs clean
