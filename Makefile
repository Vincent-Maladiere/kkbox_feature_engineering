scripts := $(wildcard notebooks/plot_*.py)
notebooks := $(patsubst %.py,%.ipynb,$(scripts))
rendered-notebooks := $(patsubst %.py,%.html,$(scripts))

PHONY: all download sessionize notebooks rendered-notebooks

all: download sessionize notebooks

download:
	python scripts/download.py

sessionize:
	python scripts/sessionize.py

notebooks: $(notebooks) $(rendered-notebooks)

%.ipynb: %.py
	jupytext $< --output $@

%.html: %.ipynb
	jupyter nbconvert $< --execute --to html

clean:
	rm -f notebooks/*.ipynb
