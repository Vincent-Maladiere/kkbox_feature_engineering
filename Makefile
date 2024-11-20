.PHONY: all download sessionize notebooks

all: download sessionize notebooks

download:
	python download.py

sessionize:
	python sessionize.py

notebooks:
	jupytext notebooks/plot_sessions.py --output notebooks/plot_sessions.ipynb
	jupyter nbconvert notebooks/plot_sessions.ipynb --execute --to html
