# Build the slides

From the `presentation/` folder, run the following with a LaTeX installation
that includes `latexmk`, pdfLaTeX, Beamer and TikZ.

```sh
latexmk -pdf recommender-systems.tex
```

This builds [recommender-systems.pdf](recommender-systems.pdf) using the figure
PDFs already in `figures/`.

To regenerate those figures from the saved experiment results, run the following
from the repository root, then rebuild the slides.

```sh
uv run --locked presentation/figures/plot_mmr.py
```

The plot script reads the results from Git commit `4a252ea`. To recompute the
supplementary exposure and calibration metrics as well, place MovieLens 1M in
`data/ml-1m/` and run `uv run --locked presentation/figures/exposure_calibration.py`
before the plot script.
