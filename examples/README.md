# physics-lint examples

- `broken_model_gallery.ipynb` — three MSE-vs-physics-lint disagreement
  cases per release criterion 7 (Week 4 Day 4). Cases 1 and 2 are
  constructed pathologies (synthetic prediction arrays) labelled after
  real failure modes observed on trained neural PDE surrogates;
  Case 3 is a real trained model
  (`physics_lint.validation.broken_cnn`). See the notebook's introduction
  for the rationale.
- `broken_model_gallery.md` — jupytext-flavoured markdown source for
  the same notebook. Either file is authoritative; edit whichever and
  regenerate the other via `jupytext --sync` or `jupytext --to ipynb`.

**Running the gallery.** From the repo root:

```bash
pip install -e ".[dev]"
pip install jupytext jupyterlab
jupyter lab examples/broken_model_gallery.ipynb
```

Case 3 trains two 2-layer CNNs from scratch (~30 seconds on CPU); Cases
1-2 are immediate. No external checkpoints, no GPU required.
