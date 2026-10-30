# Machine Learning Notebooks (AutoDL snapshot)

Archived machine-learning course notebooks from an AutoDL shared environment, with accompanying data, images, helper Python modules and a record of the original Python dependencies.

> This is a course material snapshot, **not an official Stanford or DeepLearning.AI publication**. Original course material may be subject to third-party terms; verify redistribution rights before making the materials public. No license grant for the original educational content is implied.

## Contents

- `machine_learning/` — 42 substantive `.ipynb` files in seven lesson groups, plus helper modules and datasets.
- `machine_learning/environment/` — captured Python/Conda dependency lists from AutoDL (Python 3.8.10, TensorFlow 2.6.0).
- `site-requirements.txt` — **different**: tools to build a static JupyterLite website, not packages used by the old AutoDL instance.
- `docs/archive-diff.md` — comparison of the two provided archives.
- `docs/build-feasibility.md` — JupyterLite/Pyodide compatibility notes.

The exported `environment.yml` was **empty** in the provided archive and is excluded. `requirements.txt` includes local `file:///...` references and should not be passed unchanged to pip in a new environment.

## Build a static JupyterLite site

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r site-requirements.txt
jupyter lite build --contents machine_learning --output-dir dist
python -m http.server 8000 -d dist
```

Browse `http://localhost:8000/lab/`. This runs Python inside browsers using Pyodide; **the original TensorFlow 2.6.0 package cannot be installed as a native browser package**, and some interactive plots or extension-dependent labs will need adaptation. Building the site is not equivalent to verifying that every Notebook runs.

## GitHub Pages

The included GitHub Actions workflow builds `machine_learning/` and publishes static files to GitHub Pages on pushes to `main`. In repository **Settings → Pages**, choose **GitHub Actions** as the deployment source.

Expected URL after successful deployment: `https://scluzlep.github.io/machine_learning/`.

## Preserving the original environment

The original `machine_learning/environment/requirements.txt` and `conda-packages.txt` record the installed environment as observed, including potentially inconsistent versions (TensorFlow 2.6.0 vs `tensorflow-estimator` 2.15.0). Do not use these old pinned versions in a browser kernel.
