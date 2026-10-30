#!/usr/bin/env python3
"""Regression for hidden course imports and optional Pyodide packages."""
import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parent.parent
p = root / "tools/ml_kernel/imports_bootstrap.py"
spec = importlib.util.spec_from_file_location("imports_bootstrap_test", p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

lesson = root / "machine_learning/1. Linear Regression"
import json
notebook = json.loads((lesson / "4. Gradient Descent for Linear Regression.ipynb").read_text(encoding="utf-8"))
cell = "".join(notebook["cells"][3]["source"])
assert "from lab_utils_uni import" in cell
found = module.discover(cell, lesson)
assert "ipywidgets" in found, (found, cell)
assert module.discover("import numpy as np", lesson) == set()
assert module.discover("%matplotlib widget\nimport matplotlib.pyplot as plt", lesson) == {"ipympl", "ipywidgets"}
assert module.discover("get_ipython().run_line_magic('matplotlib', 'widget')", lesson) == {"ipympl", "ipywidgets"}
assert module.discover("get_ipython().run_line_magic('matplotlib', 'ipympl')", lesson) == {"ipympl", "ipywidgets"}
assert module.discover("get_ipython().run_line_magic('matplotlib', 'inline')", lesson) == set()

assert module.discover("from lab_utils_uni import plt_house_x", lesson) >= {"ipywidgets"}
print("Indirect lab_utils_uni -> ipywidgets dependency: PASS")
print("Matplotlib widget -> ipympl and ipywidgets dependency: PASS")
