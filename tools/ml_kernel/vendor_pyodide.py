#!/usr/bin/env python3
"""Ship the Pyodide 314 runtime on the site's own origin.

The npm package contains the WebAssembly runtime and standard library, but not
the large scientific Python wheels. Those retain the version-matched CDN
fallback until the site's separately cached wheel index is built.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

CORE = (
    "pyodide.mjs",
    "pyodide.asm.mjs",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
)
PYODIDE_VERSION = "314.0.6"
FALLBACK = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"

# Fixed machine-learning course dependencies. Pyodide loads these as part of
# kernel initialization rather than stalling individual Notebook cells.
COURSE_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "micropip",
)



def vendor(dist: Path, npm_package: Path, base_path: str) -> None:
    dest = dist / "pyodide"
    dest.mkdir(parents=True, exist_ok=True)
    for file in CORE:
        src = npm_package / file
        if not src.is_file():
            raise FileNotFoundError(f"Missing official Pyodide {PYODIDE_VERSION} asset: {src}")
        shutil.copyfile(src, dest / file)

    settings_file = dist / "jupyter-lite.json"
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    config = data.setdefault("jupyter-config-data", {})
    plugins = config.setdefault("litePluginSettings", {})
    kernel = plugins.setdefault("@jupyterlite/pyodide-kernel-extension:kernel", {})
    kernel["pyodideUrl"] = f"{base_path.rstrip('/')}/pyodide/pyodide.mjs"
    options = kernel.setdefault("loadPyodideOptions", {})
    options["packageBaseUrl"] = FALLBACK
    options["packages"] = list(COURSE_PACKAGES)
    settings_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Pyodide runtime hosted on", kernel["pyodideUrl"])
    print("Remaining Pyodide scientific package source:", FALLBACK)
    print("Scientific modules initialized with kernel:", ", ".join(COURSE_PACKAGES))
    print("Local runtime size (MiB):",
          round(sum((dest / name).stat().st_size for name in CORE) / 1024**2, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--npm-package", type=Path, default=Path(".pyodide-vendor/node_modules/pyodide"))
    parser.add_argument("--base-path", default="/machine_learning")
    args = parser.parse_args()
    vendor(args.dist, args.npm_package, args.base_path)
