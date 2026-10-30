#!/usr/bin/env python3
"""Ship the Pyodide 314 runtime on the site's own origin.

The npm package contains the WebAssembly runtime and standard library, but not
the large scientific Python wheels. Those retain the version-matched CDN
fallback until the site's separately cached wheel index is built.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
from urllib.request import urlopen
from urllib.parse import urljoin, urlparse
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



def _wheel_closure(packages: dict, roots: tuple[str, ...]) -> set[str]:
    """All official Pyodide wheels needed for the preloaded scientific stack."""
    selected = set()
    todo = list(roots)
    while todo:
        name = todo.pop()
        if name in selected:
            continue
        selected.add(name)
        item = packages.get(name)
        if item:
            todo.extend(item.get("depends") or [])
    return {name for name in selected if name in packages}


def _download_wheel(name: str, metadata: dict, dest: Path) -> int:
    file_name = metadata.get("file_name")
    if not file_name:
        return 0
    parsed = urlparse(file_name)
    if parsed.scheme and parsed.scheme not in ("https",):
        raise ValueError(f"Unsupported Pyodide wheel URL for {name}: {file_name}")
    relative = not parsed.scheme
    if relative and (Path(file_name).name != file_name or file_name.startswith(".")):
        raise ValueError(f"Unsafe Pyodide wheel path: {file_name}")
    url = file_name if not relative else urljoin(FALLBACK, file_name)
    target = dest / Path(parsed.path).name
    digest = metadata.get("sha256")
    if target.is_file() and (not digest or
                             hashlib.sha256(target.read_bytes()).hexdigest() == digest):
        return target.stat().st_size
    for attempt in range(3):
        try:
            with urlopen(url, timeout=90) as response:
                payload = response.read()
            if digest and hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"Pyodide wheel checksum mismatch: {name}")
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(target)
            return len(payload)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Unreachable")


def _vendor_scientific_wheels(dest: Path, lock_file: Path):
    """Serve fixed scientific wheels locally; preserve CDN URLs for other packages."""
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    packages = lock["packages"]
    roots = _wheel_closure(packages, COURSE_PACKAGES)
    missing = [n for n in COURSE_PACKAGES if n not in packages]
    if missing:
        raise RuntimeError(f"Official Pyodide lock lacks requested packages: {missing}")
    wanted = [(name, packages[name]) for name in sorted(roots)
              if packages[name].get("file_name")]
    with ThreadPoolExecutor(max_workers=6) as pool:
        sizes = list(pool.map(lambda pair: _download_wheel(*pair, dest), wanted))
    for name, item in packages.items():
        original = item.get("file_name")
        if not original:
            continue
        if name not in roots and not urlparse(original).scheme:
            # Avoid 404 on an optional package that wasn't bundled.
            item["file_name"] = urljoin(FALLBACK, original)
    lock_file.write_text(json.dumps(lock, ensure_ascii=False, separators=(",", ":")) + "\n",
                         encoding="utf-8")
    print(f"Bundled {len(wanted)} official Pyodide scientific wheels "
          f"({sum(sizes)/1024**2:.1f} MiB); other packages keep CDN fallback")


def vendor(dist: Path, npm_package: Path, base_path: str) -> None:
    dest = dist / "pyodide"
    dest.mkdir(parents=True, exist_ok=True)
    for file in CORE:
        src = npm_package / file
        if not src.is_file():
            raise FileNotFoundError(f"Missing official Pyodide {PYODIDE_VERSION} asset: {src}")
        shutil.copyfile(src, dest / file)

    _vendor_scientific_wheels(dest, dest / "pyodide-lock.json")

    settings_file = dist / "jupyter-lite.json"
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    config = data.setdefault("jupyter-config-data", {})
    plugins = config.setdefault("litePluginSettings", {})
    kernel = plugins.setdefault("@jupyterlite/pyodide-kernel-extension:kernel", {})
    kernel["pyodideUrl"] = f"{base_path.rstrip('/')}/pyodide/pyodide.mjs"
    options = kernel.setdefault("loadPyodideOptions", {})
    options["packageBaseUrl"] = f"{base_path.rstrip('/')}/pyodide/"
    options["packages"] = list(COURSE_PACKAGES)
    settings_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Pyodide runtime hosted on", kernel["pyodideUrl"])
    print("Pyodide scientific package source:", options["packageBaseUrl"])
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
