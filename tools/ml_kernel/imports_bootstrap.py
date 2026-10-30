"""Discover Python dependencies imported through local course helper modules.

Pyodide's built-in package detector sees imports in the *cell*, not inside
`lab_utils_uni.py`. Before a cell is executed, inspect its local imports
recursively and install missing pure-Python browser dependencies.
"""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

# Other packages, such as numpy/scipy/matplotlib, are handled by Pyodide's
# native package loader when imported directly in a cell.
_MANAGED = frozenset({"ipywidgets", "ipympl", "seaborn"})
_MAX_MODULES = 48
_MAX_BYTES = 1_000_000


def _parse(code: str):
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError):
        # Support a cell with IPython magics and ordinary imports.
        ordinary = "\n".join(
            line for line in code.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        try:
            return ast.parse(ordinary)
        except (SyntaxError, ValueError):
            return None


def _imports(tree):
    if tree is None:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _local_path(directory: Path, modname: str):
    parts = modname.split(".")
    if not all(p.isidentifier() for p in parts):
        return None
    target = directory.joinpath(*parts)
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if candidate.is_file() and candidate.stat().st_size <= _MAX_BYTES:
            return candidate
    return None


def discover(code: str, directory: Path | None = None) -> set[str]:
    """Find optional packages needed by imports in a cell and its local helpers."""
    directory = directory or Path.cwd()
    targets = set()
    todo = [(code, directory)]
    seen = set()

    # Handle notebook magic prior to IPython's input transformations.
    if re.search(r"^\s*%matplotlib\s+(?:widget|ipympl)\b", code, re.MULTILINE):
        targets.update(("ipywidgets", "ipympl"))

    while todo and len(seen) < _MAX_MODULES:
        source, folder = todo.pop()
        for name in _imports(_parse(source)):
            root = name.split(".", 1)[0]
            if root in _MANAGED:
                targets.add(root)
                continue
            module = _local_path(folder, name)
            if module is None and "." in name:
                module = _local_path(folder, root)
            if module is None:
                continue
            key = module.resolve()
            if key in seen:
                continue
            seen.add(key)
            todo.append((module.read_text(encoding="utf-8", errors="replace"), module.parent))
    if "ipympl" in targets:
        targets.add("ipywidgets")
    return targets


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


async def install_missing(code: str):
    targets = discover(code)
    missing = sorted(p for p in targets if not _installed(p))
    if missing:
        # piplite is initialized by JupyterLite before the kernel starts.
        import piplite

        await piplite.install(missing)
        for name in missing:
            if not _installed(name):
                raise ModuleNotFoundError(
                    f"{name} installation completed but Python cannot import it"
                )
