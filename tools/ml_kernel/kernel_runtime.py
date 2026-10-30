"""Transparent TensorFlow.js compatibility for unmodified course cells.

This runs inside a customized Pyodide kernel before IPython executes a cell.
The .ipynb document is never changed.
"""
from __future__ import annotations

import ast
import re
import sys

_TF_MODEL_NAMES: set[str] = set()
_COURSE_DEPENDENCIES_READY = False


async def initialize_course_dependencies():
    """Load the course's Python UI packages once per kernel, not per cell.

    Large compiled scientific packages are preloaded by Pyodide load options.
    ipympl and ipywidgets are pure-Python wheels loaded from JupyterLite's
    package index.  The Notebook source is left untouched.
    """
    global _COURSE_DEPENDENCIES_READY
    if _COURSE_DEPENDENCIES_READY:
        return
    import piplite
    from .imports_bootstrap import _register_ipympl_backends

    await piplite.install(["ipywidgets", "ipympl", "seaborn"])
    _register_ipympl_backends()
    from .chinese_font import install as install_chinese_font
    await install_chinese_font()
    _COURSE_DEPENDENCIES_READY = True




def _is_tf_constructor(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "Sequential"
    return isinstance(func, ast.Attribute) and func.attr == "Sequential"


class _ModelFitTransformer(ast.NodeTransformer):
    def __init__(self, names: set[str]):
        self.names = names
        self.changed = False

    def visit_FunctionDef(self, node):
        return node

    def visit_AsyncFunctionDef(self, node):
        return node

    def visit_Lambda(self, node):
        return node

    def visit_ClassDef(self, node):
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        func = node.func
        if (isinstance(func, ast.Attribute)
                and func.attr == "fit"
                and isinstance(func.value, ast.Name)
                and func.value.id in self.names):
            self.changed = True
            return ast.copy_location(ast.Await(node), node)
        return node


def transform_tf_fit(code: str) -> str:
    """Only transform calls on variables known to hold TF.js Keras models."""
    if "Sequential" not in code and (".fit(" not in code or not _TF_MODEL_NAMES):
        return code
    if re.search(r"^\s*[%!]", code, re.MULTILINE):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    for node in tree.body:
        if isinstance(node, ast.Assign) and _is_tf_constructor(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    _TF_MODEL_NAMES.add(target.id)
        elif isinstance(node, ast.AnnAssign) and _is_tf_constructor(node.value):
            if isinstance(node.target, ast.Name):
                _TF_MODEL_NAMES.add(node.target.id)
    if not _TF_MODEL_NAMES or ".fit(" not in code:
        return code
    transformer = _ModelFitTransformer(_TF_MODEL_NAMES)
    new_tree = transformer.visit(tree)
    if not transformer.changed:
        return code
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


async def prepare_cell(code: str) -> str:
    if not isinstance(code, str):
        return code
    await initialize_course_dependencies()
    # Resolve hidden imports such as lab_utils_uni -> ipywidgets before
    # executing the original unchanged course cell.
    from .imports_bootstrap import install_missing
    await install_missing(code)
    if (re.search(r"(?:^|\n)\s*(?:import tensorflow|from tensorflow\b)", code)
            and "tensorflow" not in sys.modules):
        # The kernel's regular dependency loader runs after this hook.
        # NumPy is needed immediately by the TensorFlow.js bridge.
        from pyodide_js import loadPackagesFromImports
        await loadPackagesFromImports("import numpy")
        from .auto_tfjs.tfjs_compat import install
        await install()
    if "tensorflow" not in sys.modules:
        return code
    return transform_tf_fit(code)
