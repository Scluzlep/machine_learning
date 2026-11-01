"""Opt-in, browser-only PyTorch-compatible API for JupyterLite.

This is NOT the native PyTorch distribution. It loads the pinned torch-pyodide
JavaScript/WebGPU runtime and matching pure-Python wheel from our own static
site, only for cells containing an explicit torch import. No course Notebook
is rewritten and regular kernel startup does not fetch PyTorch assets.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys

TORCH_VERSION = "0.0.84"
TORCH_RUNTIME_URL = "/machine_learning/vendor/torch/runtime.mjs"
TORCH_WHEEL_URL = (
    "/machine_learning/vendor/torch/"
    f"torch_pyodide-{TORCH_VERSION}-py3-none-any.whl"
)

# Supports "import torch", "import torch.nn as nn", "from torch import nn"
# and "from torch.optim import Adam"; does not trigger on incidental prose.
_TORCH_IMPORT = re.compile(
    r"(?m)^\s*(?:import\s+torch\b|from\s+torch(?:\.[\w]+)*\s+import\s+)"
)
_ready = False


def needs_torch(code: str) -> bool:
    return bool(_TORCH_IMPORT.search(code))


async def ensure_torch() -> None:
    """Install runtime before the Python package, as required by upstream."""
    global _ready
    if _ready or "torch" in sys.modules:
        return
    try:
        from pyodide.code import run_js

        # Runs inside the Pyodide kernel Worker. Dynamic import and runtime
        # installation must happen in the SAME Worker/globalThis as Pyodide.
        script = (
            "import(" + json.dumps(TORCH_RUNTIME_URL) + ")"
            ".then(mod => mod.installTorchRuntime(globalThis))"
        )
        await run_js(script)

        import piplite

        await piplite.install(TORCH_WHEEL_URL)
        if importlib.util.find_spec("torch") is None:
            raise ModuleNotFoundError("torch-pyodide wheel did not expose torch")
        _ready = True
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize experimental browser PyTorch. "
            "This requires the self-hosted torch-pyodide runtime and wheel; "
            "tensor operations also require WebGPU (not CUDA). "
            "For complete CPU/CUDA PyTorch use a remote Python kernel."
        ) from exc


async def prepare_torch(code: str) -> None:
    if needs_torch(code):
        await ensure_torch()
