"""Static tests for opt-in PyTorch package loading and vendoring.

End-to-end tensor execution is a separate browser/WebGPU test: this unit test
does not claim that Pyodide can run upstream native PyTorch or CUDA.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


torch_runtime = load("course_torch_runtime", "tools/ml_kernel/torch_runtime.py")
vendor_torch = load("course_vendor_torch", "tools/ml_kernel/vendor_torch.py")

assert torch_runtime.needs_torch("import torch")
assert torch_runtime.needs_torch("import torch.nn as nn")
assert torch_runtime.needs_torch("from torch.optim import Adam")
assert torch_runtime.needs_torch("from torch import Tensor")
assert not torch_runtime.needs_torch("print('import torch')")
assert not torch_runtime.needs_torch("import torchvision")
assert not torch_runtime.needs_torch("import tensorflow as tf")
assert not torch_runtime.needs_torch("torch.tensor([1])")


async def fake_run_js(code):
    assert "/machine_learning/vendor/torch/runtime.mjs" in code
    assert "installTorchRuntime(globalThis)" in code


class FakePiplite(types.ModuleType):
    async def install(self, url):
        assert url == torch_runtime.TORCH_WHEEL_URL


async def test_loader():
    fake_code = types.ModuleType("pyodide.code")
    fake_code.run_js = fake_run_js
    fake_pyodide = types.ModuleType("pyodide")
    fake_pyodide.code = fake_code
    orig_spec = torch_runtime.importlib.util.find_spec
    with patch.dict(sys.modules, {
        "pyodide": fake_pyodide,
        "pyodide.code": fake_code,
        "piplite": FakePiplite("piplite"),
    }), patch.object(torch_runtime.importlib.util, "find_spec", side_effect=lambda name: object() if name == "torch" else orig_spec(name)):
        torch_runtime._ready = False
        await torch_runtime.ensure_torch()
        assert torch_runtime._ready
        await torch_runtime.ensure_torch()  # no second installation


asyncio.run(test_loader())

# Test strict origin and checksum handling without network access.
try:
    vendor_torch._read("http://example.com/runtime.mjs", vendor_torch.ALLOWED_DOWNLOAD_HOSTS)
except ValueError:
    pass
else:
    raise AssertionError("HTTP must not be accepted")

try:
    vendor_torch._read("https://example.com/runtime.mjs", vendor_torch.ALLOWED_DOWNLOAD_HOSTS)
except ValueError:
    pass
else:
    raise AssertionError("Unknown hosts must not be accepted")

payload = b"test-asset"
with patch.object(vendor_torch, "_read", return_value=payload):
    good_hash = hashlib.sha256(payload).hexdigest()
    assert vendor_torch._checked_asset("https://celsowm.github.io/example", good_hash, vendor_torch.ALLOWED_DOWNLOAD_HOSTS) == payload
    try:
        vendor_torch._checked_asset("https://celsowm.github.io/example", "0" * 64, vendor_torch.ALLOWED_DOWNLOAD_HOSTS)
    except ValueError:
        pass
    else:
        raise AssertionError("Checksum mismatch must be rejected")

print("Torch vendor security + lazy import static tests passed")
