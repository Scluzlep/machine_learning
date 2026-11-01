#!/usr/bin/env python3
"""Vendor a version-pinned browser PyTorch wheel and JS runtime.

Downloads the paired artifacts from the upstream versioned manifest, verifies
both SHA-256 values, and publishes them from our own GitHub Pages origin.
The third-party torch-pyodide project is a partial WebGPU implementation,
not a build of upstream native PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

VERSION = "0.0.84"
MANIFEST_URL = (
    "https://celsowm.github.io/torch-pyodide/"
    f"manifests/{VERSION}.json"
)
# Independently fixed to the published PyPI distribution checksum.
WHEEL_SHA256 = "e806b605617af79ffb8094ef04c68070799f3e6c6bf919c08f469d54447c3ffa"
WHEEL_NAME = f"torch_pyodide-{VERSION}-py3-none-any.whl"
ALLOWED_DOWNLOAD_HOSTS = {"celsowm.github.io", "files.pythonhosted.org"}


def _read(url: str, allowed_hosts: set[str]) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError(f"Unapproved dependency download origin: {url}")
    request = Request(url, headers={"User-Agent": "machine-learning-jupyterlite-vendor"})
    with urlopen(request, timeout=90) as response:
        final_url = response.geturl()
        if urlparse(final_url).hostname not in allowed_hosts:
            raise ValueError(f"Unapproved redirect for dependency: {final_url}")
        return response.read()


def _checked_asset(url: str, expected: str, hosts: set[str]) -> bytes:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing/invalid SHA-256 for {url}")
    payload = _read(url, hosts)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {url}: {actual}")
    return payload


def vendor(output: Path) -> None:
    manifest = json.loads(_read(MANIFEST_URL, {"celsowm.github.io"}))
    if manifest.get("torchVersion") != VERSION:
        raise ValueError("torch-pyodide runtime/wheel version mismatch")
    runtime_url = manifest.get("runtimeUrl")
    wheel_url = manifest.get("wheelUrl")
    runtime_hash = manifest.get("runtimeSha256")
    wheel_hash = manifest.get("wheelSha256")
    if not isinstance(runtime_url, str) or not runtime_url.endswith("/runtime.mjs"):
        raise ValueError("Invalid upstream runtime URL")
    if not isinstance(wheel_url, str) or not wheel_url.endswith("/" + WHEEL_NAME):
        raise ValueError("Invalid upstream PyTorch wheel URL")
    if wheel_hash != WHEEL_SHA256:
        raise ValueError("Upstream wheel digest does not match pinned PyPI release")

    runtime = _checked_asset(runtime_url, runtime_hash, ALLOWED_DOWNLOAD_HOSTS)
    wheel = _checked_asset(wheel_url, wheel_hash, ALLOWED_DOWNLOAD_HOSTS)
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime.mjs").write_bytes(runtime)
    (output / WHEEL_NAME).write_bytes(wheel)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "torchVersion": VERSION,
                "runtimeSha256": runtime_hash,
                "wheelSha256": wheel_hash,
                "source": MANIFEST_URL,
                "implementation": "torch-pyodide (experimental WebGPU compatibility)",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Vendored experimental torch-pyodide {VERSION}: "
          f"{len(runtime) + len(wheel)} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/vendor/torch"))
    args = parser.parse_args()
    vendor(args.output)
