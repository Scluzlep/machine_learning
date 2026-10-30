"""Python bridge to the TensorFlow.js model trainer bundled with this website.

Usage (Pyodide kernel):
    from browser_ml.tfjs_bridge import load_tfjs, Dense, Sequential
    await load_tfjs()
    model = Sequential([Dense(8, "relu", input_shape=(2,)), Dense(1, "sigmoid")])
    model.compile(loss="binaryCrossentropy", learning_rate=0.01)
    history = await model.fit(X, y, epochs=40)
    yhat = model.predict(X)
"""
from __future__ import annotations

from urllib.parse import urlsplit

import numpy as np

try:
    from js import globalThis
    from pyodide.ffi import to_js
except ImportError:
    globalThis = None
    to_js = None

_tf = None
_webgpu_loaded = False
_backend_attempts = []


def _config(**kwargs):
    return to_js(kwargs, dict_converter=globalThis.Object.fromEntries)


def _asset_url():
    loc = urlsplit(str(globalThis.location.href))
    path = loc.path
    for marker in ("/lab/", "/notebooks/", "/tree/", "/api/", "/extensions/", "/build/"):
        if marker in path:
            path = path.split(marker, 1)[0]
            break
    else:
        path = path.rstrip("/").rsplit("/", 1)[0]
    return f"{loc.scheme}://{loc.netloc}{path}/vendor/tf.min.js"



async def _load_script(url):
    """Load a locally hosted UMD bundle into a JupyterLite Pyodide worker."""
    try:
        globalThis.importScripts(url)
    except Exception:
        response = await globalThis.fetch(url)
        if not response.ok:
            raise RuntimeError(f"JavaScript asset failed: {url} (HTTP {response.status})")
        globalThis.eval(await response.text())


def _worker_has_webgpu():
    try:
        return getattr(getattr(globalThis, "navigator", None), "gpu", None) is not None
    except Exception:
        return False


async def _activate_backend(name):
    """Activate the backend and verify that it can run a GPU/CPU operation."""
    tf = _require_tf()
    result = await tf.setBackend(name)
    if not result:
        raise RuntimeError(f"Backend {name} could not be activated")
    await tf.ready()
    x = tf.tensor2d(to_js([[1.0, 2.0], [3.0, 4.0]]))
    try:
        y = x.matMul(x)
        try:
            await y.data()
        finally:
            y.dispose()
    finally:
        x.dispose()
    if str(tf.getBackend()) != name:
        raise RuntimeError(f"Backend {name} not selected")


async def load_tfjs(url=None, backend="auto", webgpu_url=None):
    """Initialize TF.js and choose WebGPU -> WebGL -> CPU.

    Choose backend="webgpu" to require WebGPU (raises if unavailable);
    use backend="cpu" for deterministic CPU comparison.
    """
    global _tf, _webgpu_loaded, _backend_attempts
    if backend not in ("auto", "webgpu", "webgl", "cpu"):
        raise ValueError("backend must be 'auto', 'webgpu', 'webgl' or 'cpu'")
    if globalThis is None:
        raise RuntimeError("TensorFlow.js requires the browser Pyodide kernel")
    if _tf is None:
        try:
            _tf = globalThis.tf
        except AttributeError:
            target = url or _asset_url()
            await _load_script(target)
            try:
                _tf = globalThis.tf
            except AttributeError as exc:
                raise RuntimeError(f"TensorFlow.js did not initialize: {target}") from exc
    _backend_attempts = []
    candidates = ("webgpu", "webgl", "cpu") if backend == "auto" else (backend,)
    for candidate in candidates:
        try:
            if candidate == "webgpu":
                if not _worker_has_webgpu():
                    raise RuntimeError("navigator.gpu unavailable in the kernel Worker")
                if not _webgpu_loaded:
                    await _load_script(webgpu_url or _asset_url("tf-backend-webgpu.min.js"))
                    _webgpu_loaded = True
            await _activate_backend(candidate)
            return _tf
        except Exception as exc:
            _backend_attempts.append({"backend": candidate, "error": str(exc)})
            if backend != "auto":
                raise RuntimeError(f"Unable to use {candidate}: {exc}") from exc
    raise RuntimeError("No usable TensorFlow.js backend: " + str(_backend_attempts))


def backend_info():
    """Show active backend, WebGPU availability and fallback reasons."""
    return {
        "backend": backend(),
        "worker_webgpu_available": _worker_has_webgpu(),
        "fallbacks": list(_backend_attempts),
    }


def _require_tf():
    if _tf is None:
        raise RuntimeError("Call await load_tfjs() first")
    return _tf


def _tensor(value):
    data = np.asarray(value, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape((-1, 1))
    if data.ndim != 2:
        raise ValueError("This adapter accepts 2D numerical arrays")
    return _require_tf().tensor2d(to_js(data.tolist()))


class Dense:
    def __init__(self, units, activation="linear", input_shape=None, name=None):
        self.units = int(units)
        self.activation = activation
        self.input_shape = input_shape
        self.name = name


class Sequential:
    def __init__(self, layers, name="model"):
        tf = _require_tf()
        self._model = tf.sequential(_config(name=name))
        for layer in layers:
            if not isinstance(layer, Dense):
                raise TypeError("Only Dense layers are supported in this adapter")
            opts = {"units": layer.units, "activation": layer.activation}
            if layer.input_shape is not None:
                opts["inputShape"] = list(layer.input_shape)
            if layer.name is not None:
                opts["name"] = layer.name
            self._model.add(tf.layers.dense(_config(**opts)))

    def compile(self, *, optimizer="adam", learning_rate=0.001, loss="meanSquaredError"):
        if optimizer != "adam":
            raise ValueError("Only the Adam optimizer is supported")
        tf = _require_tf()
        self._model.compile(_config(optimizer=tf.train.adam(float(learning_rate)), loss=loss))

    async def fit(self, X, y, *, epochs=10, batch_size=32, verbose=0):
        """Train using TF.js; model.fit is async, so remember to await it."""
        xjs, yjs = _tensor(X), _tensor(y)
        try:
            result = await self._model.fit(xjs, yjs, _config(
                epochs=int(epochs), batchSize=int(batch_size),
                verbose=int(verbose), shuffle=True,
            ))
            return result.history.to_py()
        finally:
            xjs.dispose()
            yjs.dispose()

    def predict(self, X):
        xjs = _tensor(X)
        try:
            pred = self._model.predict(xjs)
            try:
                return np.asarray(pred.arraySync().to_py(), dtype=np.float32)
            finally:
                pred.dispose()
        finally:
            xjs.dispose()

    def summary(self):
        self._model.summary()

    def dispose(self):
        self._model.dispose()


def backend():
    return str(_require_tf().getBackend())
