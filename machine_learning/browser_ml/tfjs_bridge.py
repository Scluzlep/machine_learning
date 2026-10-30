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


async def load_tfjs(url=None):
    """Load the site's local TF.js bundle in the Pyodide worker."""
    global _tf
    if _tf is not None:
        return _tf
    if globalThis is None:
        raise RuntimeError("This bridge needs the Pyodide browser kernel")
    try:
        _tf = globalThis.tf
    except AttributeError:
        target = url or _asset_url()
        try:
            globalThis.importScripts(target)
        except Exception:
            response = await globalThis.fetch(target)
            if not response.ok:
                raise RuntimeError(f"TensorFlow.js download failed: {target} (HTTP {response.status})")
            globalThis.eval(await response.text())
        try:
            _tf = globalThis.tf
        except AttributeError as exc:
            raise RuntimeError(f"TensorFlow.js did not initialize from {target}") from exc
    await _tf.ready()
    return _tf


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
