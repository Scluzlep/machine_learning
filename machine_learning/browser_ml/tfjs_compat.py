"""Small TensorFlow/Keras compatibility module for the original course notebooks.

The course's original code and plotting helpers may retain tensorflow imports.
Use in a Pyodide Notebook, before importing any original TensorFlow helpers:

    from browser_ml.tfjs_compat import install
    tf = await install()  # choose GPU when the Worker supports WebGPU

Existing model.fit calls must use `await model.fit(...)` (TF.js is asynchronous).
"""
import sys
import types
import numpy as np
from pyodide.ffi import to_js
from js import globalThis
from . import tfjs_bridge as bridge


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return _Tensor(1 / (1 + np.exp(-np.clip(x, -100, 100))))


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return _Tensor(ex / np.sum(ex, axis=-1, keepdims=True))


class Input:
    def __init__(self, shape=None, **kwargs):
        self.shape = tuple(shape) if shape is not None else None


class Normalization:
    def __init__(self, axis=-1, **kwargs):
        if axis != -1:
            raise NotImplementedError("Only axis=-1 normalization is supported")
        self.mean = None
        self.std = None

    def adapt(self, data):
        data = np.asarray(data, dtype=np.float32)
        self.mean = np.mean(data, axis=0)
        self.std = np.sqrt(np.var(data, axis=0) + 1e-7)

    def __call__(self, data):
        if self.mean is None:
            raise ValueError("Call adapt(data) before normalization")
        return (np.asarray(data, dtype=np.float32) - self.mean) / self.std


class Adam:
    def __init__(self, learning_rate=0.001, lr=None, **kwargs):
        self.learning_rate = float(lr if lr is not None else learning_rate)


class _Loss:
    def __init__(self, *args, **kwargs):
        pass


class BinaryCrossentropy(_Loss):
    tfjs_name = "binaryCrossentropy"


class MeanSquaredError(_Loss):
    tfjs_name = "meanSquaredError"


class SparseCategoricalCrossentropy(_Loss):
    tfjs_name = "sparseCategoricalCrossentropy"


class _Layer:
    def __init__(self, js_layer):
        self._layer = js_layer

    def get_weights(self):
        tensors = self._layer.getWeights()
        return [np.asarray(t.arraySync().to_py(), dtype=np.float32) for t in tensors]

    def set_weights(self, values):
        tf = bridge._require_tf()
        tensors = [tf.tensor(to_js(np.asarray(v, dtype=np.float32).tolist())) for v in values]
        try:
            self._layer.setWeights(globalThis.Array.of(*tensors))
        finally:
            for tensor in tensors:
                tensor.dispose()


class Sequential:
    def __init__(self, layers, name="model"):
        layer_defs = []
        input_shape = None
        for layer in layers:
            if isinstance(layer, Input):
                input_shape = layer.shape
            elif isinstance(layer, bridge.Dense):
                if not layer_defs and input_shape is not None and layer.input_shape is None:
                    layer = bridge.Dense(layer.units, layer.activation, input_shape, layer.name)
                layer_defs.append(layer)
            else:
                raise TypeError("This compatibility module currently supports Input and Dense")
        self._model = bridge.Sequential(layer_defs, name=name)

    def summary(self):
        self._model.summary()

    def get_layer(self, name):
        return _Layer(self._model._model.getLayer(name))

    def compile(self, loss=None, optimizer=None, **kwargs):
        if loss is None:
            loss = MeanSquaredError()
        if optimizer is None:
            optimizer = Adam()
        if isinstance(loss, _Loss):
            loss = loss.tfjs_name
        if not isinstance(optimizer, Adam):
            raise TypeError("Only Adam optimizer is supported")
        self._model.compile(loss=loss, learning_rate=optimizer.learning_rate)

    async def fit(self, X, y, epochs=10, batch_size=32, verbose=0, **kwargs):
        return await self._model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, X, **kwargs):
        return self._model.predict(X)

    def dispose(self):
        self._model.dispose()


def _module(name, **values):
    obj = types.ModuleType(name)
    obj.__dict__.update(values)
    sys.modules[name] = obj
    return obj


async def install(backend="auto"):
    """Register enough `tensorflow.keras` modules for original coffee course labs."""
    await bridge.load_tfjs(backend=backend)
    layers = _module("tensorflow.keras.layers",
                     Dense=bridge.Dense, Input=Input, Normalization=Normalization)
    activations = _module("tensorflow.keras.activations",
                          sigmoid=_sigmoid, relu=lambda x: np.maximum(x, 0),
                          linear=lambda x: x)
    losses = _module("tensorflow.keras.losses",
                     BinaryCrossentropy=BinaryCrossentropy,
                     MeanSquaredError=MeanSquaredError,
                     SparseCategoricalCrossentropy=SparseCategoricalCrossentropy)
    optimizers = _module("tensorflow.keras.optimizers", Adam=Adam)
    models = _module("tensorflow.keras.models", Sequential=Sequential)
    keras = _module("tensorflow.keras", Sequential=Sequential, Input=Input,
                    layers=layers, activations=activations, losses=losses,
                    optimizers=optimizers, models=models, __version__="tfjs-compat")
    tf = _module("tensorflow", keras=keras,
                 random=types.SimpleNamespace(set_seed=np.random.seed),
                 autograph=types.SimpleNamespace(set_verbosity=lambda *_: None),
                 nn=types.SimpleNamespace(softmax=_softmax))
    return tf
