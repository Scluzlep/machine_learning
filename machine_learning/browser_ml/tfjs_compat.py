"""Course-oriented TensorFlow / tf.keras API compatibility in Pyodide.

The custom Pyodide kernel installs this module automatically on the first
`import tensorflow`; course notebooks keep their original Python syntax.
Unsupported operations fail explicitly rather than returning wrong math.
"""
from __future__ import annotations

import sys
import types
import numpy as np
from js import globalThis
from . import tfjs_bridge as bridge


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def __repr__(self):
        return repr(self.value)

    def __getitem__(self, item):
        return self.value[item]


def _sigmoid(x):
    v = np.asarray(x, dtype=np.float64)
    return _Tensor(1 / (1 + np.exp(-np.clip(v, -100, 100))))


def _softmax(x, axis=-1):
    v = np.asarray(x, dtype=np.float64)
    z = np.exp(v - np.max(v, axis=axis, keepdims=True))
    return _Tensor(z / np.sum(z, axis=axis, keepdims=True))


class Input:
    def __init__(self, shape=None, **kwargs):
        self.shape = tuple(shape) if shape is not None else None


class Normalization:
    def __init__(self, axis=-1, **kwargs):
        if axis != -1:
            raise NotImplementedError("Normalization currently supports axis=-1")
        self.mean = None
        self.std = None

    def adapt(self, data):
        data = np.asarray(data, dtype=np.float32)
        self.mean = data.mean(axis=0)
        self.std = np.sqrt(data.var(axis=0) + 1e-7)

    def __call__(self, data):
        if self.mean is None:
            raise ValueError("Call adapt() before using Normalization")
        return (np.asarray(data, dtype=np.float32) - self.mean) / self.std


class LeakyReLU:
    """Import-compatible activation; model-layer support is not yet enabled."""
    def __init__(self, negative_slope=0.3, alpha=None, **kwargs):
        self.alpha = float(negative_slope if alpha is None else alpha)

    def __call__(self, x):
        x = np.asarray(x)
        return np.maximum(x, self.alpha * x)


class Adam:
    def __init__(self, learning_rate=0.001, lr=None, **kwargs):
        self.learning_rate = float(lr if lr is not None else learning_rate)


class _Loss:
    tfjs_name = None
    def __init__(self, from_logits=False, **kwargs):
        if kwargs:
            raise TypeError("Unsupported loss options: " + ", ".join(kwargs))
        self.from_logits = bool(from_logits)


class BinaryCrossentropy(_Loss):
    tfjs_name = "binaryCrossentropy"


class MeanSquaredError(_Loss):
    tfjs_name = "meanSquaredError"


class SparseCategoricalCrossentropy(_Loss):
    tfjs_name = "sparseCategoricalCrossentropy"


class CategoricalCrossentropy(_Loss):
    tfjs_name = "categoricalCrossentropy"


class History:
    """Keras History object, exposing history.history['loss']."""
    def __init__(self, data):
        self.history = data

    def __repr__(self):
        return f"History({self.history!r})"


class Sequential:
    def __init__(self, layers=None, name="model"):
        layer_defs = []
        input_shape = None
        for layer in layers or []:
            if isinstance(layer, Input):
                input_shape = layer.shape
            elif isinstance(layer, bridge.Dense):
                if not layer_defs and input_shape is not None and layer.input_shape is None:
                    layer.input_shape = input_shape
                layer_defs.append(layer)
            else:
                raise TypeError(
                    f"Unsupported layer {type(layer).__name__}; "
                    "this adapter currently trains Dense networks"
                )
        self._model = bridge.Sequential(layer_defs, name=name)

    @property
    def layers(self):
        return self._model.layers

    def add(self, layer):
        if not isinstance(layer, bridge.Dense):
            raise TypeError("Only Dense layers can be added")
        self._model.add(layer)

    def summary(self):
        self._model.summary()

    def get_layer(self, name=None, index=None):
        layers = self.layers
        if index is not None:
            return layers[index]
        for layer in layers:
            if layer.name == name:
                return layer
        raise ValueError(f"No layer named {name!r}")

    def get_weights(self):
        return self._model.get_weights()

    def compile(self, loss=None, optimizer=None, metrics=None, **kwargs):
        if kwargs:
            raise TypeError("Unsupported compile arguments: " + ", ".join(kwargs))
        if loss is None:
            loss = MeanSquaredError()
        if optimizer is None:
            optimizer = Adam()
        if not isinstance(optimizer, Adam):
            raise TypeError("Only Adam is currently supported")
        if isinstance(loss, _Loss):
            if loss.from_logits:
                fn_name = {
                    BinaryCrossentropy: "binaryCrossentropyFromLogits",
                    SparseCategoricalCrossentropy: "sparseCategoricalCrossentropyFromLogits",
                }.get(type(loss))
                if not fn_name:
                    raise NotImplementedError(
                        f"{type(loss).__name__}(from_logits=True) is not supported yet"
                    )
                loss_value = getattr(globalThis.mlCourseLosses, fn_name)
            else:
                loss_value = loss.tfjs_name
        elif isinstance(loss, str):
            loss_value = loss
        else:
            raise TypeError("Unsupported loss: " + repr(loss))
        self._model.compile(loss=loss_value,
                            learning_rate=optimizer.learning_rate, metrics=metrics)

    async def fit(self, X, y, epochs=10, batch_size=32,
                  verbose=0, shuffle=True, **kwargs):
        if kwargs:
            raise TypeError("Unsupported fit arguments: " + ", ".join(kwargs))
        data = await self._model.fit(X, y, epochs=epochs,
                                     batch_size=batch_size, verbose=verbose,
                                     shuffle=shuffle)
        return History(data)

    def predict(self, X, **kwargs):
        return self._model.predict(X, **kwargs)

    def __call__(self, X, **kwargs):
        return _Tensor(self.predict(X))

    def dispose(self):
        self._model.dispose()


def _module(name, **values):
    module = types.ModuleType(name)
    module.__dict__.update(values)
    sys.modules[name] = module
    return module


async def install(backend="auto"):
    """Expose the supported TensorFlow 2/Keras course API as Python modules."""
    await bridge.load_tfjs(backend=backend)
    await bridge._load_script(bridge._asset_url("course-losses.js"))
    layers = _module("tensorflow.keras.layers",
                     Dense=bridge.Dense, Input=Input, Normalization=Normalization,
                     LeakyReLU=LeakyReLU)
    activations = _module(
        "tensorflow.keras.activations",
        sigmoid=_sigmoid, relu=lambda x: _Tensor(np.maximum(x, 0)),
        linear=lambda x: _Tensor(np.asarray(x)),
        softmax=_softmax
    )
    losses = _module(
        "tensorflow.keras.losses",
        BinaryCrossentropy=BinaryCrossentropy,
        MeanSquaredError=MeanSquaredError,
        SparseCategoricalCrossentropy=SparseCategoricalCrossentropy,
        CategoricalCrossentropy=CategoricalCrossentropy,
    )
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
