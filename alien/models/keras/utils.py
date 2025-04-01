"""Helper functions for """

from collections.abc import MutableSequence

from ...utils import is_one

# pylint: disable=import-outside-toplevel,protected-access


def dropout_call(self, inputs, training=None):
    """
    If `training` is True but not 1, uses dropout according
    to self.noise_dims. If `training is 1`, uses dropout, but holds it
    fixed along the batch.
    """
    import tensorflow as tf
    from keras.backend import learning_phase
    from keras.utils import control_flow_util

    if self.rate == 0:
        return tf.identity(inputs)

    if training is None:
        training = learning_phase()

    noise_shape = self.noise_shape
    input_shape = tf.shape(inputs)#.numpy().tolist()
    if noise_shape is None and training and isinstance(training, int):
        noise_shape = input_shape
    elif noise_shape is not None:
        noise_shape = [N or I for N, I in zip(noise_shape, input_shape)]

    if is_one(training):
        noise_shape = (1, *noise_shape[1:])

    def dropped_inputs():
        return self._random_generator.dropout(inputs, self.rate, noise_shape=noise_shape)

    return control_flow_util.smart_cond(training, dropped_inputs, lambda: tf.identity(inputs))


def humble_batchnorm_call(self, inputs, training=None):
    if is_one(training):
        training = False
    return self._hidden_call(inputs, training)


def dropout__getstate__(self):
    state = self.__dict__.copy()
    state.pop("call", None)
    state.pop("__getstate__", None)
    return state


def get_mod_layers(mod):
    if isinstance(mod, MutableSequence):
        return mod
    if hasattr(mod, "layers"):
        return mod.layers
    if hasattr(mod, "__dict__"):
        return mod.__dict__.values()
    return []


def subobjects(module, skip=frozenset(), only_layers=True):
    """
    Traverses a module and all of its components.

    Args:
        module (keras.Model): module to traverse
        skip (Container): A collection of modules to skip (along with
            their submodules)
        only_layers: If True, only yields objects which are actually
            Keras layers.

    Returns:
        an iterator over the subobjects of `module`
    """
    if only_layers:
        import tensorflow as tf

    to_read = [module]
    seen = {id(x) for x in skip}
    if id(module) in seen:
        return []
    seen.add(id(module))

    while to_read:
        mod = to_read.pop(0)
        seen.add(id(mod))
        if (not only_layers) or isinstance(mod, tf.keras.layers.Layer):
            yield mod

        for x in get_mod_layers(mod):
            if id(x) not in seen:
                to_read.append(x)


def modify_dropout(obj):
    """
    If `obj` is a Dropout, retools it to do properly correlated
    dropout inference.

    Returns:
        bool: whether `obj` is a Dropout
    """
    # pylint: disable=no-value-for-parameter
    import tensorflow as tf

    obj.__getstate__ = dropout__getstate__.__get__(obj)

    obj.__getstate__ = dropout__getstate__.__get__(obj)

    if isinstance(obj, tf.keras.layers.Dropout):
        if obj.__class__ == tf.keras.layers.SpatialDropout1D:
            obj.noise_shape = tf.TensorShape([None, 1, None])
        elif obj.__class__ == tf.keras.layers.SpatialDropout2D:
            obj.noise_shape = (
                tf.TensorShape([None, None, 1, 1])
                if obj.data_format == "channels_first"
                else tf.TensorShape([None, 1, 1, None])
            )
        elif obj.__class__ == tf.keras.layers.SpatialDropout3D:
            obj.noise_shape = (
                tf.TensorShape([None, None, 1, 1, 1])
                if obj.data_format == "channels_first"
                else tf.TensorShape([None, 1, 1, 1, None])
            )
        obj.call = dropout_call.__get__(obj)
        return True

    if isinstance(obj, tf.keras.layers.BatchNormalization) and not hasattr(obj, "_hidden_call"):
        obj._hidden_call = obj.call
        obj.call = humble_batchnorm_call.__get__(obj)

    return False
