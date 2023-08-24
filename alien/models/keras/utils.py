"""Helper functions for """
from collections.abc import MutableSequence, Hashable
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
    if noise_shape is None and training and type(training) == int:
        noise_shape = list(inputs.shape)
    if noise_shape is not None:
        noise_shape = [N if N else I for N, I in zip(noise_shape, inputs.shape)]

        if is_one(training):
            noise_shape[0] = 1
            # print(f"Doing MC dropout with noise shape {noise_shape}")

        noise_shape = tf.convert_to_tensor(noise_shape)

    # print(f"{noise_shape = }")

    def dropped_inputs():
        return self._random_generator.dropout(inputs, self.rate, noise_shape=noise_shape)

    return control_flow_util.smart_cond(training, dropped_inputs, lambda: tf.identity(inputs))


def humble_batchnorm_call(self, inputs, training=None):
    if is_one(training):
        training = False
    return self._hidden_call(inputs, training)


def get_mod_layers(mod):
    if isinstance(mod, MutableSequence):
        return mod
    elif hasattr(mod, "layers"):
        return mod.layers
    elif hasattr(mod, "__dict__"):
        return mod.__dict__.values()
    else:
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
        (bool) Whether or not it encountered any of the modules in
            `skip`
    """
    if only_layers:
        import tensorflow as tf

    to_read = [module]
    skip = {id(x) for x in skip}
    seen = set()

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
    import tensorflow as tf

    if isinstance(obj, tf.keras.layers.Dropout):
        if obj.__class__ == tf.keras.layers.SpatialDropout1D:
            obj.noise_shape = (None, 1, None)
        elif obj.__class__ == tf.keras.layers.SpatialDropout2D:
            obj.noise_shape = (
                (None, None, 1, 1) if obj.data_format == "channels_first" else (None, 1, 1, None)
            )
        elif obj.__class__ == tf.keras.layers.SpatialDropout3D:
            obj.noise_shape = (
                (None, None, 1, 1, 1)
                if obj.data_format == "channels_first"
                else (None, 1, 1, 1, None)
            )
        obj.call = dropout_call.__get__(obj)
        return True

    elif isinstance(obj, tf.keras.layers.BatchNormalization):
        obj._hidden_call = obj.call
        obj.call = humble_batchnorm_call.__get__(obj)

    return False

