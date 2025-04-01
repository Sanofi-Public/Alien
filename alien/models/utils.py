import sys

BASE_MODEL_SEARCH_DEPTH = 10


def get_base_model(model, framework=None):
    """
    Dig down into `model` to return the underlying Pytorch or Keras implementation.
    If `model` is not itself a Pytorch or Keras model, returns

        get_base_model(model.model, ...)

    Not super fast. Use in setup code.

    Args:
        framework (str, None): The underlying framework to look for. Can be
            `'torch'`, `'keras'` or `None` (automatic). Defaults to `None`.
    """
    abc = set()

    if framework is None:
        if "torch" in sys.modules:
            import torch

            abc.add(torch.nn.Module)
        if "tensorflow" in sys.modules:
            import tensorflow as tf

            abc.update({tf.keras.Model, tf.keras.layers.Layer})
    elif framework == "torch":
        import torch

        abc.add(torch.nn.Module)
    elif framework == "keras":
        import tensorflow as tf

        abc.update({tf.keras.Model, tf.keras.layers.Layer})
    else:
        raise ValueError("`framework` should be one of 'torch', 'keras', or None.")
    assert abc, "You must have Pytorch or Tensorflow to use `get_base_model`."

    return _get_model_helper(model, abc)


def _get_model_helper(model, abc):
    for _ in range(BASE_MODEL_SEARCH_DEPTH):
        if any(isinstance(model, c) for c in abc):
            return model
        try:
            model = model.model
        except AttributeError as exc:
            if isinstance(model, str):
                return model
            raise ValueError("Model doesn't have an underlying Pytorch or Keras base model.") from exc
