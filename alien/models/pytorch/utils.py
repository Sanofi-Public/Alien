"""Helper functions for Pytorch models."""

import numpy as np

from ...utils import is_one

# pylint: disable=import-outside-toplevel


def as_tensor(x):
    """Return the tensor version of x (i.e. itself it it already is ArrayLike, or x.data)"""
    import torch

    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    from ...data import NumpyDataset, TorchDataset

    if isinstance(x, TorchDataset):
        return x.data
    if isinstance(x, NumpyDataset):
        return torch.from_numpy(x.data)
    if hasattr(x, "__array__"):
        return torch.from_numpy(x.__array__())
    return torch.asarray(x)


def dropout_forward(self, x):
    import torch
    import torch.nn.functional as F

    from ...tumpy.torch_bindings import dtype_n_to_t

    if is_one(self.training):
        if seed := getattr(self, "seed", None):
            torch.manual_seed(seed)
        ones = torch.ones(x.shape[1:], dtype=dtype_n_to_t.get(x.dtype, x.dtype), device=x.device, requires_grad=True)
        return x * F.dropout(ones, self.p, True, self.inplace)
    return F.dropout(x, self.p, self.training, self.inplace)


def dropout__getstate__(self):
    state = self.__dict__.copy()
    state.pop("forward", None)
    state.pop("__getstate__", None)
    return state


def submodules(module, include_names=True, skip=frozenset()):
    """
    Iterator through submodules of `module` (paired with their names,
    if `include_names` is True). A submodule is returned only once, on its
    first occurence in a depth-first traversal.

    Any modules in `skip` (given either as the actual module, or its name)
    will be skipped, along with all their submodules.

    Args:
        module (torch.nn.module): The module, whose submodules we will
            iterate through.

        include_names (bool): True, the iterator yields pairs `(name, submodule)`,
            otherwise it yields just `submodule`. (The returned name is the name it's
            indexed as the first time it occurs in the tree. If the submodule is not
            named, its name will return as `None`)

        skip: A collection of modules to skip. Can contain
            either modules themselves, and/or their names.
    """

    skip = set(skip)

    for name, modules in module.named_children():
        if modules not in skip and name not in skip:
            skip.add(modules)
            yield (name, modules) if include_names else modules
            for sub in submodules(modules, include_names=include_names, skip=skip):
                yield sub


pl_argnames = [
    "accelerator",
    "strategy",
    "devices",
    "num_nodes",
    "precision",
    "logger",
    "callbacks",
    "fast_dev_run",
    "overfit_batches",
    "val_check_interval",
    "check_val_every_n_epoch",
    "num_sanity_val_steps",
    "log_every_n_steps",
    "enable_checkpointing",
    "enable_progress_bar",
    "enable_model_summary",
    "accumulate_grad_batches",
    "gradient_clip_val",
    "gradient_clip_algorithm",
    "benchmark",
    "inference_mode",
    "use_distributed_sampler",
    "profiler",
    "detect_anomaly",
    "barebones",
    "plugins",
    "sync_batchnorm",
    "reload_dataloaders_every_n_epochs",
    "default_root_dir",
]
