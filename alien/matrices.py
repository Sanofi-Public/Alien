from copy import copy, deepcopy

from .tumpy import tumpy as tp
from .utils import expand_ellipsis, new_shape

None_slice = slice(None)


class LazyMatrix:
    def __init__(
        self, shape, lazy_dim=0, buffer_size=None, use_buffer=True, buffer=None, device=None, preindex=None, **kwargs
    ):
        self.shape = shape
        self.ndim = len(shape)
        self.lazy_dim = lazy_dim
        self.use_buffer = use_buffer
        self.preindex = tp.asarray(preindex) if preindex is not None else None
        if use_buffer and buffer_size:
            if buffer is None:
                self.buffer_size = buffer_size or 1000
                self.buffer_shape = shape[:lazy_dim] + (self.buffer_size,) + shape[lazy_dim + 1 :]
                self.buffer = tp.to(tp.empty(self.buffer_shape, **kwargs), device)
                self.buffer = tp.to(self.buffer, device)
            else:
                buffer = tp.asarray(buffer)
                self.buffer_shape = buffer.shape
                self.buffer_size = buffer.shape[lazy_dim]
                self.buffer = buffer
            self.buffer_view = tp.moveaxis(self.buffer, lazy_dim, 0)
            self.m2b_indices = tp.to(tp.full(shape[lazy_dim], -1, dtype=tp.int64), device)  # matrix-to-buffer indices
            self.b2m_indices = tp.to(tp.arange(self.buffer_size, dtype=tp.int64), device)

            self._diagonal = None
        else:
            self.buffer = None
            self.buffer_view = None

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        indices = expand_ellipsis(indices, self.ndim)
        u2m_indices = indices[self.lazy_dim]
        return_bool, return_val = self._early_return(indices, u2m_indices)
        if return_bool:
            return return_val

        is_single_index = False
        if isinstance(u2m_indices, slice):
            start = u2m_indices.start or 0
            stop = (
                self.shape[self.lazy_dim]
                if u2m_indices.stop is None
                else min(u2m_indices.stop, self.shape[self.lazy_dim])
            )
            step = u2m_indices.step or 1
            u2m_indices = tp.to(tp.arange(start, stop, step), self.m2b_indices)
        elif isinstance(u2m_indices, int):
            u2m_indices = tp.asarray([u2m_indices])
            is_single_index = True
        else:
            u2m_indices = tp.asarray(u2m_indices)

        new_buffer_view = tp.empty_like(self.buffer_view)
        new_m2b_indices = tp.full_like(self.m2b_indices, -1)

        u2b_indices = self.m2b_indices[u2m_indices]  # buffer indices the user is requesting
        # breakpoint()
        new_m_indices = tp.unique(u2m_indices[u2b_indices < 0])
        old_m_indices = tp.unique(u2m_indices[u2b_indices >= 0])
        n_new = len(new_m_indices)
        n_old = len(old_m_indices)

        # breakpoint()
        if len(new_m_indices):
            if self.preindex is None:
                # breakpoint()
                new_buffer_view[:n_new] = tp.moveaxis(self.compute(new_m_indices), self.lazy_dim, 0)
            else:
                new_buffer_view[:n_new] = tp.moveaxis(self.compute(new_m_indices), self.lazy_dim, 0)[
                    (None_slice, *self.preindex)
                ]
            self.b2m_indices[:n_new] = new_m_indices
            new_m2b_indices[new_m_indices] = tp.to(tp.arange(0, n_new, dtype=tp.int64), new_m2b_indices)

        new_buffer_view[n_new : n_new + n_old] = self.buffer_view[self.m2b_indices[old_m_indices]]
        self.b2m_indices[n_new : n_new + n_old] = old_m_indices
        new_m2b_indices[old_m_indices] = tp.to(tp.arange(n_new, n_new + n_old), new_m2b_indices)

        # old_tail_indices is all of the matrix indices which are buffered but were not used in this call
        old_tail_indices = tp.argwhere((new_m2b_indices < 0) & (self.m2b_indices >= 0))[
            : self.buffer_size - n_new - n_old, 0
        ]
        i, j = n_new + n_old, n_new + n_old + len(old_tail_indices)
        self.b2m_indices[i:j] = old_tail_indices
        new_m2b_indices[old_tail_indices] = tp.to(tp.arange(i, j), new_m2b_indices)
        new_buffer_view[i:j] = self.buffer_view[self.m2b_indices[old_tail_indices]]
        self.b2m_indices[j:] = -1

        self.buffer_view = new_buffer_view
        self.buffer = tp.moveaxis(self.buffer_view, 0, self.lazy_dim)
        self.m2b_indices = new_m2b_indices

        buffer_indices = (
            indices[: self.lazy_dim]
            + (self.m2b_indices[u2m_indices[0] if is_single_index else u2m_indices],)
            + indices[self.lazy_dim + 1 :]
        )
        return self.buffer[buffer_indices]

    def _early_return(self, indices, u2m_indices):
        if self.lazy_dim > len(indices) or (isinstance(u2m_indices, slice) and u2m_indices == None_slice):
            r = copy(self)
            r.buffer = self.buffer[indices]
            r.buffer_view = tp.moveaxis(r.buffer, self.lazy_dim, 0)
            r.shape = new_shape(self.shape, indices)
            r.preindex = indices
            return True, r

        if self.buffer is None:
            out = self.compute(indices[self.lazy_dim])
            indices = (
                indices[: self.lazy_dim]
                + (() if out.ndim < self.ndim else (None_slice,))
                + indices[self.lazy_dim + 1 :]
            )
            return True, out[indices]
        return False, None

    def compute(self, indices):
        raise NotImplementedError

    def compute_diagonal(self):
        raise NotImplementedError()

    def __len__(self):
        return self.shape[0]

    def diagonal(self):
        if self.buffer is None:
            return self.compute_diagonal()
        elif self._diagonal is None:
            self._diagonal = self.compute_diagonal()
        return self._diagonal

    def realise(self):
        """Computes and returns the full matrix --- beware!"""
        return self.compute(slice(None))
    

class LazyMatrixCopy(LazyMatrix):
    def __init__(self, m, lazy_dim=0, buffer_size=4, **kwargs):
        super().__init__(
            m.shape, lazy_dim=lazy_dim, buffer_size=buffer_size, dtype=m.dtype, device=tp.device(m), **kwargs
        )
        self.m = m

    def compute(self, indices):
        return tp.take(self.m, indices, self.lazy_dim)


class EnsembleMatrix(LazyMatrix):
    def __init__(self, ensemble, shape=None, prior=None, block_size=None, **kwargs):
        ensemble = tp.asarray(ensemble)
        assert ensemble.ndim <= 3
        if shape is None:
            shape = (ensemble.shape[0],) * 2
            if ensemble.ndim == 3:
                shape = shape + (ensemble.shape[2:]) * 2
        super().__init__(shape, lazy_dim=0, **kwargs)
        self.ensemble = ensemble
        self.block_size = block_size
        if prior is not None:
            prior = tp.asarray(prior)
            self.ensemble = self.ensemble * prior.reshape((-1,) + (1,) * (ensemble.ndim - prior.ndim))
