from abc import abstractmethod

import numpy as np

from ..sample_generation.from_set import SetSampleGenerator

# TODO: Docstrings


class Oracle:
    @abstractmethod
    def get_label(self, x, remove=False):
        pass

    def get_labels(self, x, remove=False):
        labels = []
        for x_val in x:
            labels.append(self.get_label(x_val, remove=remove))
        return np.array(labels)


class SetOracle(SetSampleGenerator, Oracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, Xy=True, **kwargs)

    def get_label(self, x, remove=False):
        indices = self.data.find(x, first=False)
        if remove:
            self.remove_data_indices(indices)
        return self.labels[indices[0]]
