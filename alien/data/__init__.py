# Preferred invocation is to call TeachableDataset.from_data,
# rather than directly instantiating the subclasses

from .dataset import (
    ArrayDataset,
    Dataset,
    DictDataset,
    NumpyDataset,
    TeachableDataset,
    TorchDataset,
    TupleDataset,
)
from .deepchem import DeepChemDataset, as_DCDataset
