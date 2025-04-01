"""Deepchem Dataset"""

from collections.abc import Mapping

import numpy as np

from ..utils import match, as_list, update_copy
from .dataset import Dataset, DictDataset, TeachableWrapperDataset

# pylint: disable=import-outside-toplevel

def is_good_feature(x):
    #breakpoint()
    if x is None:
        return False
    try:
        if len(x) == 0 and isinstance(x, np.ndarray):
            return False
        return True
    except TypeError:
        return True


class DeepChemDataset(DictDataset):
    """
    DeepChem dataset

    Some common featurizers:

        Keras `GraphConvModel`s use the `ConvMolFeaturizer`, which may be
            abbreviated to `'convmol'` in the `featurizer` argument here.

        Pytorch `GCNModel`s use the `MolGraphConvFeaturizer`, which may be
            abbreviated to `'molgraph'` here.
    """

    def __init__(self, data=None, *args, featurizer='dummy', bdim=1, remove_feat_errors=True, **kwargs):  # NOSONAR
        import deepchem as dc
        #breakpoint()

        if data is None:
            data = {}
        elif isinstance(data, dc.data.Dataset):
            dataset = data
            data = {
                "X": dataset.X,
                "ids": dataset.ids,
            }
            try:
                data["y"] = dataset.y
                data["w"] = dataset.w
            except KeyError:
                pass
        data = update_copy(data, kwargs)  # NOSONAR

        if featurizer is not None:
            data["X"] = self.get_featurizer(featurizer).featurize(data["X"])

        if remove_feat_errors:
            i = np.fromiter((i for i, x in enumerate(data["X"]) if is_good_feature(x)), dtype=int)
            #breakpoint()
            data = {k: v[i] for k, v in data.items()}

        if "ids" not in data:
            data["ids"] = Dataset.from_data(np.arange(len(data["X"])))
        if "y" in data and "w" not in data:
            data["w"] = Dataset.from_data(np.ones(len(data["X"]), dtype=np.float32))

        for k, v in data.items():
            if not isinstance(v, Dataset):
                data[k] = Dataset(v)

        super().__init__(data, *args, bdim=bdim, has_Xy=True)

    @staticmethod
    def get_featurizer(f, **kwargs):
        import deepchem as dc

        if f is None:
            return dc.feat.DummyFeaturizer()

        if isinstance(f, dc.feat.Featurizer):
            return f
        elif isinstance(f, type) and issubclass(f, dc.feat.Featurizer):
            return f(**kwargs)
        else:
            return dc.feat.__dict__[
                match(f, dc.feat.__dict__, lambda x, y: x.lower() in y.lower())
            ](**kwargs)

    @staticmethod
    def from_csv(file, X="X", y=None, featurizer=None, **kwargs):
        """
        Loads a DeepChem dataset from a `.csv` file.

        Args:
            X, y (str): Column names for the X and y data

            featurizer: Specifies the DeepChem featurizer to use, if any.
                `featurizer` may be a DeepChem featurizer class, or a featurizer
                instance, *or* a string contained in the classname of a
                featurizer. (Eg., `'convmol'` matches the DeepChem `ConvMolFeaturizer`.)

            **kwargs: These are passed to the featurizer constructor.

        Returns:
            An `alien.data.DeepChemDataset`
        """
        import deepchem as dc

        y = as_list(y)

        loader = dc.data.CSVLoader(
            y, 
            feature_field=X,
            featurizer=DeepChemDataset.get_featurizer(featurizer), 
            **kwargs
        )
        disk_dataset = loader.create_dataset(file)
        data = {
            "X": disk_dataset.X,
            "ids": disk_dataset.ids,
        }
        if y:
            data["y"] = disk_dataset.y
            data["w"] = disk_dataset.w
        return DeepChemDataset(data)

    @staticmethod
    def from_df(df, X="X", y=None, ids="ids", weights=None, featurizer=None, **kwargs):
        """
        Returns a DeepChemDataset built from a Pandas DataFrame.

        :param df: The dataframe to convert
        :param X: The name of the feature column. Defaults to "X".
        :param y: The name of the y/label column, or a list of names for
            multi-prediction. By default, no y values are extracted.
        :param ids: The name of the ids column. By default, looks for a column
            named 'ids', and if none is found, uses the dataframe index.
        :param weights: The name of the weights column. If none is given,
            uses 1.0 for all weights.
        :param featurizer: Specifies the DeepChem featurizer to use, if any.
            `featurizer` may be a DeepChem featurizer class, or a featurizer
            instance, *or* a string contained in the classname of a
            featurizer. (Eg., `'convmol'` matches the DeepChem `ConvMolFeaturizer`.)
        :param **kwargs: Any additional keyword args will become columns in
            the dataset; for example, keyword arg `t='timestamp'`, creates
            a column with key `t` and values taken from `df['timestamp']`.
        """
        import deepchem as dc

        y = as_list(y)

        data_dict = DeepChemDataset._get_data_dict(df, X=X, y=y, ids=ids, **kwargs)

        if len(y) > 0:
            y_cols = []
            for y_col in y:
                if y_col not in df.columns:
                    raise ValueError(f"y-value `{y_col}` is not in the dataframe.")
                y_cols.append(df[y_col].values)
            data_dict["y"] = np.stack(y_cols, axis=1)

            if weights is not None:
                if weights in df.columns:
                    data_dict["w"] = df[weights].values
                else:
                    raise ValueError(f"Dataframe doesn't contain the weights column '{weights}'.")
            else:
                data_dict["w"] = np.ones(len(df), dtype=float)

        if featurizer is not None:
            data_dict["X"] = DeepChemDataset.get_featurizer(featurizer).featurize(data_dict["X"])

        return DeepChemDataset(data_dict)

    @staticmethod
    def _get_data_dict(df, X="X", ids="ids", **kwargs):
        data_dict = {}

        if X in df.columns:
            data_dict["X"] = df[X].values
        elif "X" in df.columns:
            data_dict["X"] = df["X"].values
        else:
            raise ValueError(
                f"Your dataframe must have either an 'X' column, or a user-specified X column. \nInstead, you have columns:\n{df.columns}"
            )

        if ids in df.columns:
            data_dict["ids"] = df[ids].values
        elif "ids" in df.columns:
            data_dict["ids"] = df.ids.values
        else:
            data_dict["ids"] = df.index.values

        for k, c in kwargs.items():
            data_dict[k] = df[c].values
        return data_dict

    def _to_DC(self):
        import deepchem as dc

        return dc.data.NumpyDataset(**{k: v.data for k, v in self.data.items()})

    # def append(self, x):
    #    if 'ids' not in x:
    #        warn("Failed to include 'ids' key in DeepChemDataset.append")
    #        if len(self) == 0:
    #            x['ids'] = 0
    #        elif isinstance[self.data['ids'][0]] and


def as_DCDataset(data):
    """Convert data to a DeepChem dataset."""
    import deepchem

    if isinstance(data, deepchem.data.Dataset):
        return data

    if isinstance(data, TeachableWrapperDataset):
        data = data.data

    if isinstance(data, Mapping):
        data = {k: np.asarray(v) for k, v in data.items()}
        if "ids" not in data:
            data["ids"] = np.arange(len(next(iter(data.values()))))
    else:
        data = {"X": np.asarray(data), "ids": np.arange(len(data))}

    return deepchem.data.NumpyDataset(**data)
