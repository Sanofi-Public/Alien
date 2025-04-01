"""
Module for computing metrics and plotting.
Confidence intervals, RMSE, scatter plots.
"""

# TODO: I've left TODOs throughout. Still need to look at Scatter and plot_scores.
# - x is undefined in confint

import os
import pickle
import re
from collections.abc import Iterable
from glob import glob
from statistics import median
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from numpy.typing import ArrayLike

from ..data import TeachableDataset
from ..utils import isint


def conf_int(confidence_level, standard_error, len_x: Optional[Union[int, ArrayLike]] = None):
    """Compute confidence interval using a normal or t-distribution

    Args:
        confidence_level (_type_): _description_
        standard_error (_type_): _description_
        len_x (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    if len_x is None:
        return st.norm.interval(confidence_level)[1] * standard_error
    if isint(len_x) and len_x > 1:
        return st.t.interval(confidence_level, len_x - 1)[1] * standard_error

    # TODO: x is undefined here. Not sure where it should come from?
    interval = np.zeros(x.shape[1:])
    if isinstance(len_x, np.ndarray):
        for i in np.ndindex(x.shape[1:]):
            if len_x[i] > 1:
                interval[i] = st.t.interval(confidence_level, len_x[i] - 1, scale=standard_error[i])[1]
    return interval


def sem(x: ArrayLike, axis=0) -> ArrayLike:
    """Compute standard error of the mean.

    Args:
        x (ArrayLike): array to compute SEM for.
        axis (int, optional): Axis in x to make the computation along. Defaults to 0.

    Returns:
        ArrayLike: Array of computed SEMs.
    """
    return np.std(x, axis=axis) / np.sqrt(x.shape[axis])


class Score:
    default_filename = "*.pickle"

    def __init__(
        self,
        x=np.zeros((0,)),
        y=np.zeros((0,)),
        err=None,
        name: Optional[str] = None,
        file_path: Optional[str] = None,
        axes=None,
        plot_args: Optional[Dict] = None,
        scatter=None,
        use_wandb=False,
    ):
        # TODO: module docstring
        # TODO: axes is not used anywhere.
        super().__init__()
        self.x = x if isinstance(x, TeachableDataset) else TeachableDataset.from_data(x)
        self.y = y if isinstance(y, TeachableDataset) else TeachableDataset.from_data(y)
        if err is None or isinstance(err, TeachableDataset):
            self.err = err
        else:
            self.err = TeachableDataset.from_data(err)
        self.name = name
        self.file_path = file_path
        self.plot_args = plot_args if plot_args else {}
        self.score_name = self.__class__.__name__
        self.axes = axes or ("samples", self.score_name)
        self.scatter = scatter
        if scatter and scatter.preds is not None and len(scatter.preds) > 0:
            self.compute(scatter)
        if use_wandb:
            import wandb

            self.wandb = wandb
        else:
            self.wandb = None

    def append(self, x_val: float, y_val: float):
        """Append point to self.x and self.y

        Args:
            x_val (float): x value to append
            y_val (float): y value to append
        """
        self.x.append(x_val)
        self.y.append(y_val)

    def log(self, step=None):
        if self.wandb is not None:
            self.wandb_log()
        if self.file_path is not None:
            self.save()

    def save(self, file_path: Optional[str] = None):
        """Save Score object to given filepath.

        Args:
            file_path (Optional[str], optional): Path to save object. Defaults to None.
        """
        if file_path is None:
            file_path = self.file_path
        assert file_path is not None, "File not provided"
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def wandb_log(self):
        assert self.wandb is not None, "wandb not imported"
        # if not isinstance(step, Iterable):
        #    step = [step]
        # for s in step:
        self.wandb.log({self.axes[1]: self.y[-1], self.axes[0]: self.x[-1]}, commit=False)

    @staticmethod
    def load(file_path: str):
        """Load a Score object

        Args:
            file (str): File path location.

        Returns:
            Score: Score object to load.
        """
        if "*" in file_path or os.path.isdir(file_path):
            return Score.average_runs(file_path)
        # with open(file_path, "rb") as f:
        #     x = pickle.load(f)
        # x.file = file_path
        # return x
        raise NotImplementedError("We don't load pickle files natively. Ff you would like to, use pickle.load(f)")

    @staticmethod
    def load_many(*scores, filename=default_filename) -> List:
        """Loads many Scores at once, and returns a list.

        Args:
            *args: several filenames, or one filename with wildcards, or
                one folder name (which will have wildcards appended)

        Returns:
            list[Score]: _description_
        """
        if isinstance(scores[0], str):
            if len(scores) == 1:
                a = scores[0]
                if "*" not in a:
                    a = os.path.join(a, "**", filename)
                scores = sorted(glob(a, recursive=True))
            scores = [Score.load(f) for f in scores]
        return scores

    @staticmethod
    def average_runs(
        *args,
        length: str = "longest",
        err=True,
        name: Optional[str] = None,
        file_path: Optional[str] = None,
        save: bool = False,
        filename=default_filename,
        wandb=False,
    ):  # NOSONAR
        """_summary_

        Args:
            args: _description_
            length (str, optional): _description_. Defaults to "longest".
            name (Optional[str], optional): _description_. Defaults to None.
            file_path (Optional[str], optional): File path to save object. Defaults to None.
            save (bool, optional): whether to save returned object. Defaults to False.
            log (bool, optional): whether to log to wandb. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            Score: average score to return
        """
        scores = Score.load_many(*args, filename=filename)
        if name is None:
            name = scores[0].name

        x, y, err = Score._get_scores(scores, err, length=length)

        avg_score = Score(x, y, err=err, name=name, file_path=file_path, use_wandb=wandb)
        if save:
            avg_score.save()
        if wandb:
            avg_score.wandb_log()

        return avg_score

    @staticmethod
    def _get_scores(scores, err, length: str = "longest"):
        if length == "median":
            len_x = int(median([len(s.x) for s in scores]))
            x = scores[0].x
            err = np.zeros(len_x) if err else None
            y = np.mean(np.stack([s.y[:len_x] for s in scores if len(s.y) >= len_x]), axis=0)
            if err is not None and len(scores) > 1:
                err = sem(
                    np.stack([s.y[:len_x] for s in scores if len(s.y) >= len_x]),
                    axis=0,
                )

        elif length in {"longest", "max"}:
            x, y, err = Score._get_longest_max_score(scores, err)
        else:
            raise NotImplementedError(
                ("Averaging only works with lengths 'median', 'longest' or 'max'." f"'{length}' was given")
            )
        return x, y, err

    @staticmethod
    def _get_longest_max_score(scores, err):
        i_max, n_max = max((i, n) for n, i in enumerate([len(s.x) for s in scores]))
        x = scores[n_max].x
        y = np.empty(i_max, dtype=float)
        err = np.zeros(i_max) if err else None

        i_0 = 0
        while i_0 < i_max:
            i_1 = min([len(s.x) for s in scores])
            y[i_0:i_1] = np.mean(np.stack([s.y[i_0:i_1] for s in scores]), axis=0)
            if err is not None and len(scores) > 1:
                err[i_0:i_1] = sem(np.stack([s.y[i_0:i_1] for s in scores]), axis=0)

            scores = [s for s in scores if len(s.x) > i_1]
            i_0 = i_1
        return x, y, err

    @staticmethod
    def from_folder(
        folder: str,
        name: Optional[str] = None,
        file_path: Optional[str] = None,
        save: bool = False,
    ):
        """

        Args:
            folder (str): _description_
            name (Optional[str], optional): _description_. Defaults to None.
            file_path (Optional[str], optional): _description_. Defaults to None.
            save (bool, optional): _description_. Defaults to False.
        """
        # TODO: docstring
        if name is None and file_path is None:
            name = " ".join((os.path.split(folder)[1], self.score_name))
        if file_path is None:
            file_path = os.path.join(folder, name.replace(" ", "_"))

        score = self.__class__(name=name, file=file_path)

        files = sorted(glob(os.path.join(folder, "scatter*")))
        for f in files:
            scatter = Scatter.load(f)
            score.compute(scatter)

        if save:
            score.save()

    def compute(self, a0=None, a1=None):
        # TODO: docstring and type hints
        # TODO: why have a0 and a1 if they are mutually exclusive?
        if isinstance(a0, Scatter):
            scatter = a0
            samples = scatter.samples
        elif isinstance(a1, Scatter):
            scatter = a1
            samples = a0
        else:
            scatter = self.scatter
            if self.scatter.samples is not None:
                samples = self.scatter.samples
            else:
                samples = a0
        if samples is None:
            samples = self.x[-1] + 1 if len(self.x) else 1
        self.append(samples, getattr(scatter, self.score_name)())

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["file_path"]
        d["scatter"] = None
        d["wandb"] = None
        for k, v in d.items():
            try:
                d[k] = np.asarray(v)
            except Exception as _:
                pass
        return d


class TopScore(Score):
    def compute(self, x, labels: ArrayLike, average_over: int = 1):
        """Compute top score.

        Args:
            x (_type_): _description_
            labels (ArrayLike): _description_
            average_over (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        # TODO: Docstring and type hints.
        score = np.mean(np.sort(labels)[-average_over:])
        self.x.append(x)
        self.y.append(score)
        return score


class RMSE(Score):
    default_filename = "RMSE.pickle"


class MAE(Score):
    default_filename = "MAE.pickle"


class F1(Score):
    default_filename = "F1.pickle"


class AUC(Score):
    default_filename = "AUC.pickle"


class Accuracy(Score):
    default_filename = "accuracy.pickle"


class BalancedAccuracy(Score):
    default_filename = "balanced_accuracy.pickle"


class Precision(Score):
    default_filename = "precision.pickle"


class Recall(Score):
    default_filename = "recall.pickle"


class Scatter:
    def __init__(
        self,
        labels=None,
        preds=None,
        errs=None,
        model=None,
        test=None,
        samples=None,
        name=None,
        axes=None,
        file=None,
        plot_args: Optional[Dict] = None,
    ):
        self.model = model
        self.samples = samples
        self.name = name
        self.file = file
        self.axes = axes
        self.plot_args = plot_args
        self.get_errs = None
        if test is not None:
            self.X = test.X
            self.labels = test.y
            self.get_errs = errs
            if model is not None and model.trained:
                self.compute(get_errs=bool(errs))
            else:
                self.preds = None
                self.errs = None
        else:
            self.X = None
            self.labels = labels
            self.preds = preds
            self.errs = errs
            self.get_errs = errs is not None

    def compute(self, get_errs=None, samples=None):
        if samples is not None:
            self.samples = samples
        if get_errs is None:
            get_errs = self.get_errs
        self.preds = self.model.predict(self.X)
        self.errs = self.model.std_dev(self.X) if get_errs else None

    def RMSE(self):
        return np.sqrt(np.mean(np.square(np.asarray(self.labels) - np.asarray(self.preds))))

    def MAE(self):
        return np.mean(np.abs(np.asarray(self.labels) - np.asarray(self.preds)))

    def F1(self, average="weighted", **kwargs):
        from sklearn.metrics import f1_score

        if self.preds.ndim > 1 and not np.issubdtype(np.asarray(self.preds).dtype, np.integer):
            preds = np.argmax(self.preds, axis=-1)
        else:
            preds = self.preds
        return f1_score(np.asarray(self.labels), preds, average=average, **kwargs)

    def AUC(self, **kwargs):
        from sklearn.metrics import roc_auc_score

        preds = self.preds
        if np.max(self.labels) == 1 and preds.ndim > 1:
            preds = preds[:, 1]
        return roc_auc_score(np.asarray(self.labels), np.asarray(preds), **kwargs)

    def Accuracy(self, **kwargs):
        from sklearn.metrics import accuracy_score

        if self.preds.ndim > 1 and not np.issubdtype(np.asarray(self.preds).dtype, np.integer):
            preds = np.argmax(self.preds, axis=-1)
        else:
            preds = self.preds
        return accuracy_score(np.asarray(self.labels), preds, **kwargs)

    def BalancedAccuracy(self, **kwargs):
        from sklearn.metrics import balanced_accuracy_score

        if self.preds.ndim > 1 and not np.issubdtype(np.asarray(self.preds).dtype, np.integer):
            preds = np.argmax(self.preds, axis=-1)
        else:
            preds = self.preds
        return balanced_accuracy_score(np.asarray(self.labels), preds, **kwargs)

    def Precision(self, **kwargs):
        from sklearn.metrics import precision_score

        if self.preds.ndim > 1 and not np.issubdtype(np.asarray(self.preds).dtype, np.integer):
            preds = np.argmax(self.preds, axis=-1)
        else:
            preds = self.preds
        return precision_score(np.asarray(self.labels), preds, **kwargs)

    def Recall(self, **kwargs):
        from sklearn.metrics import recall_score

        if self.preds.ndim > 1 and not np.issubdtype(np.asarray(self.preds).dtype, np.integer):
            preds = np.argmax(self.preds, axis=-1)
        else:
            preds = self.preds
        return recall_score(np.asarray(self.labels), preds, **kwargs)

    def plot(self, show_errors=True, axes=None, show=True, show_diagonal=True, block=True):
        if axes is None:
            axes = plt.gca()

        if self.axes is not None:
            axes.set_xlabel(self.axes[0])
            axes.set_ylabel(self.axes[1])

        title = []
        if self.name:
            title += [self.name]
        if self.samples:
            title += [f"{self.samples} samples"]
        title = "\n".join(title)
        if title:
            axes.set_title(title)

        show_errors = show_errors and (self.errs is not None)
        if show_errors:
            axes.errorbar(
                self.labels,
                self.preds,
                yerr=self.errs,
                fmt="o",
                ecolor="red",
                alpha=0.4,
            )

        y_max = np.amax(self.labels)
        y_min = np.amin(self.labels)

        if show_diagonal:
            plt.plot([y_min, y_max], [y_min, y_max], color="green", alpha=0.5)

        if show:
            plt.show(block=block)

        return axes

    def save(self, file=None):
        if file is None:
            file = self.file
        # for a, v in self.__dict__.items():
        #    print(a)
        #    print("    ", type(v))
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        # with open(file, "rb") as f:
        #     x = pickle.load(f)
        # x.file = file
        # return x
        raise NotImplementedError("We don't load pickle files natively. Ff you would like to, use pickle.load(f)")

    def __getstate__(self):
        d = self.__dict__.copy()
        d["model"] = None
        d["X"] = None
        del d["file"]
        for k, v in d.items():
            try:
                d[k] = np.asarray(v)
            except Exception as _:
                pass
        return d


def plot_scores(
    *XY,
    xlabel="Compounds / Number",
    ylabel=None,  # "Error / RMSE",
    show_err=True,
    confidence=0.95,
    grid=True,
    ticks=True,
    tight_layout=True,
    dpi=800,
    figsize=(5, 4),
    fontsize=12,
    xlim=None,
    ylim=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    show=True,
    block=True,
    save=False,
    file_path=None,
    title=None,
    name=None,
    legend=None,
    axes=None,
    **kwargs,
):
    """
    Returns the `matplotlib.axes.Axes` it is drawing on. You can use this to modify the
    plot after-the-fact.

    :param xlabel: Label for x-axis
    :param ylabel: Label for y-axis
    :param show_err: If True, shows error bands when available
    :param confidence: The confidence threshold of the error bands
    :param grid: If True, show a dashed gray grid
    :param ticks: If True, show ticks on the axes
    :param tight_layout: If True, calls matplotlib's `tight_layout`
    :param dpi: DPI for saved figures
    :param figsize: Size of the figure in matplotlib units
    :param fontsize: Font size for legend, axis labels
    :param xlim: Can be either an order pair `(xmin, xmax)`, or a dictionary
        `{'xmin':xmin, 'xmax':xmax}`. In fact, the dictionary may have any
        subset of the arguments to `matplotlib.axes.Axes.set_xlim`.
    :param ylim: Like xlim, but with `(ymin, ymax)`.
    :param xmin, xmax, ymin, ymax: Alternatively, you can pass plot limits directly
        as kwargs.
    :param show: Whether to call `matplotlib.pyplot.show`
    :param block: Whether the plot display should be blocking. Defaults to True.
    :param save: If `save == True`, or if `file` is given, saves the figure to a
        file. If `file` is specified, uses that filename. If `file` is not specified,
        builds a filename be sanitizing `title` or `name`.
    :param file_path:  See above
    :param title:
    :param name: `title` and `name` are synonyms, specifying the plot title
    :param legend: Whether or not to show a legend
    :param axes: You can specify matplotlib axes to plot into

    Additional keyword arguments are passed to the `plot` function.

    Note about titles/filenames: If you just want to give a name for the purpose of
    saving to a unique file, specify `file`. If you also want to show a title, there's
    no need to specify `file`---you can just specify `title` or `name`.
    """

    # load/process scores
    scores = []
    for xy in XY:
        if isinstance(xy, str):
            xy = Score.load(xy)
        if isinstance(xy, Iterable) and not isinstance(xy, Score):
            xy = Score(*xy)
        scores.append(xy)

    # set up figure and axes
    if axes is None:
        axes = plt.gca()
    elif figsize:
        plt.figure(figsize=figsize)
    if grid:
        axes.grid(True, linestyle="--", color="gray", alpha=0.35)
    if ticks:
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    if dpi:
        plt.rcParams["figure.dpi"] = dpi

    if ylabel is None and len({s.name for s in scores}) == 1:
        ylabel = scores[0].name
        if legend is None:
            legend = False

    if legend is None:
        legend = True

    # Plot the scores:
    for score in scores:
        kwargs_copy = kwargs.copy()
        kwargs_copy.update(score.plot_args)
        if score.name is not None and legend:
            kwargs_copy["label"] = score.name
        if show_err and hasattr(score, "err") and score.err is not None:
            s_y, s_err = np.asarray(score.y), conf_int(confidence, np.asarray(score.err))
            axes.fill_between(score.x, s_y - s_err, s_y + s_err, alpha=0.3)
        axes.plot(score.x, score.y, **kwargs_copy)

    # Format the axes:
    if xmin or xmax:
        xlim = (xmin, xmax)
    if isinstance(xlim, tuple):
        xlim = {"xmin": xlim[0], "xmax": xlim[1]}
    if isinstance(xlim, dict):
        axes.set_xlim(**xlim)
    if ymin or ymax:
        ylim = (ymin, ymax)
    if isinstance(ylim, tuple):
        ylim = {"ymin": ylim[0], "ymax": ylim[1]}
    if isinstance(ylim, dict):
        axes.set_ylim(**ylim)

    if xlabel:
        axes.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        axes.set_ylabel(ylabel, fontsize=fontsize)
    axes.legend()

    if name or title:
        title = title if title else name
        axes.set_title(title)

    if tight_layout:
        plt.tight_layout()

    # Save figure:
    if save or file_path:
        if file_path is None:
            assert title is not None, "You need to specify a filename or title if you're going to save."

            file_path = "".join([c for c in title.replace(" ", "_") if re.match(r"\w", c)]).lower() + ".pdf"
        plt.savefig(file_path, dpi=dpi)

    # Finally, show the plot:
    if show:
        plt.show(block=block)

    return axes
