import os
import pickle

import numpy as np

from ..data import DictDataset
from ..utils import shift_seed
from .oracle import SetOracle


def run_experiments(
    X,
    y,
    model,
    runs_dir,
    overwrite_old_runs=True,
    n_initial=None,
    batch_size=20,
    num_samples=float("inf"),
    selector=None,  # 'covariance', 'random', 'expected improvement'/'ei', 'greedy'
    selector_args=None,
    fit_args=None,
    n_runs=10,
    ids=None,
    save_ids=True,
    random_seed=1,
    split_seed=421,
    test_size=0.2,
    timestamps=None,
    stop_samples=None,
    stop_rmse=None,
    stop_frac=None,  # suggest .85,
    # The following arguments are a bit odd, and best avoided
    peek_score=0,  # 0 if no peeking
    test_samples_x=None,
    test_samples_y=None,
):
    """
    :param runs_dir: directory to store the runs and results of this training (each run in
        separate subdirectories)
    :param n_initial: number of samples to randomly select for initial training data
    :param batch_size: number of samples selected for batch
    :param num_samples: number of samples (drawn from the sample pool X) to select from.
        Default is inf, which takes all of the samples available in X.
    :param selector: the selector to use for batch selection, either given
        by one of the strings 'covariance', 'random', 'expected improvement'/'ei', 'greedy',
        or passed as an actual SampleSelector instance. Defaults to 'covariance'.
    :param selector_args: a dictionary passed as kwargs to the selector constructor.
        The following constructor arguments are already automatically included, and don't
        need to be included in this dictionary:
            model, labelled_samples, samples, num_samples, batch_size
    :param fit_args: a dictionary passed as kwargs each time model.fit(...) is called.
        Typically, this is model- or framework-specific; so, eg., different arguments would
        be appropriate for pytorch models, DeepChem models, etc.
    :param n_runs: the number of overall runs (each starting from
        a random initial selection) to do (for averaging)

    :param random_seed: random seed for most RNG generation
    :param split_seed: random seed for shuffling and splitting of data
    :param test_size: the size of the test/validation set to take from X,y.
        If test_size >= 1, then takes that many samples. if 0 < test_size < 1, takes
        that fraction of the dataset size.

    :param stop_samples: if this is not None, stops an experiment run when
        this many samples are labelled. Defaults to None
    :param stop_rmse: if this is not None, stops an experiment run when this
        RMSE has been reached. Defaults to None
    :param stop_frac: if this is not None, stops an experiment run when the
        RMSE has moved this fraction of the way from the RMSE after the first
        round to the RMSE trained on the whole dataset. We suggest something like
        .85, if you want to use this feature. Defaults to None

    """
    selector, selector_args, fit_args = _args_init(selector, selector_args, fit_args)

    data = _data_init(X, y, ids=ids, timestamps=timestamps)
    data_train, test_samples = _get_train_test(
        data, split_seed, test_size, test_samples_x=test_samples_x, test_samples_y=test_samples_y
    )

    if stop_samples is not None and 0 < stop_samples < 1:
        stop_samples = len(data_train) * stop_samples

    if stop_frac is not None:
        print("Fitting model to whole dataset, to determine stopping limit... ", flush=True)
        model.fit(data_train.X, data_train.y, **fit_args)
        final_rmse = np.sqrt(
            np.mean(
                np.array(model.predict(test_samples.X.data, return_std_dev=False))
                - np.array(test_samples.y.data)
            )
        )
        print("Done.")
    else:
        final_rmse = None

    if n_initial is None:
        n_initial = batch_size

    n_0 = _overwrite(runs_dir, overwrite_old_runs)

    selector_args = selector_args.copy()
    selector_args.update({"model": model, "batch_size": batch_size, "num_samples": num_samples})
    selector = _get_selector(selector, selector_args)
    first_seed = random_seed

    for n in range(n_0, n_0 + n_runs):
        run_dir = os.path.join(runs_dir, f"run_{n:02d}")
        os.makedirs(run_dir, exist_ok=True)

        random_seed = shift_seed(first_seed, n * 5)

        _run_retro_iter(
            n,
            run_dir,
            model,
            data_train,
            test_samples,
            batch_size=batch_size,
            n_initial=n_initial,
            selector=selector,
            fit_args=fit_args,
            save_ids=save_ids,
            random_seed=random_seed,
            stop_samples=stop_samples,
            stop_rmse=stop_rmse,
            stop_frac=stop_frac,
            final_rmse=final_rmse,
            peek_score=peek_score,
        )


def _run_retro_iter( # NOSONAR
    n,
    run_dir,
    model,
    unlabelled_samples,
    test_samples,
    batch_size,
    n_initial=None,
    selector=None,
    fit_args={}, # NOSONAR
    save_ids=True,
    random_seed=None,
    stop_samples=None,
    stop_rmse=None,
    stop_frac=None,  # suggest .85,
    final_rmse=None,
    peek_score=0,  # 0 if no peeking
):
    """Helper function to run an iteration of an experiment."""
    rng = np.random.default_rng(random_seed)
    model.initialize(init_seed=random_seed, sample_input=test_samples.X[:1])

    sample_pool = SetOracle(data=unlabelled_samples, shuffle=False, random_seed=shift_seed(random_seed, 1))
    initial_indices = rng.choice(len(sample_pool), size=n_initial, replace=False)
    labelled_samples = unlabelled_samples[initial_indices]
    sample_pool.remove_samples(labelled_samples)

    batch_ids = [list(initial_indices)]
    if save_ids:
        _save_ids(batch_ids, run_dir)

    model.data = labelled_samples
    selector.labelled_samples = labelled_samples
    selector.samples = sample_pool

    from .metrics import RMSE, Scatter, TopScore

    scatter = Scatter(model=model, test=test_samples, name="test scatter", errs=None)
    top_scores = TopScore(name="top scores", file_path=f"{run_dir}/top_scores.pickle")
    rmse = RMSE(name="RMSE", file_path=f"{run_dir}/RMSE.pickle", scatter=scatter)

    round_ = 0
    first_rmse = None
    while True:
        print(f"\n\n### Round {round_} (Run {n}) ###")
        print(f"  Model:    {model}")
        print(f"  Selector: {selector}")
        print(f"  {len(labelled_samples)} labelled samples")
        print(f"  {len(sample_pool)} samples remaining")

        print("\nTraining model...")
        model.fit(**fit_args)

        print("\nComputing metrics...")
        labels = labelled_samples.y
        if peek_score:
            samples = sample_pool.generate_samples()
            preds = model.predict(samples)
            best = np.argsort(preds)[:-peek_score]
            labels = np.concatenate((labels, samples.y))

        top_scores.compute(x=len(labelled_samples), labels=labels, average_over=5)
        top_scores.save()
        scatter.compute(samples=len(labelled_samples))
        scatter.save(f"{run_dir}/scatter_{round_:02d}.pickle")
        rmse.compute()
        rmse.save()
        if round_ == 0:
            first_rmse = rmse.y[0]

        if _check_retro_break(
            round_,
            first_rmse,
            rmse,
            final_rmse,
            sample_pool,
            labelled_samples,
            stop_samples,
            stop_frac,
            stop_rmse,
        ):
            break

        print("\nSelecting batch...")
        if len(sample_pool) <= batch_size:
            batch = sample_pool.generate_samples()
        else:
            batch = selector.select()

        print("\nLearning batch...")
        labelled_samples.extend(batch)
        sample_pool.remove_samples(batch)

        batch_ids.append(list(batch.ids.data))
        if save_ids:
            _save_ids(batch_ids, run_dir)

        round_ += 1


def _save_ids(ids, dir='', filename=None):
    with open(f"{dir}/batch_ids.json" if filename is None else filename, "w") as f:
        f.write(
            "[\n    [" + "],\n    [".join(
                ", ".join(str(i) for i in b)
                for b in ids
            ) + "]\n]\n"
        )


def _get_selector(selector, selector_args):
    """Helper function to spawn a selector."""
    if selector in {"expected improvement", "ei"}:
        from ..selection.expected_improvement import ExpectedImprovementSelector

        selector = ExpectedImprovementSelector(**selector_args)
    elif selector == "covariance":
        from ..selection.covariance import CovarianceSelector

        selector = CovarianceSelector(**selector_args)
    elif selector in {"bait", "BAIT"}:
        from ..selection.bait import BAITSelector

        selector = BAITSelector(**selector_args)
    elif selector == "random":
        from ..selection.random import RandomSelector

        selector = RandomSelector(**selector_args)
    elif selector == "greedy":
        from ..selection.greedy import GreedySelector

        selector = GreedySelector(**selector_args)
    elif selector == "kmeans":
        from ..selection.kmeans import KmeansSelector

        selector = KmeansSelector(**selector_args)
    elif selector == "timestamp":
        from ..selection import TimestampSelector

        selector = TimestampSelector(**selector_args)
    else:
        raise ValueError(f"'{selector}' is an invalid choice of selector.")
    return selector


def _get_train_test(data, split_seed, test_size, test_samples_x=None, test_samples_y=None):
    """Helper function to get train-test split of the data."""
    if test_samples_x is None:
        shuffle = np.arange(len(data.X))
        np.random.default_rng(split_seed).shuffle(shuffle)
        test_size = test_size if test_size >= 1 else int(len(data.X) * test_size)
        test_indices, train_indices = shuffle[:test_size], shuffle[test_size:]

        data_train = data[train_indices]
        test_samples = DictDataset(
            {
                "X": data.X.data[test_indices],
                "y": data.y.data[test_indices],
                "ids": data.ids.data[test_indices],
            }
        )
    else:
        data_train = data
        test_samples = DictDataset({"X": test_samples_x, "y": test_samples_y})
    return data_train, test_samples


def _data_init(X, y, ids=None, timestamps=None):
    """Helper function to initialize the data object from samples."""
    if ids is None:
        ids = np.arange(len(X))
    data_dict = {
        "X": X,
        "y": y,
        "ids": ids,
    }
    if timestamps is not None:
        data_dict["t"] = timestamps
    data = DictDataset(data_dict)
    return data


def _args_init(selector, selector_args, fit_args):
    """Helper function to initialize optional args in run_experiments."""
    if selector is None:
        selector = "covariance"
    if selector_args is None:
        selector_args = {}
    if fit_args is None:
        fit_args = {}
    return selector, selector_args, fit_args


def _overwrite(runs_dir: str, overwrite_old_runs: bool) -> int:
    """Helper function to overwrite old runs in run_experiments."""
    os.makedirs(runs_dir, exist_ok=True)
    n_0 = 0
    if overwrite_old_runs:
        import shutil

        shutil.rmtree(runs_dir)
        os.makedirs(runs_dir)
    else:
        while f"run_{n_0:02d}" in os.listdir(runs_dir):
            n_0 += 1
    return n_0


def _check_retro_break(
    round_,
    first_rmse,
    rmse,
    final_rmse,
    sample_pool,
    labelled_samples,
    stop_samples,
    stop_frac,
    stop_rmse,
):
    """Helper function to check whether to break experiment loop."""
    if len(sample_pool) == 0:
        print("Ending run because we've selected the whole dataset")
        return True
    if stop_samples is not None and len(labelled_samples) >= stop_samples:
        print(f"Ending run because we've reached limit of {stop_samples} samples.")
        return True
    if stop_frac is not None and (first_rmse - rmse.y[-1]) >= stop_frac * (
        first_rmse - final_rmse
    ):
        print("Ending run because RMSE is close enough to final RMSE.")
        return True
    if stop_rmse is not None and rmse.y[-1] <= stop_rmse:
        print("Ending run because RMSE is better than threshold.")
        return True
    return False


# I was going to create a command-line interface, with argument parser, but I
# am busy and am deprioritizing it

# def main():
#    from argparse import ArgumentParser
#    parser = ArgumentParser()
##    parser.add_argument('--dir', '-d',
#    initial_samples = 20,
#    batch_size = 20,
#    num_samples = float('inf'),
#    selector = 'greedy', # 'random', 'expected improvement'/'ei', 'covariance', 'greedy'
#    n_runs = 10,
#
#    random_seed = 1,
#    split_seed = 421,
#    test_size = .2,
