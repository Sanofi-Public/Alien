import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def KL_divergence(preds, std_devs, y, noise=None, normalize=True):
    """
    Computes the KL-divergence from the predicted distribution
    (using preds and std_devs, assuming normal distributions)
    to the ground-truth distribution (all of the probability
    mass on the true y values). In other words, this tells you
    how much information would be gained by learning the true
    values. Averaged over sample points.

    Lower is generally better. Not only does this penalize
    uncertainties which are poorly calibrated (i.e., it
    penalizes uncertainties where the actual error distribution
    has a different standard deviation), but also it penalizes
    uncertainties which are not as specific as they could be,
    i.e., which fail to discriminate between certain and
    uncertain predictions.

    :param preds: predicted values
    :param std_devs: predicted uncertainties, given as standard
        deviations
    :param y: true values
    :param normalize: if True (the default), normalizes with respect
        to the RMSE
    """
    if noise is not None:
        std_devs = std_devs + noise

    nonzeros = std_devs != 0
    preds, std_devs, y = preds[nonzeros], std_devs[nonzeros], y[nonzeros]
    # print(f"len = {len(preds)}")

    if normalize:
        scale = np.sqrt(np.mean(np.square(preds - y)))
        preds = preds / scale
        std_devs = std_devs / scale
        y = y / scale

    return -np.mean(np.log2(stats.norm.pdf(y, loc=preds, scale=std_devs)))


def best_multiple(preds, std_devs, y, noise=None, max_precision=5):
    """
    Does a simple binary search to find which multiple of std_devs gives the
    lowest KL-divergence score. To converge, assumes there is only one local
    minimum. (I expect this to be true, but I will have to check.)

    Arguments preds, std_devs, y and noise are as in KL_divergence, except
    this will compute the KL-divergence for multiples of std_devs.

    :param max_precision: the number of interval splits the score must be
        adjacent to before returning the value. Defaults to 5.
    """

    def KL_score(m, preds, std_devs, y, noise=None):
        return KL_divergence(preds, m * std_devs, y, noise=None)

    return binary_optimize(
        KL_score,
        preds,
        std_devs,
        y,
        noise=noise,
        mode="min",
        max_precision=max_precision,
    )


def binary_optimize(
    fn, *args, mode="max", start=1.0, max_precision=5, max_iterations=50, **kwargs
):
    """
    Does a simple binary search to find which scalar value gives the best
    (max/min) value of fn. Optimization converges to a local max/min.

    Each search iteration starts with the previously-explored value with the
    best score, and looks on either side of it. If the best score is at the
    beginning of the current list of value, it looks at half this value
    on the low side; if at the end of the list, looks at twice this value
    on the high side. If the best value is somewhere in the middle, it divides
    the interval on either side in half.

    :param max_precision: the number of interval splits the score must be
        adjacent to before returning the value. Defaults to 5.
    """
    opt = {"max": np.argmax, "min": np.argmin}[mode]

    vals = [start]
    scores = [fn(start, *args, **kwargs)]
    precision = [0]

    for _ in range(max_iterations):
        i = opt(scores)
        if precision[i] >= max_precision:
            break

        # look below
        v = search_lower(i, vals, precision)
        if v is None:
            break
        vals.insert(i, v)
        scores.insert(i, fn(v, *args, **kwargs))

        i += 1  # min value has shifted because of insert

        # look above
        v = search_upper(i, vals, precision)
        if v is None:
            break
        vals.insert(i + 1, v)
        scores.insert(i + 1, fn(v, *args, **kwargs))

    for prec, val, score in zip(precision, vals, scores):
        print(f"{prec:02d} - {val:03.2f} : {score:.3f}")

    return vals[opt(scores)]


def search_lower(i, vals, precision, max_precision=5):
    v = None
    if i == 0:
        v = vals[i] / 2
        precision.insert(i, 0)
    else:
        if precision[i - 1] >= max_precision:
            return v
        v = (vals[i - 1] + vals[i]) / 2
        precision.insert(i, max(precision[i - 1 : i + 1]) + 1)
    return v


def search_upper(i, vals, precision, max_precision=5):
    v = None
    if i == len(vals) - 1:
        v = vals[i] * 2
        precision.insert(i + 1, 0)
    else:
        if precision[i + 1] >= max_precision:
            return v
        v = (vals[i] + vals[i + 1]) / 2
        precision.insert(i + 1, max(precision[i : i + 2]) + 1)
    return v


def plot_errors(preds, std_devs, y, noise=None, show=True, axes=None, **kwargs):
    if axes is None:
        axes = plt.gca()

    errs = y - preds
    sort = np.argsort(std_devs)
    std_devs, errs = std_devs[sort], errs[sort]

    X = np.arange(len(errs))
    axes.plot(X, np.zeros(len(errs)), color="green")
    axes.fill_between(X, -std_devs, std_devs, alpha=0.3)

    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.35
    axes.scatter(X, errs, **kwargs)

    if show:
        plt.show()
