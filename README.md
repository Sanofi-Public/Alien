Welcome to ALIEN
======================================

Python package, methods and data for the paper [Deep Batch Active Learning for Drug Discovery](https://www.biorxiv.org/content/10.1101/2023.07.26.550653v1).

Active Learning In data ExploratioN (ALIEN) is a library for active learning,
making it easy to use active
learning with popular machine learning frameworks, such as Pytorch, Keras,
Deepchem, and others. In most cases, you can plug your existing model and
dataset into ALIEN's wrapper classes and immediately get useful functions,
such as finding the (epistemic) uncertainty of a prediction, or selecting
batches of new points to be labelled next (eg., by measuring them in a lab).

ALiEN is in Beta testing and is not a finished product!

[Documentation](https://sanofi-public.github.io/Alien/).

Installation
============
```
$ git clone https://github.com/Sanofi-Public/Alien.git
$ cd Alien
$ pip install .
```

To install extras, use `pip install .[<extra>]` where `<extra>` may be one of
* `torch`,
* `tensorflow`,
* `deepchem`,
* `deepchem-torch`, or
* `deepchem-tensorflow`.

Quick start
===========

In the following example, we take certain objects as given:

- `keras_model` is a Keras regression model
- `X` and `y` are a set of initial labeled data, in a form consumable by `keras_model`
- `unlabeled_pool` is a set of samples to select from

Then the following code will wrap the model in ALIEN's `MCDropoutRegressor` class, fit the model to `X` and `y`, and then select a batch of 10 samples from `unlabeled_pool` for labelling using the *COVDROP* selection strategy (which is what we call the combination of
MC dropout with `CovarianceSelector`):

```python
from alien.models import MCDropoutRegressor
from alien.selecion import CovarianceSelector

alien_model = MCDropoutRegressor(keras_model)
alien_model.fit(X, y)

selector = CovarianceSelector(
    model = alien_model,
    samples = unlabeled_pool,
    batch_size = 10,
)

batch = selector.select()
```

Data
====
The datasets shared in the repository are under a [license](DATA_LICENSE.md):

*The data in this repository is provided solely for the purpose of validating the methods, and reproducing the results, described in the publication ["Deep Batch Active Learning for Drug Discovery"](https://elifesciences.org/reviewed-preprints/89679) by Bailey et al. All other rights are reserved.*

*We are not providing support relating to this data, but simply releasing it AS IS, without warranty of any kind, express or implied (including any warranty of noninfringement), and are not providing any guarantees of support relating to its use.*


Contributing
============
Development of ALIEN happens on [GitHub](https://github.com/Sanofi-Public/Alien).

- If you encounter a bug, please fill out the issue form [here](https://github.com/Sanofi-Public/Alien/issues/new?assignees=lanele73%2Cmichael-bailey2&labels=bug&template=bug_report.yml&title=%5BBug%5D%3A+).
- If you have a feature request, please fill the [feature request form](https://github.com/Sanofi-Public/Alien/issues/new?assignees=lanele73%2Cmichael-bailey2&labels=enhancement&template=feature_request.yml&title=%5BFeature+Request%5D%3A+).