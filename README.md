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
**TODO: Change documentation link**
[DOCUMENTATION HERE](google.com)

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

```
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

Contributing
============
Development of ALIEN happens on [GitHub](https://github.com/Sanofi-Public/Alien).

- If you encounter a bug, please fill out the issue form [here](https://github.com/Sanofi-Public/Alien/issues/new?assignees=lanele73%2Cmichael-bailey2&labels=bug&template=bug_report.yml&title=%5BBug%5D%3A+).
- If you have a feature request, please fill the [feature request form](https://github.com/Sanofi-Public/Alien/issues/new?assignees=lanele73%2Cmichael-bailey2&labels=enhancement&template=feature_request.yml&title=%5BFeature+Request%5D%3A+).