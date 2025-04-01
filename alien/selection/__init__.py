"""
This module implements a number of batch selection strategies, and provides
a framework for implementing others. Each strategy is implemented as a
subclass of :class:`SampleSelector`. The docs for :class:`SampleSelector` 
give a good introduction to the interface. 

A selector object needs at least two things:

   - A :class:`Model <alien.models.Model>` (or precomputed covariances/entropies)
   - A pool of unlabeled samples to select from. This can be a 
     :class:`Dataset <alien.data.Dataset>`,
     or any sufficiently array-like object. Alternatively, this can be a
     :class:`SampleGenerator <alien.sample_generation.SampleGenerator>`.

Additionally, some selectors also need

   - The labeled samples the model was trained on.

Given these, and various hyperparameters and auxiliary data, you can call on
the selector to select a batch of candidates for labeling, chosen from the
unlabeled samples.

   * See :class:`SampleSelector` for the shared parameters and methods. 
      Each specific selector documents its own special parameters and behaviour.

Example
-------

.. code-block::

   from alien.selection import CovarianceSelector

   selector = CovarianceSelector(
      model = deep_model,
      samples = unlabeled_pool,
      batch_size = 10,
   )

   batch = selector.select()

ALIEN has the following selection strategies implemented.

* *Active learning strategies* - These are aimed at improving model
  performance as quickly as possible.

   - :class:`CovarianceSelector` (known as COVDROP in the literature)
   - :class:`BAITSelector`
   - :class:`KmeansSelector`
   - :class:`EntropySelector` (known as DEWDROP in not-yet-published work)

* *Optimization strategies* - These are aimed at finding the highest
  scorers as quickly as possible.

   - :class:`ExpectedImprovementSelector`
   - :class:`GreedySelector`
   - :class:`ThompsonSelector`

* *Baselines*

   - :class:`RandomSelector`
   - :class:`TimestampSelector`

You can write your own strategies! ALIEN's Selector architecture is designed
to make it as simple as possible for you to implement other selection
strategies. *Documentation coming soon!*

## How to choose? ##

For regression models, :class:`CovarianceSelector` is usually state-of-the-art.
In some cases, it's too slow or uses too much memory, in which case
:class:`EntropySelector` or :class:`BAITSelector` can be good alternatives.
:class:`EntropySelector` has the advantage that it can be used for non-differentiable
models, like gradient-boosting.

For classification models, :class:`EntropySelector` or :class:`BAITSelector`
tend to work well.

For Bayesian optimization, :class:`ThompsonSelector` or :class:`ExpectedImprovementSelector`
work well, with the caveat that most methods of determining uncertainty
are systematic underestimates, compared to the scale of the predictions.
This can be addressed by, eg., using a :class:`GaussianProcessRegressor`, or
using a `multiple` parameter in the selector, to artificially scale the
uncerainty estimates.

Bypassing this issue, :class:`CovarianceSelector` can be used for Bayesian
optimization by setting `prior='prediction'`, which biases the selection
towards high scorers in a tunable way.
"""

from .bait import BAITSelector
from .covariance import CovarianceSelector
from .entropy import EntropySelector
from .expected_improvement import ExpectedImprovementSelector
from .greedy import GreedySelector
from .kmeans import KmeansSelector
from .random import RandomSelector
from .selector import SampleSelector
from .timestamp import TimestampSelector
from .thompson import ThompsonSelector
