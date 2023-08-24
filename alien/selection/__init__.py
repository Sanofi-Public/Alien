"""
This module implements a number of batch selection strategies, and provides
a framework for implementing others. Each strategy is implemented as a
subclass of :class:`SampleSelector`. The docs for :class:`SampleSelector` 
give a good introduction to the interface. 

A selector object needs at least two things:

   - A :class:`Model <alien.models.Model>`
   - A pool of unlabeled samples to select from. This can be a 
     :class:`Dataset <alien.data.Dataset>`,
     or any sufficiently array-like object. Alternatively, this can be a
     :class:`SampleGenerator <alien.sample_generation.SampleGenerator>`.

Additionally, some selectors also need

   - The labeled samples the model was trained on.

Given these, and various hyperparameters and auxiliary data, you can call on
the selector to select a batch of candidates for labeling, chosen from the
unlabeled samples.

Example
-------

.. code-block::

   from alien.selecion import CovarianceSelector

   selector = CovarianceSelector(
      model = deep_model,
      samples = unlabeled_pool,
      batch_size = 10,
   )

   batch = selector.select()

ALIEN has the following selection strategies implemented.

* *Active learning strategies* - These are aimed at improving model
  performance as quickly as possible.

   - :class:`CovarianceSelector`
   - :class:`BAITSelector`
   - :class:`KmeansSelector`

* *Optimization strategies* - These are aimed at finding the highest
  scorers as quickly as possible.

   - :class:`ThompsonSelector`

* *Baselines*

   - :class:`RandomSelector`
   - :class:`TimestampSelector`

You can write your own strategies! ALIEN's Selector architecture is designed
to make it as simple as possible for you to implement other selection
strategies. *Documentation coming soon!*
"""

from .bait import BAITSelector
from .covariance import CovarianceSelector
from .kmeans import KmeansSelector
from .random import RandomSelector
from .selector import SampleSelector, UncertaintySelector
from .thompson import ThompsonSelector
from .timestamp import TimestampSelector
