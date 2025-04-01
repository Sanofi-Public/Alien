Choosing Hyperparameters
===================

- The most important choice you make when running ALIEN is which selection strategy
to use (see :mod:`alien.selection`).

- You may or may not be free to choose the size of your selection batches. Does your
oracle provide labels one-at-a-time, or in batches of some specific size? Given the
choice, if you will label 100 candidates, or all at once, your model will need fewer 
labels for a given performance if is is labelling in smaller batches and retraining
after every round (so, 1-at-a-time is best).

However, batch-optimized strategies like COVDROP (:class:`CovarianceSelector`), 
DEWDROP (:class:`EntropySelector`) and BAIT (:class:`BAITSelector`) are not as
sensitive to batch sizes, so it's not as important. And in real life applications 
(eg. in a lab) batch sizes are often given and fixed.

- If you have an existing labeled dataset in a similar problem domain, you can try
running a :ref:`retrospective experiment <retrospective>` with the different
options. However, we do have some hints:

MC dropout, with the :class:`CovarianceSelector`, does best for regression
problems in our extensive benchmarks, so that's a good place to start.

