**ALIEN**
======================================

**A**\ctive **L**\earning **I**\n data **E**\xploratio\ **N** (ALIEN) is a library for active learning,
making it easy to use active
learning with popular machine learning frameworks, such as Pytorch, Keras,
Deepchem, LightGBM, CatBoost and others. In most cases, you can plug your 
existing model and dataset into ALIEN's wrapper classes and immediately get 
useful functions, such as finding the (epistemic) uncertainty of a prediction, 
or selecting batches of new points to be labelled next (eg., by measuring them 
in a lab).

   * :doc:`active_learning`

We have some handy installation instructions: :doc:`installation`

The two main submodules for the end-user of ALIEN are 

* :mod:`alien.models`, 
which contains wrapper classes which give your models the tools to compute
uncertainties and get embeddings, and 

* :mod:`alien.selection`, 
containing :class:`SampleSelector` subclasses which implement a number of different batch
selection strategies. (ALIEN is designed to make it maximally easy for
you to implement new selection strategies.)

The other submodules are :mod:`alien.data`, containing wrapper classes for various
data formats (you may or may not have to use these), :mod:`alien.sample_generation`,
containing classes to help with generating sample pools for the selectors, and
:mod:`alien.benchmarks`, containing functions for running "retrospective experiments"
and benchmarking selector performance.

.. toctree::
   active_learning.rst
   installation.rst
   alien.models
   alien.selection
   alien.benchmarks
   alien.data
   alien.sample_generation
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
