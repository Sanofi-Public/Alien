.. _getting-started:

Getting Started
===============

First, use one of the two methods to install the package, and setup the environment:

    * :doc:`installation`

If you're using OneAI, it could be as simple as

.. code-block:: bash

    $ pip install alien

Then, why not look into the Jupyter notebook `notebooks/keras_classifier.ipynb`? 
Or, you can import the package and start using it yourself. You need to wrap a model
in ALIEN's :class:`Model` class (or use one of ALIEN's specialized model classes), 
and then connect it to a :class:`SampleSelector` object.

Here's a simple example using a deep learning regression model:

.. code-block:: python

    from alien.models import Model

    # deep_model is your deep learning model (Pytorch, Keras, etc.)
    wrapped_model = alien.models.Model(model=deep_model, mode='regression', uncertainty='dropout')

    # Train the model (you can do this directly with your model, or with the wrapper)
    model.fit(X=X, y=y, **other_args)

    # Make predictions
    predictions = model.predict(dataset)

    # Get uncertainties
    std_dev = model.uncertainty(dataset)

    # Select new points to label
    from alien.selection import CovarianceSelector

    selector = CovarianceSelector(model = deep_model)

    batch = selector.select(samples = unlabeled_pool, batch_size = 10)




