.. _active-learning:

What is active learning?
========================

Active learning is an approach to data acquisition which get us better 
models with less labeled data. Active learning doesn't just select the 
"most promising" candidates for the next batch of labeling. It also 
prioritizes candidates which would be most informative, i.e., which 
would improve our model best (which ultimately leads to finding 
promising candidates sooner).

Given a fixed model architecture (eg., the number of layers, the type of 
neurons, and the connectivity), machine learning fits the model parameters
to the data. With limited data, we will have limited confidence in the 
precise values of the parameters. In an ideal (Bayesian) case, model
fitting would not give precise values for the parameters; rather, it would
give a posterior distribution on parameter-space.

Many active learning selection strategies prioritize samples whose
labeling would most reduce uncertainty in the model.