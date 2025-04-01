Installation
============

Method 1—Only on OneAI
----------------------

The latest Release version of ALIEN is deployed in Sanofi's JFrog artifactory and can be 
installed as a PyPI package from OneAI workbenches.

.. code-block:: bash

    pip install alien --no-cache-dir

This installs the base version of the package, without deep-learning dependencies. Optional 
dependencies can be installed with the extras notation:

.. code-block:: bash

    pip install alien[torch] --no-cache-dir

In the above command, `torch` may be any of
    `torch`, `tensorflow`, `deepchem`, `deepchem-torch`, or `deepchem-tensorflow`.
We provide no special install options for LightGBM  or CatBoost—install them in the usual way.

Method 2—Works anywhere you can access Sanofi-GitHub
----------------------------------------------------

This method gets you a clone of the repository. If you can't use Method 1, if you want the latest
pre-release version, or if you want to play with the source code, use this method.

1.  Once you have access to the repository, find a nice quiet folder on your machine, and clone the repository:

    .. code-block:: bash

        git clone https://github.com/Sanofi-GitHub/UDS-active_learning_sdk

    You may have to supply your GitHub username and access token, either by adding them into 
    the above command, or through git configuration options.

2.  From inside the UDS-active_learning_sdk folder, run `make install`

3.  Activate the new Conda environment: `conda activate alenv`

4.  If you want to install optional dependencies, then from within this same directory, run 
    `make \<dependency\>`, where `\<dependency\>` is one of:
        `torch`, `tensorflow`, `deepchem`, `deepchem-torch`, or `deepchem-tensorflow`
