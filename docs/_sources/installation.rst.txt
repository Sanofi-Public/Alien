Installation
============

.. code-block:: bash

    pip install .

Installs the base version of the package, without deep-learning dependencies. Optional dependencies
can be install ed with the extras notation:

.. code-block:: bash
    
    pip install .[<extra>]

In the above command, `<extra>` may be any of 
    `torch`, `tensorflow`, `deepchem`, `deepchem-torch`, or `deepchem-tensorflow`.
