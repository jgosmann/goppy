.. image:: https://github.com/jgosmann/goppy/actions/workflows/ci.yml/badge.svg
  :target: https://github.com/jgosmann/goppy/actions/workflows/ci.yml
  :alt: CI and release pipeline
.. image:: https://codecov.io/gh/jgosmann/goppy/branch/main/graph/badge.svg?token=mkgZs4nds5 
  :target: https://codecov.io/gh/jgosmann/goppy

Overview
--------

GopPy (Gaussian Online Processes for Python) is a pure Python module providing
a Gaussian process implementation which allows to add new data efficiently
online. I wrote this module because all other Python implementations I knew did
not support efficient online updates.

The feature list:

* `scikit-learn <http://scikit-learn.org>`_ compatible interface.
* Efficient online updates.
* Prediction of first order derivatives.
* Estimation of the log likelihood and its derivative.
* Well documented.
* `Good test coverage. <https://app.codecov.io/gh/jgosmann/goppy>`_
* Supports Python 2.6, 2.7, 3.3, 3.4, and 3.5. Later versions are likely to work as
  well.
* MIT license.

Documentation
-------------

The documentation can be found at https://goppy.readthedocs.io/en/latest/.

Installation
------------

You can install GopPy with pip::

    pip install goppy

Or you `download the latest source distribution from PyPI
<https://pypi.python.org/pypi/GopPy/>`_, extract it and run::

    python setup.py install
