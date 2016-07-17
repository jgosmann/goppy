.. image:: https://travis-ci.org/jgosmann/goppy.svg?branch=master
  :target: https://travis-ci.org/jgosmann/goppy
.. image:: https://coveralls.io/repos/jgosmann/goppy/badge.png?branch=master
  :target: https://coveralls.io/r/jgosmann/goppy?branch=master

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
* `Good test coverage. <https://coveralls.io/r/jgosmann/goppy>`_
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
