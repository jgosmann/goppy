.. image:: https://travis-ci.org/jgosmann/goppy.svg?branch=master
  :target: https://travis-ci.org/jgosmann/goppy
.. image:: https://coveralls.io/repos/jgosmann/goppy/badge.png?branch=master
  :target: https://coveralls.io/r/jgosmann/goppy?branch=master

Overview
--------

GopPy (Gaussian Online Processes for Python) is a pure Python module providing
a Gaussian process implementation which allows to efficiently add new data
online. I wrote this module because all other Python implementations I know did
not support efficient online updates.

The features include:

* `scikit-learn <http://scikit-learn.org>`_ compatible interface.
* Efficient online updates.
* Prediction of first order derivatives.
* Estimation of the log likelihood and its derivative.
* Well documented.
* `Good test coverage. <https://coveralls.io/r/jgosmann/goppy>`_
* MIT license.

Documentation
-------------

The documentation can be found at http://goppy.readthedocs.org/en/latest/.

Installation
------------

You can install GopPy with pip::

    pip install goppy

Or you `download the latest source distribution from PyPI
<https://pypi.python.org/pypi/GopPy/>`_, extract it and run::

    python setup.py install
