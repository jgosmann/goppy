.. GopPy documentation master file, created by
   sphinx-quickstart on Sun Mar 16 15:40:24 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GopPy's documentation!
=================================

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
* Supports Python 2.6, 2.7, 3.2, and 3.3. Later versions are likely to work as
  well.
* :download:`MIT license <../LICENSE>`.

Contents:

.. toctree::
   :maxdepth: 2

   installation
   usage
   kernel
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

