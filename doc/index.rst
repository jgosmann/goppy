.. GopPy documentation master file, created by
   sphinx-quickstart on Sun Mar 16 15:40:24 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: _static/goppy-sm.png
  :alt: GopPy logo

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
* :download:`MIT license <../LICENSE>`.

Contents:

.. toctree::
   :maxdepth: 2

   installation
   usage
   kernel
   api
   changelog



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
