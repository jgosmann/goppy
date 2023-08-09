Usage
=====

This section gives a small tutorial of the core functions of GopPy. A basic
familiarity with Gaussian processes is assumed. Otherwise you might want to take
a look at [1]_ first (there is also a free online version).

Creating and Fitting Gaussian Process
-------------------------------------

Let us first create some toy data from a cosine:

.. literalinclude:: pyplots/usage/common.py
   :lines: 1, 4-7

The orientation of data matrices used by GopPy is samples (rows) times
dimensions (columns). Here we use one-dimensional input and output spaces.
Hence, both arrays have 15 rows and 1 column. This is the same way
`scikit-learn <http://scikit-learn.org>`_ handles data.

Then we create a Gaussian process using a squared exponential kernel with
a length scale of 0.5. We also use a noise variance of 0.1. The process is
fitted to the data using the :meth:`.OnlineGP.fit` method:

.. literalinclude:: pyplots/usage/common.py
   :lines: 10-13

After fitting we can use the Gaussian process to make predictions about the
mean function and obtain the associated uncertainty:

.. literalinclude:: pyplots/usage/fit.py
   :lines: 1, 3-

.. plot:: pyplots/usage/fit.py

Adding New Data to a Gaussian Process
-------------------------------------

When further data is obtained, these can be added easily to the Gaussian process
by using the :meth:`.OnlineGP.add` method:

.. literalinclude:: pyplots/usage/add.py
   :lines: 5-

.. plot:: pyplots/usage/add.py

If you called the :meth:`.OnlineGP.fit` method multiple times, the process
would be retrained discarding previous data instead of adding new data. You may
also use :meth:`.OnlineGP.add` for the initial training without ever calling
:meth:`.OnlineGP.fit`.

Tips
----

* If you know how many samples will be added overall to the Gaussian process, it
  can be more efficient to pass this number as ``expected_size`` to the
  :class:`.OnlineGP` constructor on creation.
* You can also predict first order derivatives. Take a look at the
  documentation of :meth:`.OnlineGP.fit`.
* You can also calculate the log likelihood and the derivative of the log
  likelihood. Take a look at the documentation of
  :meth:`.OnlineGP.calc_log_likelihood`.

References
----------

.. [1] Rasmussen, C E, and C K I Williams. Gaussian Processes for
   Machine Learning. MIT Press, 2006. http://www.gaussianprocess.org/gpml/
