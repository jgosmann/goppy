Implementing Kernels
====================

Implementing an own kernel is easy. Just create a class which is derived from
:class:`.Kernel` and implement a :attr:`.Kernel.params` attribute and the
:meth:`.Kernel.full` method. Here is some example code to get you started::

    from goppy import Kernel

    class MyKernel(Kernel):
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

        @property
        def params(self):
            return np.array([self.param1, self.param2])

        @params.setter
        def params(self, values):
            self.param1 = values[0]
            self.param2 = values[1]

        def full(x, y, what=('y',)):
            # Evaluate your kernel
            pass

By implementing :attr:`.Kernel.params` as a property it is possible to access
the parameter as an array (which is needed for the evaluating log likelihood
derivatives of Gaussian processes), but also by more expressive names like
``k.param1``.

The :meth:`.Kernel.full` full method should by default evaluate the kernel
normally and return the result in a dictionary with the key ``'y'``. This is
sufficient for the basic functionality when used in conjunction with
:class:`.OnlineGP`. For more advanced usage involving predicting derivatives
:meth:`.Kernel.full` has also be able to return the derivatives of the kernel.
See the documentation of :meth:`.Kernel.full` for more information.

Sometimes only the diagonal of the Gram matrix is needed. This can usually be
more efficiently calculated. Thus, it might be a good idea to add code for this
special case by implementing :meth:`.Kernel.diag`. Otherwise the full Gram
matrix will be calculated, but only the diagonal returned.

Look at :download:`the source of the kernel module <../goppy/kernel.py>` to see
some complete example implementations of kernels.
