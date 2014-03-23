"""Provides array types which can be enlarged after creation."""

import numpy as np


class GrowableArray(object):
    """An array which can be enlarged after creation.

    Though this is not a subclass of :class:`numpy.ndarray`, it implements
    the same interface.

    Parameters
    ----------
    shape : int or tuple of int
        Initial shape of the created empty array.
    dtype : data-type, optional
        Desired output data-type.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in C (row-major) or Fortran
        (column-major) order in memory.
    buffer_shape : int or tuple of int, optional
        Initial shape of the buffer to hold the actual data. As long as the
        array shape stays below the buffer shape no new memory has to
        reallocated.

    Examples
    --------

    >>> from goppy.growable import GrowableArray
    >>> a = GrowableArray((1, 1))
    >>> a[:, :] = 1
    >>> print a
    [[ 1.]]
    >>> a.grow_by((1, 2))
    >>> a[:, :] = 2
    >>> print a
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    """

    ENLARGE_FACTOR = 2

    def __init__(self, shape, dtype=float, order='C', buffer_shape=None):
        if buffer_shape is None:
            buffer_shape = self.ENLARGE_FACTOR * np.asarray(shape, dtype=int)
        self._data = np.empty(buffer_shape, dtype, order)
        self._view = self.__get_view_for_shape(self._data, shape)

    @staticmethod
    def __get_view_for_shape(data, shape):
        return data.__getitem__([slice(end) for end in shape])

    def __getattr__(self, name):
        return getattr(self._view, name)

    def __getitem__(self, key):
        return self._view.__getitem__(key)

    def __setitem__(self, key, value):
        self._view.__setitem__(key, value)

    def __delitem__(self, key):
        self._view.__delitem__(key)

    def __len__(self):
        return self._view.__len__()

    # TODO unittest
    def __repr__(self):
        return repr(self._view)

    # TODO unittest
    def __str__(self):
        return str(self._view)

    def grow_by(self, amount):
        """Grow the array.

        Parameters
        ----------
        amount : int or tuple of int
            Amount by which each dimension will be enlarged.
        """
        amount = np.asarray(amount, dtype=int)
        # TODO unit test equal 0 works
        assert np.all(amount >= 0)
        new_shape = np.asarray(self.shape, dtype=int) + amount
        # TODO unittest data stays at the right place
        if np.any(self._data.shape < new_shape):
            self._data = np.empty(self.ENLARGE_FACTOR * new_shape)
            self._data[[slice(None, s) for s in self._view.shape]] = self._view
        self._view = self.__get_view_for_shape(self._data, new_shape)
