"""Provides an array class which can be enlarged."""

import numpy as np


class GrowableArray(object):
    MARGIN_FACTOR = 2

    def __init__(self, shape, dtype=float, order='C', buffer_shape=None):
        if buffer_shape is None:
            buffer_shape = self.MARGIN_FACTOR * np.asarray(shape, dtype=int)
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

    def grow_by(self, amount):
        amount = np.asarray(amount, dtype=int)
        assert np.all(amount > 0)
        new_shape = np.asarray(self.shape, dtype=int) + amount
        self._view = None
        if np.any(self._data.shape < new_shape):
            self._data.resize(self.MARGIN_FACTOR * new_shape)
        self._view = self.__get_view_for_shape(self._data, new_shape)
