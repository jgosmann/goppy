"""Provides an array class which can be enlarged."""

import numpy as np


def GrowableArray(shape, dtype=None, order='C'):
    return _GrowableArray.create(shape, dtype, order)


class _GrowableArray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.ndarray.__new__(np.ndarray, *args, **kwargs)

    @staticmethod
    def create(shape, dtype, order):
        data = np.ndarray(shape=shape, dtype=dtype, order=order)
        view = np.ndarray.__new__(
            _GrowableArray, shape=shape, dtype=dtype, buffer=data,
            strides=data.strides, order=order)
        setattr(view, '_data', data)
        return view
