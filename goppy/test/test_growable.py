"""Unit test for growable module."""

import numpy as np
import pytest
from hamcrest import assert_that, is_, is_not, same_instance
from numpy.testing import assert_array_equal

from ..growable import GrowableArray


class TestGrowableArray:
    def test_creation_of_array(self):
        shape = (2, 3, 4)
        dtype = np.dtype("int")
        garray = GrowableArray(shape, dtype)

        assert_that(garray.shape, is_(shape))
        assert_that(garray.dtype, is_(dtype))

    def test_array_access(self):
        garray = GrowableArray((2, 2))
        garray[:, :] = 3
        assert_array_equal(garray, [[3, 3], [3, 3]])

    def test_new_from_template(self):
        garray = GrowableArray((3,))
        garray[:] = [1, 2, 3]
        new_one = garray[1:]

        assert_that(new_one, is_not(same_instance(garray)))
        assert_array_equal(new_one, [2, 3])

    def test_new_from_template_cannot_be_grown(self):
        garray = GrowableArray((3,))
        garray[:] = [1, 2, 3]
        new_one = garray[1:]
        with pytest.raises(AttributeError):
            new_one.grow_by((1,))

    def test_can_grow_array(self):
        garray = GrowableArray((2, 2))
        garray.grow_by((10, 20))
        garray[:, :] = 1
        assert_array_equal(garray, np.ones((12, 22)))

    def test_can_grow_array_by_0(self):
        garray = GrowableArray((2, 2))
        garray[:, :] = 1
        garray.grow_by((0, 0))
        assert_array_equal(garray, np.ones((2, 2)))

    def test_growing_array_does_not_mess_up_data(self):
        garray = GrowableArray((2, 2))
        data = [[1, 2], [3, 4]]
        garray[:, :] = data
        garray.grow_by((2, 2))
        assert_array_equal(garray[:2, :2], data)

    def test_can_be_used_in_ufunc(self):
        garray = GrowableArray((3,))
        garray[:] = [1, 2, 3]
        assert_that(np.sum(garray), is_(6))

    def test_can_be_used_in_array_calculation(self):
        a = np.array([3, 2, 1])
        garray = GrowableArray((3,))
        garray[:] = [1, 2, 3]
        expected = [4, 4, 4]
        assert_array_equal(garray + a, expected)

    def test_can_set_initial_buffer_size(self):
        garray = GrowableArray((3,), buffer_shape=(77,))
        assert_that(garray.base.shape, is_((77,)))

    def test_has_length(self):
        garray = GrowableArray((4, 2))
        assert_that(len(garray), is_(4))

    def test_cannot_delete_array_elements(self):
        garray = GrowableArray((3,))
        with pytest.raises(ValueError):
            del garray[1]

    def test_has_array_repr(self):
        garray = GrowableArray((3,))
        array = np.array([1, 2, 3], dtype=garray.dtype)
        garray[:] = array
        assert_that(repr(garray), is_(repr(array)))

    def test_has_array_str_conversion(self):
        garray = GrowableArray((3,))
        array = np.array([1, 2, 3], dtype=garray.dtype)
        garray[:] = array
        assert_that(str(garray), is_(str(array)))
