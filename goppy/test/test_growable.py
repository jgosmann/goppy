"""Unit test for growable module."""

from hamcrest import assert_that, equal_to, instance_of, is_, is_not, \
    same_instance
from nose.tools import raises
import numpy as np
from numpy.testing import assert_array_equal

from ..growable import GrowableArray


class TestGrowableArray(object):
    def test_creation_of_array(self):
        shape = (2, 3, 4)
        dtype = np.dtype('int')
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

    @raises(AttributeError)
    def test_new_from_template_cannot_be_grown(self):
        garray = GrowableArray((3,))
        garray[:] = [1, 2, 3]
        new_one = garray[1:]
        new_one.grow_by((1,))

    def test_can_grow_array(self):
        garray = GrowableArray((2, 2))
        garray.grow_by((10, 20))
        garray[:, :] = 1
        assert_array_equal(garray, np.ones((12, 22)))

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

    # setting buffer size
