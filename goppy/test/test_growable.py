"""Unit test for growable module."""

from hamcrest import assert_that, equal_to, instance_of, is_, is_not, \
    same_instance
import numpy as np
from numpy.testing import assert_array_equal

from ..growable import GrowableArray


class TestGrowableArray(object):
    def test_creation_of_array(self):
        shape = (2, 3, 4)
        dtype = np.dtype('int')
        order = 'C'
        garray = GrowableArray(shape, dtype, order)

        assert_that(garray.shape, is_(shape))
        assert_that(garray.dtype, is_(dtype))
        assert_that(garray.flags['C'], is_(True))

    def test_array_access(self):
        garray = GrowableArray((2, 2))
        garray[:, :] = 3
        assert_array_equal(garray, [[3, 3], [3, 3]])

    #def test_view_casting(self):
        #array = np.array([[1, 2], [3, 4]])
        #garray = array.view(GrowableArray)
        #assert_that(garray, is_(instance_of(GrowableArray)))
        #assert_array_equal(garray, array)

    #def test_view_casted_cannot_be_grown(self):
        #pass  # FIXME

    def test_new_from_template(self):
        garray = GrowableArray((3,))
        garray[:] = [1, 2, 3]
        new_one = garray[1:]

        assert_that(new_one, is_not(same_instance(garray)))
        assert_array_equal(new_one, [2, 3])

    def test_new_from_template_cannot_be_grown(self):
        pass  # FIXME

    def test_can_grow_array(self):
        garray = GrowableArray((2, 2))
        garray.grow_by((10, 20))
        garray[:, :] = 1
        assert_array_equal(garray, np.ones((12, 22)))

    def test_invalid_assignment(self):
        pass  # FIXME

    # enlarge view casted array
    # enlarge new from template

    # can be enlarged
    # setting buffer size
