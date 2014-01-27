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
        garray = GrowableArray(shape, dtype)

        assert_that(garray.shape, is_(shape))
        assert_that(garray.dtype, is_(dtype))

    def test_array_access(self):
        garray = GrowableArray((2, 2))
        garray[:, :] = 3
        assert_array_equal(garray, [[3, 3], [3, 3]])

    # FIXME should not be possible
    def test_view_casting(self):
        array = np.array([[1, 2], [3, 4]])
        garray = array.view(GrowableArray)

        assert_that(garray, is_(instance_of(GrowableArray)))
        assert_array_equal(garray, array)

    # FIXME content should match
    def test_new_from_template(self):
        garray = GrowableArray((3,))
        garray[:] = [1, 2, 3]
        new_one = garray[1:]
        print type(new_one), id(new_one.base), id(garray.base), id(garray)
        # FIXME produces another GrowableArray which should not be the case.
        # Resizing would not work as expected. But the _data attribute could be
        # used to ensure that its not only a view.

        assert_that(new_one, is_not(same_instance(garray)))
        assert_that(new_one, is_(instance_of(GrowableArray)))

    #def test_initializes_buffer_shape_with_twice_initial_sizes(self):
        #array = GrowableArray((2, 2, 3))
        #assert_that(array.buffer_size, is_(equal_to((4, 4, 6))))


    # enlarge view casted array
    # enlarge new from template

    # can be enlarged
    # setting buffer size
