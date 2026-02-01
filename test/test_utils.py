import unittest

import numpy as np
from numpy.testing import assert_array_equal

from jed_teleop.utils.utils import combine_points_width_depth


class TestUtils(unittest.TestCase):
    def test_combine_points_width_depth(self):
        depth_image = np.zeros((101, 201))
        points = np.array([[0.2, 0.4],
                  [0.1, 0.9],
                  [0.5, 0.8]])
        depth_image[40, 40] = 9
        depth_image[90, 20] = 2
        depth_image[80, 100] = 1
        actual = combine_points_width_depth(points, depth_image)
        expected_points = np.array([[0.2, 0.4, 9],
                  [0.1, 0.9, 2],
                  [0.5, 0.8, 1]])
        assert_array_equal(actual, expected_points)

if __name__ == '__main__':
    unittest.main()
