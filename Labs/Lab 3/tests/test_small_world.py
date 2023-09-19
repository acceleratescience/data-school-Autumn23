import unittest
import numpy as np

from small_world_propensity import get_average_paths
from scipy.sparse import csgraph
import math


class TestGetAveragePaths(unittest.TestCase):
    def test_empty_graph(self):
        # Test when the input matrix is an empty graph (all zeros)
        W = np.zeros((5, 5))
        result = get_average_paths(W)
        print(result)
        self.assertTrue(math.isinf(result) or math.isnan(result))

    def test_single_node(self):
        # Test when there is a single node in the graph
        W = np.array([[0]])
        result = get_average_paths(W)
        self.assertTrue(math.isnan(result))

    def test_two_connected_nodes(self):
        # Test when there are only two connected nodes
        W = np.array([[0, 1], [1, 0]])
        result = get_average_paths(W)
        self.assertEqual(result, 1.0)

    def test_odd_number_of_nodes(self):
        # Test when there is an odd number of nodes in the graph
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = get_average_paths(W)
        self.assertAlmostEqual(result, 1.3333, places=4)

    def test_even_number_of_nodes(self):
        # Test when there is an even number of nodes in the graph
        W = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
        result = get_average_paths(W)
        self.assertAlmostEqual(result, 1.6667, places=4)


if __name__ == '__main__':
    unittest.main()