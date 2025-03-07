import unittest
import pandas as pd

from include.utils import limit_dataset


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

    def test_limit_dataset(self):
        result = limit_dataset(self.df, 3)

        self.assertEqual(3, result.shape[0])
