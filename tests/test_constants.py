from unittest import TestCase
from mo_vector import DistanceMetric


class TestDistanceMetric(TestCase):
    def test_l2(self):
        self.assertEqual("L2", DistanceMetric.L2.value)

    def test_cosine(self):
        self.assertEqual("COSINE", DistanceMetric.COSINE.value)

    def test_to_sql_func_l2(self):
        self.assertEqual("l2_distance", DistanceMetric.L2.to_sql_func())

    def test_to_sql_func_cosine(self):
        self.assertEqual("cosine_distance", DistanceMetric.COSINE.to_sql_func())
