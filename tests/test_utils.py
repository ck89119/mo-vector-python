import unittest
import numpy as np
from mo_vector.utils import encode_vector, decode_vector


class TestUtils(unittest.TestCase):
    def test_encode_vector_none(self):
        self.assertIsNone(encode_vector(None))

    def test_encode_vector_list(self):
        self.assertEqual(encode_vector([1.0, 2.0, 3.0]), "[1.0, 2.0, 3.0]")

    def test_encode_vector_ndarray(self):
        self.assertEqual(encode_vector(np.array([1.0, 2.0, 3.0])), "[1.0,2.0,3.0]")

    def test_encode_vector_invalid_ndim(self):
        with self.assertRaises(ValueError):
            encode_vector(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_encode_vector_invalid_dim(self):
        with self.assertRaises(ValueError):
            encode_vector([1.0, 2.0, 3.0], dim=2)

    def test_decode_vector_none(self):
        self.assertIsNone(decode_vector(None))

    def test_decode_vector_empty(self):
        np.testing.assert_array_equal(decode_vector("[]"), np.array([], dtype=np.float64))

    def test_decode_vector(self):
        np.testing.assert_array_equal(decode_vector("[1.0,2.0,3.0]"), np.array([1.0, 2.0, 3.0], dtype=np.float64))
