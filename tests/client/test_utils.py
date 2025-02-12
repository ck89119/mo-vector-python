import unittest
from mo_vector.client.utils import rerank_data, rrf_rerank, weighted_rank, convert_metric_score, arctan_normalize


class TestUtils(unittest.TestCase):
    def test_rerank_data_with_rrf(self):
        vector_data = ['doc1', 'doc2']
        full_text_data = ['doc3', 'doc4']
        rerank_option = {'rerank_type': 'RRF', 'rank_value': 60}
        result = rerank_data(vector_data, full_text_data, 4, rerank_option)
        self.assertEqual(result, [
            [1/61, 'doc1'],
            [1/61, 'doc3'],
            [1/62, 'doc2'],
            [1/62, 'doc4']
        ])

    def test_rerank_data_with_weighted_rank(self):
        vector_data = ['doc1', 'doc2']
        full_text_data = ['doc3', 'doc4']
        rerank_option = {'rerank_type': 'WeightedRank', 'weighted_score': [0.6, 0.4]}
        result = rerank_data(vector_data, full_text_data, 4, rerank_option)
        self.assertEqual(result, [
            [0.15, 'doc1'],
            [0.1, 'doc3'],
            [0.08855017059025992, 'doc2'],
            [0.059033447060173286, 'doc4']
        ])

    def test_rrf_rerank_with_valid_data(self):
        vector_data = ['doc1', 'doc2']
        full_text_data = ['doc3', 'doc4']
        result = rrf_rerank(vector_data, full_text_data, 4, 60)
        self.assertEqual(result, [
            [1/61, 'doc1'],
            [1/61, 'doc3'],
            [1/62, 'doc2'],
            [1/62, 'doc4']
        ])

    def test_weighted_rank_with_valid_data(self):
        vector_data = ['doc1', 'doc2']
        full_text_data = ['doc3', 'doc4']
        result = weighted_rank(vector_data, full_text_data, 4, [0.6, 0.4])
        self.assertEqual(result, [
            [0.15, 'doc1'],
            [0.1, 'doc3'],
            [0.08855017059025992, 'doc2'],
            [0.059033447060173286, 'doc4']
        ])

    def test_convert_metric_score_with_l2(self):
        score = convert_metric_score(0.5, 'l2')
        self.assertGreater(score, 0)
        self.assertLess(score, 1)

    def test_arctan_normalize(self):
        score = arctan_normalize(1)
        self.assertEqual(score, 0.75)

        score = arctan_normalize(-1)
        self.assertEqual(score, 0.25)
        