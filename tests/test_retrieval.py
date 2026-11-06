import math
import unittest

from src.data.retrieval import BM25, HybridIndex, build_chunks, chunk_text, fuse, ranked


class RetrievalTests(unittest.TestCase):
    def test_dense_normalization_and_annotation_independence(self):
        chunks = [{'id': 'a', 'text': 'cat', 'source_id': 'gold'}, {'id': 'b', 'text': 'dog'}]
        first = HybridIndex(chunks, [[2, 0], [0, 3]]).search('cat', [10, 0])
        chunks[0]['source_id'] = 'different'
        second = HybridIndex(chunks, [[2, 0], [0, 3]]).search('cat', [10, 0])
        self.assertEqual(first, second)
        self.assertEqual(first['hybrid'][0]['id'], 'a')
        self.assertAlmostEqual(first['dense'][0][1], 1)

    def test_chunk_preserves_unicode_and_limit(self):
        text = 'abc café سلام\n' * 5
        chunks = list(chunk_text(text, len, 9))
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(0 < len(c) <= 9 for c in chunks))

    def test_no_tool_payload_or_assistant_text(self):
        sessions = [{'turns': [{'role': 'user', 'content': 'request', 'tool_calls': ['SECRET']},
                              {'role': 'assistant', 'content': 'ANSWER'}]}]
        self.assertEqual([c['text'] for c in build_chunks(sessions, len, 50)], ['request'])

    def test_bm25_hand_computed(self):
        score = BM25(['cat', 'dog']).scores('cat')
        self.assertAlmostEqual(score[0], math.log(2))
        self.assertEqual(score[1], 0)
        self.assertEqual(ranked(['b', 'a'], [0, 0])[0][0], 'a')

    def test_fusion_agreement_and_ties(self):
        rows = fuse([('a', 2), ('b', 1)], [('b', 2), ('a', 1)])
        self.assertEqual([r['id'] for r in rows], ['a', 'b'])
        self.assertAlmostEqual(rows[0]['rrf'], 1 / 61 + 1 / 62)
        with self.assertRaises(ValueError):
            fuse([('a', 1)], [('b', 1)])


if __name__ == '__main__':
    unittest.main()
