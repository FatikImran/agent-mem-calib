import unittest

from scripts.check_natural_retrieval import hashes_compatible, normalize_path_hashes


class NaturalRetrievalManifestTests(unittest.TestCase):
    def test_normalizes_legacy_research_paths(self):
        legacy = {r'phase2\evidence\paired-compute-v1.json': 'abc'}
        current = {'research/evidence/paired-compute-v1.json': 'abc'}

        self.assertEqual(normalize_path_hashes(legacy), current)

    def test_rejects_normalized_path_collisions(self):
        hashes = {
            r'phase2\evidence\paired-compute-v1.json': 'abc',
            r'research\evidence\paired-compute-v1.json': 'abc',
        }

        with self.assertRaisesRegex(ValueError, 'duplicate hash path'):
            normalize_path_hashes(hashes)

    def test_only_complete_results_ignore_changed_runner_hash(self):
        implementation = r'C:\project\scripts\check_natural_retrieval.py'
        saved = {
            r'phase2\evidence\natural-retrieval-v1.json': 'manifest',
            implementation: 'old-runner',
            r'data\chunks.json': 'chunks',
        }
        expected = {
            'research/evidence/natural-retrieval-v1.json': 'manifest',
            implementation: 'new-runner',
            'data/chunks.json': 'chunks',
        }

        self.assertTrue(hashes_compatible(
            saved, expected, complete=True, implementation_path=implementation))
        self.assertFalse(hashes_compatible(
            saved, expected, complete=False, implementation_path=implementation))
        expected['data/chunks.json'] = 'changed'
        self.assertFalse(hashes_compatible(
            saved, expected, complete=True, implementation_path=implementation))


if __name__ == '__main__':
    unittest.main()
