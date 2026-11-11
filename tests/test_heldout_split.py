import unittest
from pathlib import Path
import tempfile

from scripts.verify_heldout_split import build, validate_manifest


class HeldoutSplitTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [
            {
                'qa_id': f'qa_{number:03d}',
                'source_conversation_ids': [f'source_{number:03d}'],
                'query': f'query {number}',
            }
            for number in range(1, 401)
        ]
        exposed = [f'qa_{number:03d}' for number in range(1, 47)]
        linked = [f'qa_{number:03d}' for number in range(47, 50)]
        development = sorted(exposed + linked)
        heldout = [f'qa_{number:03d}' for number in range(50, 401)]
        self.manifest = {
            'version': 1,
            'release_commit': 'pinned-commit',
            'qa_sha256_lf': 'pinned-hash',
            'development_exposed': exposed,
            'development_linked': linked,
            'development': development,
            'heldout': heldout,
        }
        self.release = {
            'commit': 'pinned-commit',
            'files': {'Mem2ACTBench_small/qa_dataset.jsonl': {'sha256_lf': 'pinned-hash'}},
        }

    def test_valid_local_manifest(self):
        self.assertEqual(
            validate_manifest(self.manifest, self.tasks, self.release),
            self.manifest,
        )

    def test_rejects_split_with_shared_source_or_query(self):
        self.tasks[-1]['source_conversation_ids'] = ['source_001']

        with self.assertRaisesRegex(ValueError, 'crosses the frozen split'):
            validate_manifest(self.manifest, self.tasks, self.release)

    def test_rejects_incomplete_partition(self):
        self.manifest['heldout'].pop()

        with self.assertRaisesRegex(ValueError, 'does not partition'):
            validate_manifest(self.manifest, self.tasks, self.release)

    def test_rejects_wrong_dataset_hash(self):
        self.manifest['qa_sha256_lf'] = 'different-hash'

        with self.assertRaisesRegex(ValueError, 'dataset hash'):
            validate_manifest(self.manifest, self.tasks, self.release)

    def test_missing_local_manifest_explains_requirement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, 'intentionally local-only'):
                build(Path(temp_dir) / 'missing-split.json')


if __name__ == '__main__':
    unittest.main()
