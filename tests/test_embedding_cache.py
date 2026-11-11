import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.data.embedding_cache import load_bge_cache


class CacheTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / 'data/test-tmp'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        assert Path(self.temp.name).resolve().is_relative_to(parent.resolve())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.text = self.root / 'texts.json'
        self.text.write_text(json.dumps([{'text': 'memory', 'sha256': hashlib.sha256(b'memory').hexdigest()}]))
        config = {'texts_sha256': hashlib.sha256(self.text.read_bytes()).hexdigest()}
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        folder = self.root / 'embeddings' / key
        folder.mkdir(parents=True)
        (folder / 'config.json').write_text(json.dumps(config))
        values = np.zeros((1, 1024), dtype=np.float32)
        values[0, 0] = 1
        self.batch = folder / '000000.npy'
        np.save(self.batch, values)
        summary = {'config': config, 'cache_key': key, 'rows': 1,
            'parts': {'000000.npy': hashlib.sha256(self.batch.read_bytes()).hexdigest()}}
        (self.root / 'embedding-summary.json').write_text(json.dumps(summary))

    def test_valid_cache_and_modified_text_rejection(self):
        vectors, _ = load_bge_cache(self.root)
        self.assertEqual(vectors[hashlib.sha256(b'memory').hexdigest()][0], 1)
        self.text.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'alignment'):
            load_bge_cache(self.root)

    def test_corrupt_vector_file_rejected(self):
        self.batch.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_bge_cache(self.root)


if __name__ == '__main__':
    unittest.main()
