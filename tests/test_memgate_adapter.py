import unittest

from src.agent.memgate_adapter import configure_gate_only, filter_gate_only


class LoadedWrapper:
    threshold = .5
    safety_memory = 'checkpoint safety'
    objective_memory = 'checkpoint objective'

    def eval(self):
        return self

    def score_candidates(self, query, candidates):
        return [.1] * len(candidates)

    def __call__(self, *, query, candidates, top_k, preserve_order):
        assert top_k is None and preserve_order
        return candidates, [.1] * len(candidates), []


class AdapterTests(unittest.TestCase):
    def test_checkpoint_guidance_is_disabled_and_sixth_candidate_survives(self):
        defender = LoadedWrapper()
        memories = [{'id': f'm{i}', 'text': str(i)} for i in range(6)]
        with self.assertRaises(ValueError):
            filter_gate_only(defender, 'query', memories)
        configure_gate_only(defender)
        result = filter_gate_only(defender, 'query', memories)
        self.assertEqual(result['kept_ids'], [m['id'] for m in memories])
        self.assertIsNone(defender.safety_memory)
        self.assertIsNone(defender.objective_memory)


if __name__ == '__main__':
    unittest.main()
