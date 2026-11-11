import json
import unittest

from src.agent.memory_prompts import generator_input, judge_input, judge_response_schema, require_capacity


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.schema = {'name': 'image', 'parameters': {'properties': {'width': {}, 'height': {}}}}
        self.memories = [{'id': 'm0', 'text': '400 pixels', 'source_id': 'EVALUATOR_ONLY'},
                         {'id': 'm1', 'text': 'DO_NOT_ADMIT'}]

    def test_parameter_scope_and_omission(self):
        result = generator_input('query', self.schema, self.memories, {'m0': ['width']})
        prompt = result['messages'][0]['content']
        self.assertIn('Use only for these parameters: width. 400 pixels', prompt)
        self.assertNotIn('DO_NOT_ADMIT', prompt)
        self.assertNotIn('EVALUATOR_ONLY', prompt)
        with self.assertRaises(ValueError):
            generator_input('query', self.schema, self.memories, {'m0': ['unknown']})

    def test_judge_payload_excludes_provenance_and_schema_covers_all_pairs(self):
        payload = judge_input('query', self.schema, self.memories)['messages'][1]['content']
        self.assertNotIn('EVALUATOR_ONLY', payload)
        self.assertEqual(json.loads(payload)['memories'][0], {'id': 'm0', 'text': '400 pixels'})
        required = judge_response_schema(self.schema, self.memories)
        self.assertEqual(required['required'], ['m0', 'm1'])
        self.assertEqual(required['properties']['m0']['required'], ['width', 'height'])

    def test_budget_overflow_raises_without_truncation(self):
        require_capacity(1024, 1024, 2048)
        with self.assertRaises(ValueError):
            require_capacity(1025, 1024, 2048)


if __name__ == '__main__':
    unittest.main()
