import unittest
from unittest.mock import patch

from src.eval.pilot_outcomes import summarize


class PilotOutcomeTests(unittest.TestCase):
    def test_parse_failures_and_argument_errors_have_separate_denominators(self):
        predictions = [None, {"arguments": {"city": "ROME"}}, {"arguments": {"city": "Milan"}}]
        records = [
            {"parsed": pred is not None, "prediction": pred, "gold_arguments": {"city": "Rome"}}
            for pred in predictions
        ]
        result = summarize(records)
        self.assertEqual(result["unparsed"], 1)
        self.assertEqual(result["parsed_exact_arguments"], 1)
        self.assertEqual(result["parsed_wrong_arguments"], 1)
        self.assertAlmostEqual(result["all_tasks"]["f1"], 0.4)
        self.assertAlmostEqual(result["parsed_calls_only"]["f1"], 0.5)

    def test_all_unparsed_has_no_conditional_score(self):
        self.assertIsNone(summarize([])["parsed_calls_only"])
        records = [{"parsed": False, "prediction": None, "gold_arguments": {"n": 1}}]
        self.assertIsNone(summarize(records)["parsed_calls_only"])

    def test_malformed_structured_call_can_recover_from_text(self):
        # Exercise the graph entry point without loading model weights.
        with patch("llama_cpp.Llama"):
            from src.agent import graph

        message = {
            "tool_calls": [{"function": {"name": "lookup", "arguments": "{"}}],
            "content": '<tool_call>{"name":"lookup","arguments":{"n":1}}</tool_call>',
        }
        state = {
            "query": "Look it up", "tool_schema": {"name": "lookup"}, "memory_context": [],
            "messages": [], "final_tool_call": None,
        }
        with patch.object(graph, "llm") as model:
            usage = {"prompt_tokens": 26, "completion_tokens": 15, "total_tokens": 41}
            model.create_chat_completion.return_value = {
                "choices": [{"message": message, "finish_reason": "stop"}], "usage": usage,
            }
            result = graph.agent_app.invoke(state)
            self.assertEqual(result["final_tool_call"], {"name": "lookup", "arguments": {"n": 1}})
            self.assertEqual(result["generation_metadata"], {"finish_reason": "stop", "usage": usage})
            self.assertEqual(model.create_chat_completion.call_args.kwargs["max_tokens"], 512)
            message["content"] = None
            self.assertIsNone(graph.agent_app.invoke(state)["final_tool_call"])


if __name__ == "__main__":
    unittest.main()
