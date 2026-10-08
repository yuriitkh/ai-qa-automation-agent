import os
import unittest
from unittest.mock import MagicMock, patch

from qa_agent.llm.json_schema import (
    normalize_strict_json_schema,
    qa_test_plan_schema,
)
from qa_agent.llm.openai_compatible import OpenAICompatibleProvider
from qa_agent.models import QATestPlan


class OpenAICompatibleSchemaTests(unittest.TestCase):
    def test_normalizer_closes_objects_recursively_without_mutating_input(self):
        source = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"anyOf": [
                        {"type": "object", "properties": {"value": {"type": "string"}}},
                        {"$ref": "#/$defs/Child"},
                    ]},
                },
            },
            "$defs": {
                "Child": {"type": "object", "properties": {"child": {"type": "string"}}}
            },
        }

        result = normalize_strict_json_schema(source)

        def assert_objects_closed(value):
            if isinstance(value, dict):
                if value.get("type") == "object" or "properties" in value:
                    self.assertIs(value.get("additionalProperties"), False)
                for child in value.values():
                    assert_objects_closed(child)
            elif isinstance(value, list):
                for child in value:
                    assert_objects_closed(child)

        assert_objects_closed(result)
        self.assertNotIn("additionalProperties", source)

    def test_qa_plan_schema_represents_action_specific_parameters(self):
        schema = normalize_strict_json_schema(qa_test_plan_schema())
        variants = schema["properties"]["steps"]["items"]["anyOf"]
        by_action = {
            variant["properties"]["action"]["enum"][0]: variant
            for variant in variants
        }
        visible = by_action["assert_visible"]["properties"]["parameters"]
        self.assertEqual(visible["required"], ["url", "expected", "selector", "expected_text", "value"])
        self.assertEqual(visible["properties"]["selector"]["type"], "string")
        self.assertEqual(visible["properties"]["expected_text"]["type"], ["string", "null"])
        self.assertFalse(visible["additionalProperties"])
        click = by_action["click"]["properties"]["parameters"]
        self.assertEqual(click["properties"]["selector"]["type"], "string")
        for action in ("check", "uncheck", "assert_unchecked"):
            with self.subTest(action=action):
                parameters = by_action[action]["properties"]["parameters"]
                self.assertIn("selector", parameters["required"])
                self.assertEqual(parameters["properties"]["selector"]["type"], "string")
        fill = by_action["fill"]["properties"]["parameters"]
        self.assertEqual(fill["properties"]["selector"]["type"], "string")
        self.assertEqual(fill["properties"]["value"]["type"], "string")
        navigate = by_action["navigate"]["properties"]["parameters"]
        self.assertEqual(navigate["properties"]["url"]["type"], "string")
        select = by_action["select_option"]["properties"]["parameters"]
        self.assertIn("option_label", select["required"])
        self.assertFalse(by_action["assert_page_loaded"]["additionalProperties"])

    def test_provider_sends_normalized_schema_for_strict_plan_response(self):
        plan = QATestPlan(url="https://example.com", steps=[{"action": "assert_page_loaded"}])
        response = MagicMock()
        response.choices[0].message.content = plan.model_dump_json()
        with patch.dict(os.environ, {"TEST_LLM_KEY": "unit-test-key"}, clear=True), \
             patch("qa_agent.llm.openai_compatible.OpenAI") as client_cls:
            client_cls.return_value.chat.completions.create.return_value = response
            provider = OpenAICompatibleProvider("test", "TEST_LLM_KEY", "test-model")
            self.assertEqual(provider.create_test_plan("task", plan.url, "{}"), plan)

        request = client_cls.return_value.chat.completions.create.call_args.kwargs
        schema = request["response_format"]["json_schema"]["schema"]
        self.assertEqual(request["response_format"]["json_schema"]["strict"], True)
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(schema, normalize_strict_json_schema(qa_test_plan_schema()))

        def assert_every_object_closed(value):
            if isinstance(value, dict):
                if value.get("type") == "object" or "properties" in value:
                    self.assertIs(value.get("additionalProperties"), False)
                for child in value.values():
                    assert_every_object_closed(child)
            elif isinstance(value, list):
                for child in value:
                    assert_every_object_closed(child)

        assert_every_object_closed(schema)

    def test_provider_accepts_generic_structured_output_schema(self):
        response = MagicMock()
        response.choices[0].message.content = '{"steps":[]}'
        schema = {
            "type": "object",
            "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
            "required": ["steps"],
        }
        with patch.dict(os.environ, {"TEST_LLM_KEY": "unit-test-key"}, clear=True), \
             patch("qa_agent.llm.openai_compatible.OpenAI") as client_cls:
            client_cls.return_value.chat.completions.create.return_value = response
            provider = OpenAICompatibleProvider("test", "TEST_LLM_KEY", "test-model")
            result = provider.create_structured_output("safe prompt", schema, "test_case_authoring")

        self.assertEqual(result, '{"steps":[]}')
        request = client_cls.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["messages"], [{"role": "user", "content": "safe prompt"}])
        self.assertEqual(
            request["response_format"]["json_schema"]["schema"],
            normalize_strict_json_schema(schema),
        )


if __name__ == "__main__":
    unittest.main()
