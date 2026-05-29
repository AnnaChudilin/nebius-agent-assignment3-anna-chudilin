import ast
import json
import tempfile
import os
import unittest
import pandas as pd
from app.dataset import set_dataset
from app.tools.structured_tools import (
    count_by_intent,
    list_categories,
    list_intents,
    intent_distribution,
)
from app.tools.summarization_tools import summarize_category
from app.tools.utility_tools import calculate_expression


class TestTools(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, "sample.csv")
        data = {
            "category": ["refund", "shipping", "refund", "feedback"],
            "intent": ["get_refund", "track_refund", "get_refund", "complaint"],
            "text": [
                "Please process my refund.",
                "Where is my package?",
                "I want my money back.",
                "I am unhappy with the service.",
            ],
        }
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)
        set_dataset(self.csv_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def parse_tool_output(value: str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)

    def test_count_by_intent(self):
        result = count_by_intent.invoke("get_refund")
        parsed = self.parse_tool_output(result)
        self.assertEqual(parsed.get("count"), 2)

    def test_list_categories(self):
        result = list_categories.invoke({})
        parsed = self.parse_tool_output(result)
        self.assertEqual(sorted(parsed.get("categories", [])),
                         ["FEEDBACK", "REFUND", "SHIPPING"])

    def test_list_intents(self):
        result = list_intents.invoke({})
        parsed = self.parse_tool_output(result)
        self.assertEqual(sorted(parsed.get("intents", [])), [
                         "complaint", "get_refund", "track_refund"])

    def test_intent_distribution(self):
        result = intent_distribution.invoke({})
        parsed = self.parse_tool_output(result)
        self.assertEqual(parsed.get(
            "intent_distribution", {}).get("get_refund"), 2)
        self.assertEqual(parsed.get(
            "intent_distribution", {}).get("track_refund"), 1)

    def test_summarize_category(self):
        result = summarize_category.invoke("refund")
        parsed = json.loads(result)
        self.assertIn(
            "Total records count for the aggregated REFUND category", parsed.get("summary", ""))
        self.assertIn("refund", parsed.get("summary", "").lower())

    def test_calculate_expression(self):
        result = calculate_expression.invoke("12 + 4 * 2")
        self.assertEqual(result, "20")

    def test_calculate_expression_invalid(self):
        result = calculate_expression.invoke("12 + bad")
        self.assertTrue(result.startswith(
            "Error evaluating mathematical expression:"))


if __name__ == "__main__":
    unittest.main()
