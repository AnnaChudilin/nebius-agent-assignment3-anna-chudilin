import unittest
from app.router import classify_query


class TestRouter(unittest.TestCase):
    def test_structured_query(self):
        result = classify_query("How many refund requests are there?")
        self.assertEqual(result.query_type, "structured")
        self.assertIn("concrete dataset question", result.reason)

    def test_unstructured_query(self):
        result = classify_query("Summarize the FEEDBACK category.")
        self.assertEqual(result.query_type, "unstructured")
        self.assertIn("summary, explanation", result.reason)

    def test_out_of_scope_query(self):
        result = classify_query("Who is the president of France?")
        self.assertEqual(result.query_type, "out_of_scope")
        self.assertIn(
            "unrelated to the customer service dataset", result.reason)

    def test_empty_query(self):
        result = classify_query("")
        self.assertEqual(result.query_type, "out_of_scope")
        self.assertIn("empty or not meaningful", result.reason)


if __name__ == "__main__":
    unittest.main()
