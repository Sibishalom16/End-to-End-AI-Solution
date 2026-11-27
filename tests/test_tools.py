"""Unit tests for custom study tools in tools.py."""

import unittest
from tools import (
    calculate_study_hours,
    generate_study_plan,
    calculate_study_hours_tool,
    generate_study_plan_tool,
)


class TestStudyTools(unittest.TestCase):
    """Test suite for calculate_study_hours and generate_study_plan."""

    # ------------------ calculate_study_hours tests ------------------ #

    def test_calculate_study_hours_standard_success(self):
        result = calculate_study_hours(available_days=7, hours_per_day=3.0)
        self.assertIn("21.0 hour(s)", result)
        self.assertIn("7 day(s)", result)
        self.assertIn("Study Hours Calculation Report", result)
        self.assertIn("Pomodoro Blocks", result)

    def test_calculate_study_hours_float_inputs(self):
        result = calculate_study_hours(available_days=10, hours_per_day=2.5)
        self.assertIn("25.0 hour(s)", result)

    def test_calculate_study_hours_string_number_coercion(self):
        result = calculate_study_hours(available_days="5", hours_per_day="4")
        self.assertIn("20.0 hour(s)", result)

    def test_calculate_study_hours_zero_or_negative_days(self):
        res_zero = calculate_study_hours(available_days=0, hours_per_day=4)
        self.assertIn("Validation Error", res_zero)
        self.assertIn("greater than 0", res_zero)

        res_neg = calculate_study_hours(available_days=-3, hours_per_day=4)
        self.assertIn("Validation Error", res_neg)

    def test_calculate_study_hours_excessive_days(self):
        res_excess = calculate_study_hours(available_days=400, hours_per_day=2)
        self.assertIn("Validation Error", res_excess)
        self.assertIn("cannot exceed 365 days", res_excess)

    def test_calculate_study_hours_zero_or_negative_hours(self):
        res_zero = calculate_study_hours(available_days=10, hours_per_day=0)
        self.assertIn("Validation Error", res_zero)
        self.assertIn("greater than 0", res_zero)

        res_neg = calculate_study_hours(available_days=10, hours_per_day=-1.5)
        self.assertIn("Validation Error", res_neg)

    def test_calculate_study_hours_excessive_hours(self):
        res_burnout = calculate_study_hours(available_days=10, hours_per_day=20)
        self.assertIn("Validation Error", res_burnout)
        self.assertIn("cannot exceed 16 hours", res_burnout)

    def test_calculate_study_hours_non_numeric(self):
        res_bad_days = calculate_study_hours(available_days="abc", hours_per_day=4)
        self.assertIn("Validation Error", res_bad_days)

        res_bad_hours = calculate_study_hours(available_days=7, hours_per_day="xyz")
        self.assertIn("Validation Error", res_bad_hours)

    # ------------------ generate_study_plan tests ------------------ #

    def test_generate_study_plan_multi_day_success(self):
        result = generate_study_plan(
            topics="Data Structures, Algorithms, System Design",
            available_days=7,
            hours_per_day=4.0
        )
        self.assertIn("Personalized Strategic Study Plan", result)
        self.assertIn("28.0 Hours", result)
        self.assertIn("Data Structures", result)
        self.assertIn("Algorithms", result)
        self.assertIn("System Design", result)
        self.assertIn("Day 1", result)
        self.assertIn("Day 7", result)
        self.assertIn("Break Strategy", result)

    def test_generate_study_plan_list_input(self):
        topics = ["Calculus", "Linear Algebra", "Differential Equations"]
        result = generate_study_plan(topics=topics, available_days=5, hours_per_day=3.0)
        self.assertIn("Personalized Strategic Study Plan", result)
        self.assertIn("Calculus", result)
        self.assertIn("Linear Algebra", result)

    def test_generate_study_plan_single_day(self):
        result = generate_study_plan(topics="Biology, Chemistry", available_days=1, hours_per_day=6)
        self.assertIn("Day 1: Intensive Single-Day Protocol", result)
        self.assertIn("Biology", result)
        self.assertIn("Chemistry", result)

    def test_generate_study_plan_empty_topics(self):
        res_empty = generate_study_plan(topics="", available_days=7, hours_per_day=3)
        self.assertIn("Validation Error", res_empty)
        self.assertIn("cannot be empty", res_empty)

        res_whitespace = generate_study_plan(topics="   ,  , ", available_days=7, hours_per_day=3)
        self.assertIn("Validation Error", res_whitespace)

    def test_generate_study_plan_invalid_days_or_hours(self):
        res_zero_days = generate_study_plan(topics="Physics", available_days=0, hours_per_day=3)
        self.assertIn("Validation Error", res_zero_days)

        res_too_many_days = generate_study_plan(topics="Physics", available_days=100, hours_per_day=3)
        self.assertIn("Validation Error", res_too_many_days)
        self.assertIn("cannot exceed 90 days", res_too_many_days)

        res_burnout_hours = generate_study_plan(topics="Physics", available_days=5, hours_per_day=24)
        self.assertIn("Validation Error", res_burnout_hours)
        self.assertIn("exceeds safe limit", res_burnout_hours)

    # ------------------ LangChain tool wrappers tests ------------------ #

    def test_tool_wrappers_metadata_and_invocation(self):
        self.assertEqual(calculate_study_hours_tool.name, "calculate_study_hours_tool")
        res_calc = calculate_study_hours_tool.invoke({"available_days": 5, "hours_per_day": 2})
        self.assertIn("10.0 hour(s)", res_calc)

        self.assertEqual(generate_study_plan_tool.name, "generate_study_plan_tool")
        res_plan = generate_study_plan_tool.invoke({
            "topics": "History, Literature",
            "available_days": 3,
            "hours_per_day": 2
        })
        self.assertIn("Personalized Strategic Study Plan", res_plan)


if __name__ == "__main__":
    unittest.main()
