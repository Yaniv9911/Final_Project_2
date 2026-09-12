"""Unit tests for validate_ticket in 2_extract_tickets.py — no API calls.

The module under test starts with a digit, so it can't be imported with a
normal `import` statement — load it from its file path instead.

Run with: python test_extract_tickets.py
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "extract_tickets", Path(__file__).parent / "2_extract_tickets.py"
)
extract_tickets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_tickets)

validate_ticket = extract_tickets.validate_ticket


class TestValidateTicket(unittest.TestCase):
    def test_valid_ticket_has_no_errors(self):
        ticket = {
            "id": "r001", "category": "shipping", "severity": "medium",
            "refund_requested": False, "summary": "Package never arrived.",
        }
        self.assertEqual(validate_ticket(ticket, "r001"), [])

    def test_rejects_invalid_category(self):
        ticket = {
            "id": "r002", "category": "delivery", "severity": "low",
            "refund_requested": True, "summary": "Wrong item shipped.",
        }
        errors = validate_ticket(ticket, "r002")
        self.assertTrue(any("delivery" in e for e in errors))

    def test_rejects_invalid_severity(self):
        ticket = {
            "id": "r003", "category": "shipping", "severity": "critical",
            "refund_requested": False, "summary": "Box was crushed.",
        }
        errors = validate_ticket(ticket, "r003")
        self.assertTrue(any("critical" in e for e in errors))

    def test_rejects_missing_key(self):
        ticket = {
            "id": "r004", "category": "shipping", "severity": "low",
            "refund_requested": False,
        }
        errors = validate_ticket(ticket, "r004")
        self.assertTrue(any("summary" in e and "missing" in e for e in errors))

    def test_rejects_extra_key(self):
        ticket = {
            "id": "r005", "category": "shipping", "severity": "low",
            "refund_requested": False, "summary": "x", "priority": "high",
        }
        errors = validate_ticket(ticket, "r005")
        self.assertTrue(any("priority" in e and "extra" in e for e in errors))

    def test_rejects_wrong_type(self):
        ticket = {
            "id": "r006", "category": "shipping", "severity": "low",
            "refund_requested": "yes", "summary": "x",
        }
        errors = validate_ticket(ticket, "r006")
        self.assertTrue(any("refund_requested" in e and "boolean" in e for e in errors))

    def test_rejects_mismatched_id(self):
        ticket = {
            "id": "r999", "category": "shipping", "severity": "low",
            "refund_requested": False, "summary": "x",
        }
        errors = validate_ticket(ticket, "r006")
        self.assertTrue(any("expected 'r006'" in e for e in errors))

    def test_rejects_non_dict(self):
        self.assertEqual(len(validate_ticket("not a dict", "r001")), 1)


if __name__ == "__main__":
    unittest.main()
