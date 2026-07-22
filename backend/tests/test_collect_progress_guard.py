import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.collect_progress import apply_progress_guard


class TestCollectProgressGuard(unittest.TestCase):
    def test_blocks_current_regression(self):
        next_current, next_total, percent, blocked_current, blocked_total = apply_progress_guard(
            existing_current=5,
            existing_total=10,
            existing_percent=50,
            desired_current=4,
            desired_total=10,
            allow_reset=False,
            protect_total_regression=True,
        )
        self.assertTrue(blocked_current)
        self.assertFalse(blocked_total)
        self.assertEqual(next_current, 5)
        self.assertEqual(next_total, 10)
        self.assertEqual(percent, 50)

    def test_blocks_total_regression_when_protected(self):
        next_current, next_total, percent, blocked_current, blocked_total = apply_progress_guard(
            existing_current=2,
            existing_total=10,
            existing_percent=20,
            desired_current=3,
            desired_total=9,
            allow_reset=False,
            protect_total_regression=True,
        )
        self.assertFalse(blocked_current)
        self.assertTrue(blocked_total)
        self.assertEqual(next_total, 10)
        self.assertEqual(next_current, 3)
        self.assertEqual(percent, 30)

    def test_allows_total_regression_when_not_protected(self):
        next_current, next_total, percent, blocked_current, blocked_total = apply_progress_guard(
            existing_current=0,
            existing_total=10,
            existing_percent=0,
            desired_current=0,
            desired_total=6,
            allow_reset=False,
            protect_total_regression=False,
        )
        self.assertFalse(blocked_total)
        self.assertEqual(next_total, 6)
        self.assertEqual(next_current, 0)
        self.assertEqual(percent, 0)


if __name__ == "__main__":
    unittest.main()
