import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.dispatcher import route_task


class TestAgentInstallDispatcher(unittest.TestCase):
    def test_route_task_agent_install_removed(self):
        with self.assertRaises(ValueError):
            route_task({"type": "agent_install"})


if __name__ == "__main__":
    unittest.main()
