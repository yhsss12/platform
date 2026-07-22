import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.agent_installer_linux_script import (
    apply_linux_installer_replacements,
    build_run_agent_sh,
    resolve_linux_installer_script,
    validate_service_user,
)


class TestAgentInstallerLinuxScript(unittest.TestCase):
    def test_build_run_agent_venv(self) -> None:
        s = build_run_agent_sh(
            use_conda=False,
            agent_dir="/opt/eai-agent",
            ros_setup="/opt/ros/humble/setup.bash",
            ros_ws_setup="",
            conda_sh="",
            conda_env="",
        )
        self.assertIn("#!/usr/bin/env bash", s)
        self.assertIn("cd ", s)
        self.assertIn("/opt/eai-agent", s)
        self.assertIn("cannot cd", s)
        self.assertIn("source /opt/ros/humble/setup.bash", s)
        self.assertIn("PYTHONUNBUFFERED=1", s)
        self.assertIn("/opt/eai-agent/venv/bin/python", s)
        self.assertIn("agent_main.py", s)
        self.assertIn("-u", s)
        self.assertIn("eai-agent: run-agent.sh begin", s)
        self.assertIn("eai-agent: exec", s)
        self.assertIn("trap 'echo eai-agent: err", s)
        self.assertNotIn("conda activate", s)

    def test_build_run_agent_conda_order(self) -> None:
        s = build_run_agent_sh(
            use_conda=True,
            agent_dir="/opt/eai-agent",
            ros_setup="/opt/ros/humble/setup.bash",
            ros_ws_setup="/opt/ws/install/setup.bash",
            conda_sh="/home/u/miniconda3/etc/profile.d/conda.sh",
            conda_env="eai",
        )
        idx_cd = s.index("cd ")
        idx_conda_src = s.index("source /home/u/miniconda3/etc/profile.d/conda.sh")
        idx_conda_act = s.index("conda activate eai")
        idx_ros = s.index("source /opt/ros/humble/setup.bash")
        self.assertLess(idx_cd, idx_conda_src)
        self.assertLess(idx_conda_src, idx_conda_act)
        self.assertLess(idx_conda_act, idx_ros)
        self.assertIn("if [ -f /opt/ws/install/setup.bash ]", s)
        self.assertIn("exec python -u agent_main.py", s)

    def test_resolve_conda_requires_paths(self) -> None:
        st = SimpleNamespace(
            AGENT_INSTALL_USE_CONDA=True,
            AGENT_INSTALL_CONDA_SH="",
            AGENT_INSTALL_CONDA_ENV="eai",
            AGENT_INSTALL_ROS_SETUP="/opt/ros/humble/setup.bash",
            AGENT_INSTALL_ROS_WS_SETUP="",
            AGENT_INSTALL_SERVICE_USER="root",
        )
        with self.assertRaises(ValueError):
            resolve_linux_installer_script(settings=st)

    def test_validate_service_user_auto(self) -> None:
        self.assertEqual(validate_service_user(""), "__SUDO_USER__")
        self.assertEqual(validate_service_user("  __SUDO_USER__  "), "__SUDO_USER__")

    def test_resolve_default_service_is_sudo_placeholder(self) -> None:
        st = SimpleNamespace(
            AGENT_INSTALL_USE_CONDA=False,
            AGENT_INSTALL_CONDA_SH="",
            AGENT_INSTALL_CONDA_ENV="",
            AGENT_INSTALL_ROS_SETUP="/opt/ros/humble/setup.bash",
            AGENT_INSTALL_ROS_WS_SETUP="",
        )
        repl = resolve_linux_installer_script(settings=st)
        self.assertEqual(repl.service_user, "__SUDO_USER__")

    def test_apply_replacements(self) -> None:
        st = SimpleNamespace(
            AGENT_INSTALL_USE_CONDA=False,
            AGENT_INSTALL_CONDA_SH="",
            AGENT_INSTALL_CONDA_ENV="",
            AGENT_INSTALL_ROS_SETUP="/opt/ros/humble/setup.bash",
            AGENT_INSTALL_ROS_WS_SETUP="",
            AGENT_INSTALL_SERVICE_USER="root",
        )
        repl = resolve_linux_installer_script(settings=st)
        raw = 'U="{{SERVICE_USER}}" C="{{AGENT_USE_CONDA}}" B="{{RUN_AGENT_SH_B64}}"'
        out = apply_linux_installer_replacements(raw, repl)
        self.assertIn('U="root"', out)
        self.assertIn('C="0"', out)
        self.assertGreater(len(out), len(raw))


if __name__ == "__main__":
    unittest.main()
