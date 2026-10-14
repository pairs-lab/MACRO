"""Prevent simulator-specific dependencies from leaking back into core."""

import ast
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ModuleBoundaryTests(unittest.TestCase):
    def test_core_has_no_simulator_imports(self):
        forbidden = {"simulations", "cv2", "rclpy", "ai2thor", "matplotlib"}
        for path in (ROOT / "src/core").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0
                           else [])
                for module in modules:
                    self.assertNotIn(module.split(".")[0], forbidden, str(path))

    def test_importing_coordinator_does_not_load_a_simulator(self):
        import os
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        code = ("import core.coordination.coordinator; import sys; "
                "assert not any(m in sys.modules for m in ('cv2','rclpy','ai2thor'))")
        subprocess.run([sys.executable, "-c", code], env=environment, check=True)

    def test_three_adapters_are_separate(self):
        for name in ("grid", "ai2thor", "ros2"):
            self.assertTrue((ROOT / "src/simulations" / name / "__init__.py").is_file())
