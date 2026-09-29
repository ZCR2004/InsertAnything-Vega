"""Dependency-boundary checks for the trimmed, standalone Vega deployment."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import unittest
import numpy as np

from roco_vega import dependencies
from roco_vega.session import twist_between
from roco_vega.policy_runtime.transforms import euler_xyz_to_quat


class RuntimeDependenciesTest(unittest.TestCase):
    def test_calibration_dynamic_imports(self):
        # Import only: never instantiate the robot, camera or CAN driver.
        for name in ("vega_capture_snapshot", "vega_tool_frame_record",
                     "vega_tool_frame_calibration", "vega_board_five_point_calibrate"):
            with self.subTest(tool=name):
                module = importlib.import_module("tools." + name)
                self.assertTrue(callable(module.main))
                self.assertTrue(Path(module.__file__).resolve().is_relative_to(dependencies.STEADYHAND))

    def test_vega_bundle_and_dynamic_assets_present(self):
        from steadyhand.config import load_bundle
        from steadyhand.skill_config import load_vega_skills
        from steadyhand.grippers.vega import _resolve_driver_path
        bundle = load_bundle("vega")
        self.assertTrue(load_vega_skills())
        root = dependencies.STEADYHAND
        for p in (root / bundle["robot"]["urdf_path"],
                  _resolve_driver_path(bundle["robot"]["gripper"]["driver_path"]),
                  root / bundle["tasks"]["source_coordinates_file"],
                  root / "calibration/vega_tool_frame.template.json"):
            with self.subTest(asset=str(p)):
                self.assertTrue(p.is_file())
        manifest = json.loads((dependencies.ROOT / "third_party/versions.json").read_text())
        for p in manifest["steadyhand"]["included_files"]:
            self.assertTrue((root / p).is_file(), p)

    def test_twist_frame_sign_and_units(self):
        before = np.array([.1, .2, .3, 0, 0, 0, 1.])
        after = before.copy()
        after[:3] += [.001, -.002, .003]
        after[3:] = euler_xyz_to_quat(0, 0, .001)
        measured = twist_between(before, 10., after, 10.1)
        np.testing.assert_allclose(measured[:3], [.01, -.02, .03], atol=1e-12)
        np.testing.assert_allclose(measured[3:], [0, 0, .01], atol=1e-8)
        after[3:] *= -1  # q and -q describe the same orientation.
        np.testing.assert_allclose(twist_between(before, 10., after, 10.1), measured)

    def test_twist_stationary_and_near_zero_interval(self):
        p = np.array([.1, .2, .3, 0, 0, 0, 1.])
        np.testing.assert_array_equal(twist_between(p, 1., p, 1.1), np.zeros(6))
        np.testing.assert_array_equal(twist_between(p, 1., p, 1.), np.zeros(6))


if __name__ == "__main__":
    unittest.main()
