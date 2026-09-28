"""Pose conversion and floor checks for the Vega InsertAnything backend."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SIM2REAL = Path(__file__).resolve().parent
STEADYHAND_ROOT = _SIM2REAL.parents[7] / "ROCO-SteadyHand"
sys.path.insert(0, str(_SIM2REAL))
sys.path.insert(0, str(STEADYHAND_ROOT))

from steadyhand.models import Pose
from vega_backend import (
    VegaInsertRobot,
    pose6_to_steadyhand_pose,
    pose7_to_pose6,
    steadyhand_pose_to_pose7,
    twist_between,
)


class _FakeAdapter:
    def __init__(self, pose: Pose):
        self.pose = pose
        self.moves = []
        self.opened = 0
        self.grips = []

    def observe(self):
        class _Obs:
            joint_positions = (0.0,) * 7
            extras = {"joint_velocity": (0.0,) * 7}

        return _Obs()

    def get_tcp_pose(self):
        return self.pose

    def read_wrench(self):
        return (1.0, 2.0, 3.0, 0.1, 0.2, 0.3)

    def gripper_position(self):
        return 0.4

    def move_tcp(self, pose, *, speed_scale):
        self.moves.append((pose, speed_scale))

    def open_gripper(self):
        self.opened += 1

    def grip(self, current_a=None):
        self.grips.append(current_a)

    def close(self):
        return None


class VegaBackendTest(unittest.TestCase):
    def test_identity_pose_roundtrip(self):
        pose = Pose(position_m=(0.4, -0.1, 0.5), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))
        pose7 = steadyhand_pose_to_pose7(pose)
        self.assertTrue(np.allclose(pose7, [0.4, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0]))
        pose6 = pose7_to_pose6(pose7)
        recovered = pose6_to_steadyhand_pose(pose6)
        self.assertTrue(np.allclose(recovered.position_m, pose.position_m))
        self.assertTrue(np.allclose(recovered.quaternion_wxyz, pose.quaternion_wxyz))

    def test_twist_is_zero_without_motion(self):
        pose7 = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        twist = twist_between(pose7, 1.0, pose7, 1.1)
        self.assertTrue(np.allclose(twist, np.zeros(6)))

    def test_floor_rejects_low_target(self):
        adapter = _FakeAdapter(Pose(position_m=(0.4, 0.0, 0.5), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)))
        robot = VegaInsertRobot(
            STEADYHAND_ROOT,
            adapter=adapter,
            confirm_head_motion=True,
            confirm_physical_motion=True,
        )
        robot._min_tcp_z_m = 0.456
        with self.assertRaises(RuntimeError):
            robot.send_pose6(np.array([0.4, 0.0, 0.40, 0.0, 0.0, 0.0]))
        self.assertEqual(adapter.moves, [])

    def test_state_uses_tcp_and_optional_wrench(self):
        adapter = _FakeAdapter(Pose(position_m=(0.2, 0.0, 0.5), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)))
        robot = VegaInsertRobot(
            STEADYHAND_ROOT,
            adapter=adapter,
            include_wrist_wrench=True,
            confirm_head_motion=True,
            confirm_physical_motion=True,
        )
        state = robot.fetch_state()
        self.assertTrue(np.allclose(state.pose_BA[:3], [0.2, 0.0, 0.5]))
        self.assertTrue(np.allclose(state.force_K, [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(state.vel_A, np.zeros(6)))

    def test_close_gripper_uses_current_limited_grip(self):
        adapter = _FakeAdapter(Pose(position_m=(0.2, 0.0, 0.5), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)))
        robot = VegaInsertRobot(STEADYHAND_ROOT, adapter=adapter, grip_current_a=1.0)
        robot.close_gripper()
        self.assertEqual(adapter.grips, [1.0])


if __name__ == "__main__":
    unittest.main()
