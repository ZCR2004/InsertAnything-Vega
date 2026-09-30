"""Motion/production-frontend checks using synthetic readback, never the SDK."""
from __future__ import annotations
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import test_roco_vega as fixtures
from roco_vega.motion import VerticalMotion, pick_heights
from roco_vega.session import RobotSession, to_pose, from_pose
from roco_vega.simulation import FakeAdapter, FakeClock
from roco_vega.frontend import SteadyHandFrontend
from roco_vega.hardware import FreshVegaAdapter
from roco_vega.orchestrator import Journal, Orchestrator
from roco_vega.task_spec import load_config, EXECUTION_ORDER, BATTERIES
from roco_vega.preflight import check_preflight
from roco_vega.policy_runtime.transforms import euler_xyz_to_quat


class RecordingAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.trace = []

    def move_tcp(self, target, *, speed_scale):
        self.trace.append(("move", from_pose(target)))
        super().move_tcp(target, speed_scale=speed_scale)

    def open_gripper(self, part=None):
        self.trace.append(("open", from_pose(self.pose)))
        super().open_gripper(part)

    def grip(self, part=None, *, current_a):
        self.trace.append(("grip", from_pose(self.pose)))
        super().grip(part, current_a=current_a)

    def gripper_status(self):
        return {"position": self.fraction}

    def gripper_position(self):
        return self.fraction


class MotionTest(unittest.TestCase):
    def test_initial_tilt_rejected_before_gripper_home_or_arm_motion(self):
        adapter = RecordingAdapter()
        adapter.pose = to_pose([0, 0, .2, *euler_xyz_to_quat(.2, 0, 0)])
        session = RobotSession(adapter, floor_m=0)
        with self.assertRaisesRegex(RuntimeError, "NOT_VERTICAL"):
            session.connect()
        self.assertNotIn("home", adapter.calls)
        self.assertEqual(adapter.trace, [])

    def test_bad_target_rejected_before_first_segment(self):
        adapter = RecordingAdapter()
        session = RobotSession(adapter, floor_m=0)
        with self.assertRaisesRegex(RuntimeError, "NOT_VERTICAL"):
            session.move([.1, 0, .2, *euler_xyz_to_quat(0, .1, 0)], .5)
        self.assertEqual(adapter.trace, [])

    def test_yaw_changes_without_tilting_along_cartesian_path(self):
        adapter = RecordingAdapter()
        session = RobotSession(adapter, floor_m=0)
        target = [.11, .02, .21, *euler_xyz_to_quat(0, 0, 1.0)]
        session.move(target, .5)
        previous = np.array([0, 0, .2, 0, 0, 0, 1.])
        self.assertGreater(len(adapter.trace), 10)
        from roco_vega.geometry import angle
        for _, p in adapter.trace:
            session.motion.check(p, 0)
            self.assertLessEqual(np.linalg.norm(p[:3]-previous[:3]), .010001)
            self.assertLessEqual(angle(p, previous), .050001)
            previous = p
        np.testing.assert_allclose(previous, target)

    def test_motion_tilt_is_detected_before_sdk_command_finishes(self):
        adapter = RecordingAdapter()
        release = threading.Event()
        def drift(target, **kwargs):
            adapter.pose = to_pose([0, 0, .2, *euler_xyz_to_quat(.2, 0, 0)])
            release.wait(2)
        adapter.move_tcp = drift
        adapter.stop = release.set
        session = RobotSession(adapter, floor_m=0)
        with self.assertRaisesRegex(RuntimeError, "NOT_VERTICAL"):
            session.move([.01, 0, .2, 0, 0, 0, 1], .5)
        self.assertTrue(session.poisoned)
        self.assertTrue(release.is_set())

    def test_endpoint_mismatch_stops_remaining_segments(self):
        adapter = RecordingAdapter()
        adapter.move_tcp = lambda *args, **kw: adapter.calls.append("ignored")
        with self.assertRaisesRegex(RuntimeError, "READBACK_MISMATCH"):
            RobotSession(adapter, floor_m=0).move([.1, 0, .2, 0, 0, 0, 1], .5)
        self.assertEqual(adapter.calls.count("ignored"), 1)

    def test_late_ik_failure_prevents_all_motion(self):
        adapter = RecordingAdapter()
        def plan(targets, motion, floor):
            self.assertGreater(len(targets), 1)
            raise RuntimeError("late IK failure")
        adapter.plan_tcp_path = plan
        with self.assertRaisesRegex(RuntimeError, "late IK"):
            RobotSession(adapter, floor_m=0).move([.1, 0, .2, 0, 0, 0, 1], .5)
        self.assertEqual(adapter.trace, [])

    def test_joint_midpoint_tilt_rejected_even_when_endpoints_are_vertical(self):
        adapter = object.__new__(FreshVegaAdapter)
        adapter._require_robot = lambda: None
        adapter._read_joint_positions = lambda: [0, 0, .2, 0, 0, 0, 0]
        adapter._check_joint_limits = lambda q: None
        def forward(q):
            tilt = .2 if .004 < q[0] < .006 else 0
            return to_pose([*q[:3], *euler_xyz_to_quat(tilt, 0, 0)])
        adapter._kinematics = SimpleNamespace(forward=forward,
            solve=lambda p, seed: np.r_[p.position_m, np.zeros(4)])
        with self.assertRaisesRegex(RuntimeError, "NOT_VERTICAL"):
            adapter.plan_tcp_path([[.01, 0, .2, 0, 0, 0, 1]], VerticalMotion(), 0)


class ProductionFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg, self.tasks = fixtures.ConfigurationTest().valid_config(self.temp.name)

    def test_full_order_is_batteries_first(self):
        self.assertEqual([t.task_id for t in self.tasks], list(EXECUTION_ORDER))

    def test_battery_subset_accepts_unfinished_rl_tasks_and_needs_no_weights(self):
        from run_roco_vega import prepare_actor
        path = Path(self.temp.name)/"run.local.json"
        cfg = json.loads(path.read_text())
        cfg.update(calibrated=False, base_calibrated=True)
        for t in cfg["tasks"]:
            if t["task_id"] not in BATTERIES:
                t.update(calibrated=False, success_pose_xyzw=None, policy_mapping=None)
        path.write_text(json.dumps(cfg))
        value, tasks = load_config(path, batteries_only=True)
        with patch("roco_vega.preflight.verify_checkpoint", side_effect=AssertionError("weights accessed")):
            check_preflight(value, tasks)
            self.assertIsNone(prepare_actor(value, tasks))
        self.assertEqual([t.task_id for t in tasks], list(BATTERIES))

    def test_preflight_checks_tilt_of_camera_and_reset_routes(self):
        for key in ("camera_clear_waypoints_xyzw", "reset_waypoints_xyzw"):
            with self.subTest(key=key):
                cfg = dict(self.cfg)
                cfg[key] = [[0, 0, .3, *euler_xyz_to_quat(.2, 0, 0)]]
                with self.assertRaisesRegex(ValueError, "NOT_VERTICAL"):
                    check_preflight(cfg, self.tasks, check_weights=False)

    def test_preflight_rejects_low_transfer_and_legacy_hover(self):
        task = self.tasks[0]
        low = replace(task, transfer_waypoints=(task.entry_pose,))
        with self.assertRaisesRegex(ValueError, "above pick lift"):
            check_preflight(self.cfg, [low], check_weights=False)
        with self.assertRaisesRegex(ValueError, "Legacy hover"):
            pick_heights({**task.pick, "hover_z_m": .2})

    def test_real_frontend_two_batteries_grasp_lift_insert_release_15cm_reset(self):
        clock, adapter = FakeClock(), RecordingAdapter()
        session = RobotSession(adapter, floor_m=0, clock=clock)
        frontend = object.__new__(SteadyHandFrontend)
        frontend.s, frontend.cfg = session, dict(self.cfg)
        frontend.cfg["reset_waypoints_xyzw"] = [[0, 0, .30, 0, 0, 0, 1]]
        journal = Journal(Path(self.temp.name)/"actual_frontend")
        frontend.j = journal
        frontend._wrist = lambda: None
        frontend._verify = lambda stage, task: True
        def prepare(task):
            frontend.pick_xy = np.array([-.05, 0.])
            return task
        frontend.prepare = prepare  # Only the camera/scene is replaced; motion code is real.
        def servo(*args, **kwargs):
            p = session.sample().pose
            p[0] += .002
            session.move(p, .5)
            return {"status": "converged"}
        with patch("roco_vega.frontend.run_xy_servo", side_effect=servo), \
             contextlib.redirect_stdout(io.StringIO()):
            completed = Orchestrator(session, frontend, None, journal, clock=clock,
                                     sleep=clock.sleep).run(self.tasks[:2])
        self.assertEqual(completed, list(BATTERIES))
        self.assertEqual(adapter.calls.count("home"), 1)
        self.assertEqual(adapter.calls.count("grip"), 2)
        for index, (name, p) in enumerate(adapter.trace):
            if name != "grip":
                continue
            np.testing.assert_allclose(p[:3], [-.048, 0, .1])
            # Immediately after grip, lift vertically before transporting XY.
            lifted = []
            for event, point in adapter.trace[index+1:]:
                if event != "move" or np.linalg.norm(point[:2]-p[:2]) > 1e-8:
                    break
                lifted.append(point)
            self.assertTrue(lifted)
            self.assertGreaterEqual(max(q[2] for q in lifted), .20-1e-9)
        events = [json.loads(v) for v in (journal.folder/"events.jsonl").read_text().splitlines()]
        for name in BATTERIES:
            rows = [v for v in events if v.get("task") == name]
            lift = next(v for v in rows if v["event"] == "post_release_lift")
            np.testing.assert_allclose(np.array(lift["target_xyzw"][:3])-lift["start_xyzw"][:3], [0, 0, .15])
            labels = [v["event"] for v in rows]
            self.assertLess(labels.index("release_verified"), labels.index("post_release_lift"))
            self.assertLess(labels.index("post_release_lift"), labels.index("ready_for_next_task"))
            self.assertTrue(next(v for v in rows if v["event"] == "ready_for_next_task")["reset_used"])
        np.testing.assert_allclose(from_pose(adapter.pose), frontend.cfg["reset_waypoints_xyzw"][-1])

    def test_no_reset_leaves_robot_at_high_retreat_for_next_task(self):
        clock, adapter = FakeClock(), RecordingAdapter()
        session = RobotSession(adapter, floor_m=0, clock=clock)
        frontend = object.__new__(SteadyHandFrontend)
        frontend.s, frontend.cfg = session, self.cfg
        frontend.j = SimpleNamespace(emit=lambda *a, **kw: None)
        frontend._verify = lambda *a: True
        task = self.tasks[0]
        adapter.pose = to_pose(task.success_pose)
        frontend.retreat(task)
        count = len(adapter.trace)
        self.assertTrue(frontend.placed(task))
        frontend.finish_task(task)
        self.assertEqual(len(adapter.trace), count)
        self.assertAlmostEqual(from_pose(adapter.pose)[2], task.success_pose[2]+.15)


if __name__ == "__main__":
    unittest.main()
