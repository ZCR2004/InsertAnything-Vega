"""Hardware-free contract and failure-path tests: python -m unittest -v test_roco_vega"""
from __future__ import annotations
import contextlib
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from datetime import datetime, timezone
import unittest
from unittest.mock import patch
import numpy as np
from roco_vega import dependencies
from roco_vega.simulation import FakeClock, FakeAdapter, FakeActor, FakeFrontend, synthetic_tasks
from roco_vega.session import RobotSession, to_pose, from_pose
from roco_vega.orchestrator import Journal, Orchestrator
from roco_vega.monitor import InsertionMonitor, Sample
from roco_vega.insertion import RLController, PlannedController, run_insertion
from roco_vega.geometry import PolicyFrame, constrain_target
from roco_vega.frontend import select_part
from roco_vega.task_spec import ORDER, EXECUTION_ORDER, BATTERIES, load_config, TaskSpec
from roco_vega.template import calibration_template
from roco_vega.preflight import check_preflight, digest
from transforms import euler_xyz_to_quat


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock, self.adapter, self.actor = FakeClock(), FakeAdapter(), FakeActor()
        self.session = RobotSession(self.adapter, floor_m=0, clock=self.clock)
        self.frontend = FakeFrontend(self.session)
        self.journal = Journal(Path(self.temp.name)/"run")
        self.runner = Orchestrator(self.session, self.frontend, self.actor, self.journal,
                                   clock=self.clock, sleep=self.clock.sleep)
        self.tasks = synthetic_tasks()

    def run_tasks(self, tasks=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.runner.run(self.tasks if tasks is None else tasks)

    def events(self):
        return [json.loads(v) for v in (self.journal.folder/"events.jsonl").read_text().splitlines()]

    def test_nine_tasks_one_session_and_seven_recurrent_resets(self):
        self.assertEqual(self.run_tasks(), list(ORDER))
        self.assertEqual(self.actor.resets, 7)
        self.assertEqual(self.adapter.calls.count("connect"), 1)
        self.assertEqual(self.adapter.calls.count("home"), 1)
        self.assertEqual(self.adapter.calls.count("close"), 1)
        for obs in self.actor.observations:
            self.assertEqual(obs.shape, (26,))
            np.testing.assert_array_equal(obs[17:20], 0)
        rows = self.events()
        for name in ORDER:
            ev = [r["event"] for r in rows if r.get("task") == name]
            self.assertLess(ev.index("release_verified"), ev.index("retreated"))
            self.assertLess(ev.index("retreated"), ev.index("task_completed"))
        progress = json.loads((self.journal.folder/"progress.json").read_text())
        self.assertEqual(progress["completed"], list(ORDER))

    def test_release_failure_does_not_retreat_or_advance(self):
        self.frontend.fail_stage = "release"
        with self.assertRaisesRegex(RuntimeError, "RELEASE_FAILED"):
            self.run_tasks()
        self.assertEqual(self.journal.completed, [])
        self.assertNotIn((ORDER[0], "retreat"), self.frontend.calls)
        self.assertNotIn((ORDER[1], "prepare"), self.frontend.calls)
        self.assertIn("stop", self.adapter.calls)

    def test_unverified_placement_stops_sequence(self):
        self.frontend.fail_stage = "placement"
        with self.assertRaisesRegex(RuntimeError, "PLACEMENT_UNVERIFIED"):
            self.run_tasks()
        self.assertEqual(self.journal.completed, [])
        self.assertNotIn((ORDER[1], "prepare"), self.frontend.calls)

    def test_lost_grasp_no_release_no_retreat(self):
        self.frontend.fail_stage = "retention"
        with self.assertRaisesRegex(RuntimeError, "GRASP_LOST"):
            self.run_tasks()
        self.assertEqual(self.adapter.calls.count("open"), 1)  # only empty pre-grasp opening
        self.assertNotIn((ORDER[0], "retreat"), self.frontend.calls)

    def test_failed_grip_never_enters_insertion(self):
        self.adapter.verify_grasp = lambda part: False
        with self.assertRaisesRegex(RuntimeError, "GRASP_FAILED"):
            self.run_tasks()
        self.assertFalse(any(r["event"] == "insertion_sample" for r in self.events()))

    def test_fresh_already_held_mode_cannot_home(self):
        with self.assertRaisesRegex(ValueError, "existing connected"):
            self.runner.run(self.tasks[:1], already_held=True)
        self.assertEqual(self.adapter.calls, [])

    def test_invalid_actor_action_aborts(self):
        self.actor.act = lambda obs: np.full(6, np.nan)
        with self.assertRaisesRegex(RuntimeError, "INVALID_POLICY_ACTION"):
            self.run_tasks()
        self.assertEqual(self.adapter.calls.count("open"), 1)

    def test_cancellation_never_submits_followup(self):
        started, release = threading.Event(), threading.Event()
        def blocked(target, *, speed_scale):
            started.set()
            release.wait(2)
        self.adapter.move_tcp = blocked
        self.adapter.stop = lambda: release.set()
        def cancel():
            started.wait(1)
            self.session.cancelled.set()
        t = threading.Thread(target=cancel)
        t.start()
        with self.assertRaisesRegex(RuntimeError, "ABORTED"):
            self.session.move(self.tasks[0].entry_pose, .5)
        t.join(1)
        with self.assertRaisesRegex(RuntimeError, "ABORTED"):
            self.session.open_gripper()
        self.assertNotIn("open", self.adapter.calls)


class MonitorTest(unittest.TestCase):
    def setUp(self):
        self.task = synthetic_tasks()[0]
        self.monitor = InsertionMonitor(self.task, 0)

    def sample(self, when, *, dz=0., dx=0., sequence=None):
        p = self.task.success_pose.copy()
        p[:3] += [dx, 0, dz]
        return Sample(p, when, when if sequence is None else sequence, np.zeros(6))

    def test_requires_continuous_dwell(self):
        self.assertEqual(self.monitor.update(self.sample(.01), .01)[0], "hold")
        self.assertEqual(self.monitor.update(self.sample(.11), .11)[0], "hold")
        self.assertEqual(self.monitor.update(self.sample(.22), .22)[0], "success")

    def test_lateral_error_prevents_depth_success(self):
        self.assertEqual(self.monitor.update(self.sample(.01, dx=.006), .01)[0], "running")

    def test_dwell_resets_on_gap(self):
        self.monitor.update(self.sample(.01), .01)
        self.assertEqual(self.monitor.update(self.sample(.5), .5)[0], "hold")

    def test_stale_repeated_overshoot_timeout(self):
        for label, sample, now in [("STALE_STATE", self.sample(.1), .4),
                                    ("OVERSHOOT", self.sample(.1, dz=-.003), .1),
                                    ("INSERT_TIMEOUT", self.sample(31), 31)]:
            with self.subTest(label=label), self.assertRaisesRegex(RuntimeError, label):
                InsertionMonitor(self.task, 0).update(sample, now)
        self.monitor.update(self.sample(.1, sequence=5), .1)
        with self.assertRaisesRegex(RuntimeError, "REPEATED_STATE"):
            self.monitor.update(self.sample(.2, sequence=5), .2)

    def test_no_progress_timeout(self):
        self.monitor.update(self.sample(.1, dz=.02), .1)
        with self.assertRaisesRegex(RuntimeError, "NO_PROGRESS"):
            self.monitor.update(self.sample(4, dz=.02), 4)


class GeometryTest(unittest.TestCase):
    def test_virtual_success_is_not_zero_and_mapping_inverts(self):
        task = synthetic_tasks()[0]
        task.policy_mapping["base_yaw_rad"] = .7
        frame = PolicyFrame(task)
        mapped = frame.to_policy(task.success_pose)
        np.testing.assert_allclose(mapped, task.policy_mapping["success_pose_policy"], atol=1e-12)
        real = task.entry_pose
        real[:2] += [.003, -.001]
        np.testing.assert_allclose(frame.to_real(frame.to_policy(real)), real, atol=1e-12)

    def test_guard_keeps_step_and_workspace_intersection(self):
        task = synthetic_tasks()[0]
        candidate = task.success_pose.copy()
        candidate[:3] += [10, -10, -10]
        out = constrain_target(task.entry_pose, candidate, task)
        self.assertTrue(np.all(np.abs(out[:3]-task.entry_pose[:3]) <= task.step_m+1e-12))
        bad = task.entry_pose.copy()
        bad[0] += 1
        with self.assertRaisesRegex(RuntimeError, "OUTSIDE"):
            constrain_target(bad, candidate, task)

    def test_translation_includes_placement_view(self):
        task = synthetic_tasks()[0]
        task = replace(task, verify={"placement": {"view_pose_xyzw": task.entry_pose.tolist()}})
        shifted = task.translated([.01, -.02])
        np.testing.assert_allclose(shifted.verify["placement"]["view_pose_xyzw"][:2], [.01, -.02])
        np.testing.assert_allclose(task.success_pose[:2], [0, 0])

    def test_yaw_wrap_near_pi(self):
        from action_postprocessor import ActionPostprocessor, ActionPostprocessorConfig
        processor = ActionPostprocessor(ActionPostprocessorConfig(yaw_enable=True, ema_factor=1))
        current = np.r_[0, 0, 0, euler_xyz_to_quat(0, 0, np.deg2rad(-179))]
        ref = np.r_[0, 0, 0, euler_xyz_to_quat(0, 0, np.deg2rad(179))]
        result, _ = processor.process(current, ref, ref, np.zeros(6), processor.get_initial_state())
        self.assertAlmostEqual(np.rad2deg(result.yaw_target), -179)
        self.assertFalse(result.was_yaw_abs_clipped)

    def test_six_cm_handoff_rejected_before_actor(self):
        clock, adapter = FakeClock(), FakeAdapter()
        task = synthetic_tasks()[0]
        p = task.success_pose.copy()
        p[2] += .06
        adapter.pose = to_pose(p)
        session = RobotSession(adapter, floor_m=0, clock=clock)
        session.holding = True
        actor = FakeActor()
        with self.assertRaisesRegex(RuntimeError, "PREPOSE_NOT_REACHED"):
            run_insertion(session, task, RLController(actor), emit=lambda *a, **k: None,
                          clock=clock, sleep=clock.sleep)
        self.assertEqual(actor.resets, 0)


class ConfigurationTest(unittest.TestCase):
    def test_template_is_rejected_before_connect(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"config.json"
            path.write_text(json.dumps(calibration_template()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "template"):
                load_config(path)

    def test_only_batteries_are_planned(self):
        tasks = synthetic_tasks()
        self.assertEqual(sum(t.strategy == "rl" for t in tasks), 7)
        self.assertEqual([t.task_id for t in tasks if t.strategy == "planned"], list(ORDER[-2:]))

    def test_part_association_rejects_ambiguous_components(self):
        p = {"center_board_m": [.001, 0], "box_board_px": [0, 0, 20, 20]}
        args = ([0, 0], .04, .003, [[.001, .001], [.03, .03]])
        self.assertIs(select_part([p], *args), p)
        with self.assertRaisesRegex(RuntimeError, "AMBIGUOUS"):
            select_part([p, copy.deepcopy(p)], *args)
        with self.assertRaisesRegex(RuntimeError, "NOT_FOUND"):
            select_part([], *args)

    def valid_config(self, folder):
        """Use upstream record structure with synthetic task coordinates; never connect."""
        folder = Path(folder)
        robot = json.loads((dependencies.STEADYHAND/"configs/robots/vega.json").read_text())
        board = json.loads((dependencies.STEADYHAND/"calibration/vega_board_manual_fallback.json").read_text())
        board["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        board["permanent_fallback"] = False
        for name, value in (("robot.local.json", robot), ("board.local.json", board)):
            (folder/name).write_text(json.dumps(value), encoding="utf-8")
        cfg = calibration_template()
        cfg.update(calibrated=True, tcp_floor_m=0., board_max_translation_m=.01,
                   board_max_rotation_deg=1., camera_clear_waypoints_xyzw=[[0, 0, .2, 0, 0, 0, 1]],
                   free_speed_scale=.5, head_q_rad=board["head_q_rad"], head_speed_scale=.5)
        cfg["calibration_identity"].update(robot_name=robot["robot_name"], base_frame=robot["kinematics"]["base_frame"],
            robot_config_sha256=digest(folder/"robot.local.json"), board_sha256=digest(folder/"board.local.json"),
            measured_at_utc=board["generated_at_utc"], grasp_convention="SYNTHETIC TEST ONLY")
        for t, synthetic in zip(cfg["tasks"], synthetic_tasks()):
            high = synthetic.entry_pose.copy()
            high[2] = .20
            t.update(calibrated=True, success_pose_xyzw=synthetic.success_pose.tolist(),
                     xy_workspace_m=.02, speed_scale=.5, policy_mapping=synthetic.policy_mapping,
                     success=dict(vars(synthetic.criteria)), transfer_waypoints_xyzw=[high.tolist()])
            t["pick"].update(source_board_xy_m=[0,0], match_radius_m=.02, ambiguity_margin_m=.003,
                size_range_m=[[.001,.001],[.03,.03]], grasp_z_m=.1,
                quaternion_xyzw=[0,0,0,1], approach_waypoints_xyzw=[[0,0,.2,0,0,0,1]],
                feature_uv=[100,100], goal_uv=[100,100], current_a=.2, speed_dps=60,
                descent_speed_scale=.5, lift_speed_scale=.5, held_fraction_range=[.1,.7], open_min_fraction=.9)
            t["verification"]["placement"]["view_pose_xyzw"] = synthetic.entry_pose.tolist()
        path = folder/"run.local.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return load_config(path)

    def test_complete_preflight_including_three_axis_head(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, tasks = self.valid_config(folder)
            robot = check_preflight(cfg, tasks, check_weights=False)
            self.assertEqual(robot["working_arm"], "right")

    def test_preflight_rejects_changed_robot_file(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, tasks = self.valid_config(folder)
            path = Path(cfg["robot_config"])
            path.write_text(path.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "robot_config_sha256"):
                check_preflight(cfg, tasks, check_weights=False)

    def test_preflight_rejects_unplanned_head_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, tasks = self.valid_config(folder)
            cfg["head_q_rad"][0] += .1
            with self.assertRaisesRegex(ValueError, "same head pose"):
                check_preflight(cfg, tasks, check_weights=False)

    def test_preflight_rejects_incomplete_last_task_before_start(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, tasks = self.valid_config(folder)
            tasks[-1].pick["held_fraction_range"] = [0., 1.]
            with self.assertRaisesRegex(ValueError, "thresholds"):
                check_preflight(cfg, tasks, check_weights=False)

    def test_checkpoint_hash_mismatch(self):
        from roco_vega.insertion import verify_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"wrong.pth"
            path.write_bytes(b"not our checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_checkpoint(path)

    def test_repeated_wrist_frame_not_rescued_by_new_receive_time(self):
        from types import SimpleNamespace
        from roco_vega.frontend import SteadyHandFrontend
        frontend = object.__new__(SteadyHandFrontend)
        frontend.last_wrist_stamp = (1, 200)
        frame = SimpleNamespace(frame_id=1, timestamp_ns=200, received_monotonic_ns=999)
        frontend.s = SimpleNamespace(adapter=SimpleNamespace(capture_wrist=lambda: frame))
        with self.assertRaisesRegex(RuntimeError, "STALE_WRIST"):
            frontend._wrist()

    def test_legacy_guard_rejects_current_outside_box(self):
        from safety_guard import SafetyGuard, SafetyGuardConfig
        guard = SafetyGuard(SafetyGuardConfig(hole_box_upper=(.05,.05,.05),
                                            enable_step_clip=True, step_max_xyz=(.004,)*3))
        hole = np.array([0,0,0,0,0,0,1.])
        current = hole.copy()
        current[2] = .08
        with self.assertRaisesRegex(ValueError, "outside hole box"):
            guard.apply(current, hole, hole, hole)


class StageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock, self.adapter, self.actor = FakeClock(), FakeAdapter(), FakeActor()
        self.session = RobotSession(self.adapter, floor_m=0, clock=self.clock)
        self.frontend = FakeFrontend(self.session)
        self.journal = Journal(Path(self.temp.name)/"run")
        task = synthetic_tasks()[4]
        self.task = replace(task, manual_load_waypoints=(task.entry_pose,),
                            transfer_waypoints=(task.entry_pose,), pick={**task.pick, "speed_dps": 60})

    def execute(self, stage, answer=None):
        from roco_vega.commissioning import run_test
        with contextlib.redirect_stdout(io.StringIO()):
            run_test(stage, self.session, self.frontend, self.task, self.actor, self.journal,
                     prompt=lambda _: answer or {"plan":"move", "pick":"return", "insert":"grip"}[stage],
                     clock=self.clock, sleep=self.clock.sleep)

    def test_plan_never_calls_actor_or_grips(self):
        self.execute("plan")
        self.assertEqual(self.actor.resets, 0)
        self.assertNotIn("grip", self.adapter.calls)
        np.testing.assert_allclose(from_pose(self.adapter.pose), self.task.entry_pose)
        self.assertEqual(self.journal.completed, [])

    def test_pick_returns_object_without_actor_or_transfer(self):
        self.execute("pick")
        self.assertEqual(self.actor.resets, 0)
        self.assertIn((self.task.task_id, "return_pick"), self.frontend.calls)
        self.assertFalse(self.session.holding)
        self.assertEqual(self.journal.completed, [])

    def test_manual_insert_one_connect_one_home_one_reset(self):
        self.execute("insert")
        self.assertEqual(self.adapter.calls.count("connect"), 1)
        self.assertEqual(self.adapter.calls.count("home"), 1)
        self.assertEqual(self.adapter.calls.count("close"), 1)
        self.assertEqual(self.actor.resets, 1)
        self.assertNotIn((self.task.task_id, "prepare"), self.frontend.calls)
        self.assertEqual(self.journal.completed, [self.task.task_id])

    def test_cancel_load_never_grips_or_inserts(self):
        with self.assertRaisesRegex(RuntimeError, "TEST_CANCELLED"):
            self.execute("insert", "cancel")
        self.assertNotIn("grip", self.adapter.calls)
        self.assertEqual(self.actor.resets, 0)

    def test_plan_readback_mismatch_stops(self):
        self.adapter.move_tcp = lambda target, **kwargs: None
        with self.assertRaisesRegex(RuntimeError, "READBACK_MISMATCH"):
            self.execute("plan")
        self.assertIn("stop", self.adapter.calls)

    def test_pick_decline_does_not_automatically_release(self):
        with self.assertRaisesRegex(RuntimeError, "TEST_CANCELLED"):
            self.execute("pick", "cancel")
        self.assertEqual(self.adapter.calls.count("open"), 1)
        self.assertTrue(self.session.holding)

    def partial(self, stage):
        ConfigurationTest().valid_config(self.temp.name)
        path = Path(self.temp.name)/"run.local.json"
        value = json.loads(path.read_text())
        wanted = copy.deepcopy(value["tasks"][4])
        value["tasks"] = calibration_template()["tasks"]
        value.update(calibrated=False, base_calibrated=True)
        if stage == "pick":
            wanted.update(calibrated=False, pick_calibrated=True, success_pose_xyzw=None, policy_mapping=None)
            wanted["verification"].pop("placement")
        elif stage == "plan":
            wanted.update(calibrated=False, approach_calibrated=True, pick={}, policy_mapping=None,
                          verification={}, success={})
        value["tasks"][4] = wanted
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_pick_does_not_require_A_policy_or_other_eight_tasks(self):
        cfg, tasks = load_config(self.partial("pick"), task_id="usb_a", stage="pick")
        with patch("roco_vega.preflight.verify_checkpoint", side_effect=AssertionError("should not read weights")):
            check_preflight(cfg, tasks, stage="pick")
        self.assertFalse(hasattr(tasks[0], "success_pose"))

    def test_plan_does_not_require_pick_policy_or_success_thresholds(self):
        cfg, tasks = load_config(self.partial("plan"), task_id="usb_a", stage="plan")
        with patch("roco_vega.preflight.verify_checkpoint", side_effect=AssertionError("should not read weights")):
            check_preflight(cfg, tasks, stage="plan")

    def test_single_full_task_works_while_other_tasks_are_unfinished(self):
        path = self.partial("full")
        cfg, tasks = load_config(path, task_id="usb_a")
        self.assertEqual(len(tasks), 1)
        check_preflight(cfg, tasks, check_weights=False)
        with self.assertRaisesRegex(ValueError, "template"):
            load_config(path)

    def test_manual_insert_requires_load_route(self):
        cfg, tasks = load_config(self.partial("full"), task_id="usb_a", stage="insert")
        with self.assertRaisesRegex(ValueError, "manual load"):
            check_preflight(cfg, tasks, stage="insert", check_weights=False)

    def test_battery_full_task_needs_no_checkpoint(self):
        from run_roco_vega import prepare_actor
        cfg, tasks = ConfigurationTest().valid_config(self.temp.name)
        self.assertIsNone(prepare_actor(cfg, tasks[:2]))


class PoseRecorderTest(unittest.TestCase):
    def capture(self, *, moving=False, stale=False):
        from record_roco_pose import capture_record
        from types import SimpleNamespace
        self.index = 0
        def sample():
            self.index += 1
            return {"pose": to_pose([self.index*.001 if moving else .1, 0, .2, 0, 0, 0, 1]),
                    "joint_timestamp_ns": 1 if stale else self.index, "joint_positions_rad": [0]*7}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"robot.json"
            cfg = {"robot_name": "fake", "kinematics": {"base_frame":"base", "ee_frame":"tip_r"}}
            path.write_text(json.dumps(cfg))
            return capture_record(SimpleNamespace(capture=sample), cfg, path, "usb_A", sleep=lambda _:None)

    def test_read_only_record_uses_measured_pose(self):
        out = self.capture()
        self.assertEqual(out["pose_xyzw"], [.1,0,.2,0,0,0,1])
        self.assertEqual(out["mode"], "READ_ONLY_NO_MOTION_COMMANDS")

    def test_moving_pose_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "TCP_MOVING"):
            self.capture(moving=True)

    def test_stale_pose_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "STALE_JOINT_STATE"):
            self.capture(stale=True)


class CalibrationIntegrationTest(unittest.TestCase):
    def test_board_retake_uses_delta_from_saved_camera_not_old_correction(self):
        from types import SimpleNamespace
        from roco_vega.frontend import SteadyHandFrontend
        frontend = object.__new__(SteadyHandFrontend)
        baseline = np.eye(4)
        baseline[:2,3] = [.4,.2]
        current = baseline.copy()
        current[:2,3] += [.003,-.002]
        frontend.board = {"raw":{"camera_board_read":{"T_base_board_center":baseline}},
                          "center_base_xy_m":[.45,.18]}
        frontend.cfg = {"camera_clear_waypoints_xyzw":[], "free_speed_scale":.5,
                        "head_q_rad":[.55,0,0], "head_speed_scale":.5, "board_max_translation_m":.02}
        frontend.s = SimpleNamespace(command=lambda *a:None, adapter=SimpleNamespace(move_head=lambda *a:None))
        frontend.j = SimpleNamespace(emit=lambda *a,**k:None)
        frontend._scene = lambda: {"board":{"T_base_board_center":current}}
        _, delta = frontend._retake_board()
        np.testing.assert_allclose(delta,[.003,-.002])
        np.testing.assert_allclose(frontend.center,[.453,.178])

    def test_board_wrapper_forwards_local_config_floor_and_output_without_running_hardware(self):
        import os
        from types import SimpleNamespace
        import run_roco_calibration
        with tempfile.TemporaryDirectory() as folder:
            cfg, _ = ConfigurationTest().valid_config(folder)
            path = Path(folder)/"run.local.json"
            captured = {}
            module = SimpleNamespace(load_bundle=lambda name:{"robot":{"wrong":"upstream"}},
                load_vega_skills=lambda:{"safety":{"min_tcp_z_m":999}})
            def once(argv):
                captured.update(argv=argv, robot=module.load_bundle("vega")["robot"],
                                floor=module.load_vega_skills()["safety"]["min_tcp_z_m"])
                return 0
            module._main_once = once
            with patch.dict(os.environ, {"ROBOT_NAME":cfg["calibration_identity"]["robot_name"]}), \
                 patch("run_roco_calibration.importlib.import_module", return_value=module):
                result = run_roco_calibration.main(["--tool","board","--config",str(path),
                    "--output",str(Path(folder)/"new.json"),"--confirm-physical-motion"])
            self.assertEqual(result,0)
            self.assertEqual(captured["floor"],0)
            self.assertEqual(captured["robot"]["working_arm"],"right")
            self.assertIn("--confirm-physical-motion",captured["argv"])
            self.assertEqual(captured["argv"][-1],str((Path(folder)/"new.json").resolve()))


if __name__ == "__main__":
    unittest.main()
