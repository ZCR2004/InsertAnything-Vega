"""Deterministic fake hardware for integration checks, NOT an insertion simulator."""
from __future__ import annotations
import numpy as np
from .task_spec import ORDER, TaskSpec
from .session import RobotSession, to_pose, from_pose


class FakeClock:
    def __init__(self):
        self.now = 1.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeAdapter:
    def __init__(self):
        self.pose = to_pose([0, 0, .2, 0, 0, 0, 1])
        self.calls = []
        self.sequence = 0
        self.fraction = 1.

    def connect(self):
        self.calls.append("connect")

    def connect_gripper(self):
        self.calls.append("home")

    def close(self):
        self.calls.append("close")

    def stop(self):
        self.calls.append("stop")

    def get_tcp_pose(self):
        return self.pose

    def fresh_tcp_state(self):
        self.sequence += 1
        return self.pose, self.sequence

    def move_tcp(self, target, *, speed_scale):
        self.calls.append("move")
        self.pose = target

    def open_gripper(self, part=None):
        self.calls.append("open")
        self.fraction = 1.

    def grip(self, part=None, *, current_a):
        self.calls.append("grip")
        self.fraction = .4

    def verify_grasp(self, part):
        return True

    def configure_grip(self, current_a, speed_dps):
        self.calls.append("configure_grip")


class FakeActor:
    """Analytic controller exercises the real obs/postprocessing/state machine."""
    def __init__(self):
        self.resets, self.observations = 0, []

    def reset(self):
        self.resets += 1

    def act(self, obs):
        self.observations.append(obs.copy())
        action = np.zeros(6)
        # Simulated policy success is deliberately NONZERO relative to its hole.
        action[:3] = np.clip((np.array([0, 0, .007392])-obs[:3])/.01, -1, 1)
        return action


class FakeFrontend:
    def __init__(self, session):
        self.s, self.calls = session, []
        self.fail_stage = None

    def prepare(self, task):
        self.calls.append((task.task_id, "prepare"))
        return task

    def pick(self, task):
        self.s.open_gripper(task.task_id)
        self.s.grip(task)

    def transfer(self, task):
        self.s.move(task.entry_pose, task.speed_scale)

    def prepare_approach(self, task):
        self.calls.append((task.task_id, "board_retake"))
        return task

    def return_pick(self, task):
        self.calls.append((task.task_id, "return_pick"))
        self.s.open_gripper(task.task_id)

    def verify_already_held(self, task):
        if not self.retained(task):
            raise RuntimeError("EXISTING_GRASP_UNVERIFIED")

    def retained(self, task):
        return self.s.holding and self.fail_stage != "retention"

    def opened(self, task):
        return self.s.adapter.fraction == 1 and self.fail_stage != "release"

    def retreat(self, task):
        self.calls.append((task.task_id, "retreat"))
        self.s.move(task.entry_pose, task.speed_scale)

    def placed(self, task):
        self.calls.append((task.task_id, "placed"))
        return self.fail_stage != "placement"


def synthetic_tasks():
    """These coordinates are fake and are never loaded by the live CLI."""
    result = []
    for i, name in enumerate(ORDER):
        result.append(TaskSpec.parse({
            "task_id": name, "strategy": "planned" if name.startswith("battery") else "rl",
            "calibrated": True, "success_pose_xyzw": [i*.01, 0, .1, 0, 0, 0, 1],
            "entry_height_m": .045, "xy_workspace_m": .02, "step_m": .001,
            "speed_scale": .5,
            "success": {"z_tolerance_m": .001, "xy_tolerance_m": .002,
                "orientation_tolerance_rad": .03, "overshoot_m": .001,
                "dwell_s": .2, "timeout_s": 30, "no_progress_s": 3,
                "progress_m": .0001, "max_sample_gap_s": .2},
            "policy_mapping": {"validated": True,
                "success_pose_policy": [0, 0, .007392, 1, 0, 0, 0],
                "hole_pose_policy": [0, 0, 0, 0, 0, 0, 1],
                "relative_quat_reference_xyzw": [1, 0, 0, 0], "base_yaw_rad": 0},
            "pick": {"current_a": .5}, "transfer_waypoints_xyzw": [], "verification": {},
        }))
    return result


def simulate(journal):
    from .orchestrator import Orchestrator
    clock, adapter, actor = FakeClock(), FakeAdapter(), FakeActor()
    session = RobotSession(adapter, floor_m=0, clock=clock)
    frontend = FakeFrontend(session)
    completed = Orchestrator(session, frontend, actor, journal, clock=clock, sleep=clock.sleep).run(synthetic_tasks())
    return {"mode": "fake_hardware", "completed": completed, "actor_resets": actor.resets,
            "observations": len(actor.observations), "connections": adapter.calls.count("connect"),
            "gripper_homes": adapter.calls.count("home"), "physical_success_claim": False}
