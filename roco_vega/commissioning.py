"""Isolated physical bring-up stages, sharing the production session and frontend."""
from __future__ import annotations
import time
import numpy as np
from .geometry import angle
from .orchestrator import Orchestrator


def require_word(prompt, text, expected):
    if prompt(text).strip().lower() != expected:
        raise RuntimeError("TEST_CANCELLED")


def run_test(stage, session, frontend, task, actor, journal, *, prompt=input,
             clock=time.monotonic, sleep=time.sleep):
    if stage not in ("plan", "pick", "insert"):
        raise ValueError("Unknown test stage")
    delegated = False
    try:
        session.connect()  # empty gripper, once only
        if stage == "pick":
            task = frontend.prepare(task)
            frontend.pick(task)
            journal.emit("pick_test_lift_verified", task=task.task_id)
            require_word(prompt, "Check lifted grasp; clear the original source spot. Type return to put it back: ", "return")
            session.check()
            frontend.return_pick(task)
            journal.emit("stage_test_passed", stage=stage, task=task.task_id)
            return

        task = frontend.prepare_approach(task)  # refresh board before loading an object
        if stage == "plan":
            # Empty-jaw approach test ends ABOVE A, never moves to A or runs RL.
            for index, target in enumerate((*task.transfer_waypoints, task.entry_pose)):
                journal.emit("planned_target", index=index, target_xyzw=target)
                require_word(prompt, f"Empty gripper: execute approach waypoint {index}? Type move: ", "move")
                session.move(target, task.speed_scale)
                sample = session.sample()
                error = float(np.linalg.norm(sample.pose[:3]-target[:3]))
                orientation_error = float(angle(sample.pose, target))
                journal.emit("plan_readback", index=index, target_xyzw=target, measured_xyzw=sample.pose,
                             position_error_m=error, orientation_error_rad=orientation_error)
                if error > .004 or orientation_error > .04:
                    raise RuntimeError("APPROACH_READBACK_MISMATCH")
            journal.emit("stage_test_passed", stage=stage, task=task.task_id)
            return

        # No reconnect/home after loading. Camera-clear and board retake have
        # already finished; the remaining route is load -> prepose -> insertion.
        for waypoint in task.manual_load_waypoints:
            session.move(waypoint, task.speed_scale)
        session.open_gripper(task.task_id)
        if not frontend.opened(task):
            raise RuntimeError("GRIPPER_NOT_OPEN")
        require_word(prompt, "Place the object in the taught grasp; remove hands. Type grip to close: ", "grip")
        session.check()
        session.adapter.configure_grip(task.pick["current_a"], task.pick["speed_dps"])
        session.grip(task)
        journal.emit("manual_load_gripped", task=task.task_id)
        delegated = True
        Orchestrator(session, frontend, actor, journal, clock=clock, sleep=sleep).run([task], already_held=True)
        journal.emit("stage_test_passed", stage=stage, task=task.task_id)
    except BaseException as exc:
        if not delegated:
            try:
                session.abort()
            finally:
                journal.emit("stage_test_stopped", stage=stage, error=type(exc).__name__, reason=str(exc))
        raise
    finally:
        if not delegated:
            session.close()
