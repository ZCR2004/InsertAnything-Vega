from __future__ import annotations
import threading
import time
import numpy as np
from . import dependencies
from steadyhand.models import Pose
from .task_spec import pose


def to_pose(value):
    p = pose(value, "TCP")
    return Pose(tuple(p[:3]), (float(p[6]), *map(float, p[3:6])))


def from_pose(p):
    w, x, y, z = p.quaternion_wxyz
    return pose([*p.position_m, x, y, z, w], "TCP readback")


class RobotSession:
    """One owner of the adapter; motion waits are cancelable from the main thread.

    Only one SDK command is outstanding. On timeout/cancel the session is poisoned:
    no release, retreat, reconnect or new command is allowed.
    """
    def __init__(self, adapter, *, floor_m, command_timeout_s=30, clock=time.monotonic):
        self.adapter, self.floor_m = adapter, float(floor_m)
        self.command_timeout_s, self.clock = float(command_timeout_s), clock
        self.cancelled = threading.Event()
        self.connected = False
        self.poisoned = False
        self.holding = False
        self._active = None
        self._last = None

    def check(self):
        if self.cancelled.is_set() or self.poisoned:
            raise RuntimeError("ABORTED")

    def connect(self):
        self.check()
        if self.connected:
            return
        try:
            self.adapter.connect()
            self.connected = True
            self.adapter.connect_gripper()  # once, while empty; never re-home while holding
        except BaseException:
            self.adapter.close()
            self.connected = False
            raise

    def abort(self):
        self.cancelled.set()
        self.poisoned = True
        self.adapter.stop()

    def command(self, fn, *args, **kwargs):
        self.check()
        if self._active is not None and self._active.is_alive():
            raise RuntimeError("ANOTHER_MOTION_IS_ACTIVE")
        done, errors = threading.Event(), []
        def run():
            try:
                if not self.cancelled.is_set():
                    fn(*args, **kwargs)
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()
        self._active = threading.Thread(target=run, daemon=True)
        self._active.start()
        deadline = time.monotonic()+self.command_timeout_s
        try:
            while not done.wait(.02):
                if self.cancelled.is_set() or time.monotonic() >= deadline:
                    raise RuntimeError("ABORTED" if self.cancelled.is_set() else "MOTION_TIMEOUT")
            self.check()
            if errors:
                raise errors[0]
        except BaseException:
            self.abort()
            done.wait(1)
            raise

    def move(self, value, speed):
        p = pose(value, "move target")
        if p[2] < self.floor_m:
            raise RuntimeError("TARGET_BELOW_FLOOR")
        self.command(self.adapter.move_tcp, to_pose(p), speed_scale=speed)

    # SteadyHand servo/executor facade: its internal moves also use cancellable commands.
    def get_tcp_pose(self):
        self.check()
        return self.adapter.get_tcp_pose()

    def move_tcp(self, p, *, speed_scale=1):
        self.move(from_pose(p), speed_scale)

    def stop(self):
        self.abort()

    def sample(self):
        from .monitor import Sample
        from vega_backend import twist_between
        self.check()
        # SDK freshness must be checked by the concrete runtime; local receive time
        # alone is not evidence that a sensor sample is fresh.
        measured, sequence = self.adapter.fresh_tcp_state()
        now = self.clock()
        p = from_pose(measured)
        velocity = np.zeros(6) if self._last is None else twist_between(self._last.pose, self._last.timestamp, p, now)
        out = Sample(p, now, sequence, velocity)
        self._last = out
        return out

    def open_gripper(self, part=None):
        self.command(self.adapter.open_gripper, part)
        self.holding = False

    def grip(self, task):
        self.command(self.adapter.grip, task.task_id, current_a=task.pick["current_a"])
        if self.adapter.verify_grasp(task.task_id) is not True:
            raise RuntimeError("GRASP_FAILED")
        self.holding = True

    def close(self):
        if self._active is not None and self._active.is_alive():
            # Do not race SDK teardown against an unresponsive command thread.
            self.abort()
            self._active.join(2)
            if self._active.is_alive():
                raise RuntimeError("SDK command did not stop; hardware stop must be verified")
        self.adapter.close()
        self.connected = False
