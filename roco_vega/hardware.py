"""Only this module touches the SDK, and only after explicit CLI live selection."""
from __future__ import annotations
import time
import numpy as np
from . import dependencies
from steadyhand.adapters.vega import VegaAdapter


class FreshVegaAdapter(VegaAdapter):
    def fresh_tcp_state(self):
        first = self._state_timestamp()
        deadline = time.monotonic()+.5
        while True:
            stamp = self._state_timestamp()
            if stamp > first:
                return self.get_tcp_pose(), stamp
            if time.monotonic() >= deadline:
                raise RuntimeError("STALE_SDK_STATE")
            time.sleep(.005)

    def configure_grip(self, current_a, speed_dps):
        self._require_gripper()
        self._gripper.config["grip_current_a"] = current_a
        self._gripper.config["grip_speed_dps"] = speed_dps

    def move_head(self, q, speed):
        handle = self._robot.head.move_to_joint_pos(q, velocity_scale=speed)
        # Register the head command too so session cancellation can cancel it.
        self._active_motion_handle = handle
        if handle.wait(timeout=5) != "finished":
            raise RuntimeError("HEAD_MOTION_FAILED")
        self._active_motion_handle = None
        if np.max(np.abs(np.asarray(self._robot.head.get_joint_pos())-q)) > .03:
            raise RuntimeError("HEAD_NOT_REACHED")

    def capture_head(self):
        self.connect_cameras(head=True, wrists=False)
        # Discard the currently buffered image after head/arm motion and wait
        # for a sensor timestamp advance, including the very first capture.
        first = self._head_camera.read(include_depth=False).left_timestamp_ns
        if first is None:
            raise RuntimeError("HEAD_IMAGE_HAS_NO_TIMESTAMP")
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            frame = self._head_camera.read(include_depth=False, timeout_s=3)
            if frame.left_timestamp_ns is not None and frame.left_timestamp_ns > first:
                return frame, self._robot.head.get_joint_pos()
            time.sleep(.02)
        raise RuntimeError("STALE_HEAD_IMAGE")

    def capture_wrist(self):
        self.connect_cameras(head=False, wrists=True)
        return self._wrist_cameras.read(timeout=3, fresh=True).wrist_a
