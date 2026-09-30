"""Only this module touches the SDK, and only after explicit CLI live selection."""
from __future__ import annotations
import time
import numpy as np
from . import dependencies
from steadyhand.adapters.vega import VegaAdapter


class FreshVegaAdapter(VegaAdapter):
    def plan_tcp_path(self, targets, motion, floor_m):
        """Pre-solve Cartesian segments and audit sampled joint interpolation.

        Uses no motion commands. The real SDK curve can differ from linear joint
        interpolation, so RobotSession also checks measured tilt while executing.
        """
        from .session import to_pose, from_pose
        from .geometry import angle
        self._require_robot()
        seed = np.asarray(self._read_joint_positions(), dtype=float)
        start = from_pose(self._kinematics.forward(seed))
        plan = []
        for target in targets:
            q = np.asarray(self._kinematics.solve(to_pose(target), seed), dtype=float)
            self._check_joint_limits(q)
            joint_delta = float(np.max(np.abs(q-seed)))
            if joint_delta > motion.max_joint_step_rad:
                raise RuntimeError("IK_JOINT_BRANCH_JUMP: use a different taught route")
            endpoint = from_pose(self._kinematics.forward(q))
            if (np.linalg.norm(endpoint[:3]-target[:3]) > motion.position_tolerance_m
                    or angle(endpoint, target) > motion.orientation_tolerance_rad):
                raise RuntimeError("IK_ENDPOINT_MISMATCH")
            count = max(2, int(np.ceil(joint_delta/motion.joint_path_sample_rad)))
            for fraction in np.linspace(0, 1, count+1):
                point = from_pose(self._kinematics.forward(seed+fraction*(q-seed)))
                motion.check(point, floor_m, measured=True)
                expected_xyz = start[:3]+fraction*(target[:3]-start[:3])
                if np.linalg.norm(point[:3]-expected_xyz) > motion.position_tolerance_m:
                    raise RuntimeError("IK_PATH_LEAVES_CARTESIAN_SEGMENT")
            plan.append(q.copy())
            seed, start = q, endpoint
        return plan

    def move_planned_joints(self, target, *, speed_scale):
        try:
            self._move_joints(target, speed_scale=speed_scale)
        except BaseException:
            self._stop_after_failure()
            raise

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
