"""Vega robot I/O for the InsertAnything closed loop.

This replaces the Franka HTTP server. Poses use the same API-frame convention
as that server: ``[x, y, z, roll, pitch, yaw]`` in the robot base, metres and
XYZ Euler radians. On Vega the API frame is the SteadyHand TCP
(``kinematics.ee_frame``, competition unit ``tip_r``). ``T_AT`` in the
deployment YAML still maps that frame onto the policy fingertip.

Each ``send_pose6`` blocks inside dexcontrol ``move_to_joint_pos`` until the
waypoint finishes. It does not stream Cartesian impedance at the policy rate.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from robot_state_adapter import RobotState
from transforms import euler_xyz_to_quat, quat_inv, quat_mul, quat_to_euler_xyz


def pose6_to_steadyhand_pose(pose6: np.ndarray):
    """API pose6 ``xyz + rpy`` to a SteadyHand ``Pose`` (wxyz)."""
    from steadyhand.models import Pose

    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(pose6)):
        raise ValueError(f"pose6 contains non-finite values: {pose6}")
    quat_xyzw = euler_xyz_to_quat(float(pose6[3]), float(pose6[4]), float(pose6[5]))
    return Pose(
        position_m=(float(pose6[0]), float(pose6[1]), float(pose6[2])),
        quaternion_wxyz=(
            float(quat_xyzw[3]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
        ),
    )


def steadyhand_pose_to_pose7(pose) -> np.ndarray:
    """SteadyHand ``Pose`` (wxyz) to InsertAnything pose7 ``xyz + xyzw``."""
    x, y, z = pose.position_m
    w, qx, qy, qz = pose.quaternion_wxyz
    return np.array([x, y, z, qx, qy, qz, w], dtype=np.float64)


def pose7_to_pose6(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    rpy = quat_to_euler_xyz(pose7[3:7])
    return np.concatenate([pose7[:3], rpy], axis=0)


def twist_between(prev_pose7: np.ndarray, prev_time_s: float, pose7: np.ndarray, now_s: float) -> np.ndarray:
    """Finite-difference base twist of frame A: linear xyz, angular xyz."""
    dt = float(now_s) - float(prev_time_s)
    if dt < 1e-3:
        return np.zeros(6, dtype=np.float64)
    prev_pose7 = np.asarray(prev_pose7, dtype=np.float64).reshape(7)
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    linear = (pose7[:3] - prev_pose7[:3]) / dt
    q_delta = quat_mul(pose7[3:7], quat_inv(prev_pose7[3:7]))
    if q_delta[3] < 0.0:
        q_delta = -q_delta
    angular = 2.0 * q_delta[:3] / dt
    return np.concatenate([linear, angular], axis=0)


def _optional_gripper_position(adapter) -> float | None:
    try:
        value = adapter.gripper_position()
    except Exception:
        return None
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


class VegaInsertRobot:
    """SteadyHand ``VegaAdapter`` behind the InsertAnything robot methods."""

    def __init__(
        self,
        steadyhand_root: str | Path,
        *,
        speed_scale: float = 0.15,
        grip_current_a: float | None = None,
        assume_grasped: bool = False,
        include_wrist_wrench: bool = False,
        confirm_head_motion: bool = False,
        confirm_physical_motion: bool = False,
        adapter=None,
        owns_adapter: bool | None = None,
    ) -> None:
        if not 0.0 < float(speed_scale) <= 1.0:
            raise ValueError("speed_scale must be in (0, 1]")
        self.steadyhand_root = Path(steadyhand_root).expanduser().resolve()
        self.speed_scale = float(speed_scale)
        self.grip_current_a = grip_current_a
        self.assume_grasped = bool(assume_grasped)
        self.include_wrist_wrench = bool(include_wrist_wrench)
        self.confirm_head_motion = bool(confirm_head_motion)
        self.confirm_physical_motion = bool(confirm_physical_motion)
        self._adapter = adapter
        self._owns_adapter = (adapter is None) if owns_adapter is None else bool(owns_adapter)
        self._min_tcp_z_m: float | None = None
        self._prev_pose7: np.ndarray | None = None
        self._prev_time_s: float | None = None
        self._connected = adapter is not None

    def connect(self) -> None:
        if self._connected:
            return
        if not self.confirm_head_motion or not self.confirm_physical_motion:
            raise ValueError(
                "Vega connection requires --confirm-head-motion and "
                "--confirm-physical-motion. Robot() moves the head."
            )
        root = str(self.steadyhand_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        from steadyhand.adapters.vega import VegaAdapter
        from steadyhand.config import load_bundle
        from steadyhand.skill_config import load_vega_skills

        bundle = load_bundle("vega")
        cfg = bundle["robot"]
        cfg["allow_robot_init_head_motion"] = True
        if self.grip_current_a is not None:
            cfg.setdefault("gripper", {})["grip_current_a"] = float(self.grip_current_a)
        if self.assume_grasped:
            raise ValueError(
                "assume_grasped requires an injected, already-connected and calibrated adapter; "
                "a new Grippers object cannot recover calibration by skipping home"
            )

        safety = dict(load_vega_skills().get("safety") or {})
        floor = safety.get("min_tcp_z_m")
        self._min_tcp_z_m = None if floor is None else float(floor)

        adapter = VegaAdapter(cfg)
        try:
            adapter.connect()
            adapter.connect_gripper()
        except BaseException:
            try:
                adapter.close()
            except BaseException as exc:
                print(f"Vega cleanup after failed connect: {exc}", file=sys.stderr)
            raise
        self._adapter = adapter
        self._connected = True
        ee_frame = (cfg.get("kinematics") or {}).get("ee_frame")
        print(
            f"[VEGA] Connected. TCP/API frame={ee_frame}. "
            f"speed_scale={self.speed_scale}. "
            f"min_tcp_z_m={self._min_tcp_z_m}. "
            "Each pose command blocks until the motion plugin finishes."
        )
        if self.include_wrist_wrench:
            print(
                "[VEGA] Wrist wrench will be copied into force/torque observations. "
                "Units and frame are not verified on this robot."
            )

    def fetch_state(self) -> RobotState:
        adapter = self._require()
        observation = adapter.observe()
        pose7 = steadyhand_pose_to_pose7(adapter.get_tcp_pose())
        now = time.monotonic()
        if self._prev_pose7 is None or self._prev_time_s is None:
            velocity = np.zeros(6, dtype=np.float64)
        else:
            velocity = twist_between(self._prev_pose7, self._prev_time_s, pose7, now)
        self._prev_pose7 = pose7.copy()
        self._prev_time_s = now

        joints = np.asarray(observation.joint_positions, dtype=np.float64).reshape(7)
        raw_velocity = observation.extras.get("joint_velocity")
        joint_velocity = np.asarray(
            np.zeros(7) if raw_velocity is None else raw_velocity,
            dtype=np.float64,
        ).reshape(7)
        force = None
        torque = None
        if self.include_wrist_wrench:
            wrench = np.asarray(adapter.read_wrench(), dtype=np.float64).reshape(6)
            force = wrench[:3].copy()
            torque = wrench[3:].copy()

        return RobotState(
            pose_BA=pose7.copy(),
            ee_pose6=pose7_to_pose6(pose7),
            q=joints,
            dq=joint_velocity,
            vel_A=velocity,
            gripper_pos=_optional_gripper_position(adapter),
            force_K=force,
            torque_K=torque,
            raw={"backend": "vega", "ee_frame": (
                getattr(adapter, "config", {}).get("kinematics", {}).get("ee_frame", "tip_r")
            )},
        )

    def send_pose6(self, pose6: np.ndarray) -> None:
        adapter = self._require()
        pose = pose6_to_steadyhand_pose(pose6)
        if self._min_tcp_z_m is not None and pose.position_m[2] < self._min_tcp_z_m:
            raise RuntimeError(
                f"Refusing TCP z={pose.position_m[2]:.4f} m below Vega floor "
                f"{self._min_tcp_z_m:.4f} m"
            )
        adapter.move_tcp(pose, speed_scale=self.speed_scale)

    def open_gripper(self) -> None:
        self._require().open_gripper()

    def close_gripper(self) -> None:
        """Current-limited grasp. Empty-jaw close is not used while holding a peg."""
        self._require().grip(current_a=self.grip_current_a)

    def shutdown(self) -> None:
        adapter = self._adapter
        self._adapter = None
        self._connected = False
        if adapter is None:
            return
        if self._owns_adapter:
            adapter.close()

    def stop(self) -> None:
        self._require().stop()

    def _require(self):
        if self._adapter is None:
            raise RuntimeError("Vega backend is not connected")
        return self._adapter
