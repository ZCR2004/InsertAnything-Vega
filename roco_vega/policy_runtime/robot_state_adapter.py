"""Policy state; hardware I/O lives in roco_vega.session."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .transforms import quat_to_euler_xyz


@dataclass
class RobotState:
    """Measured state passed to the checkpoint observation builder (poses in xyzw)."""

    pose_BA: np.ndarray
    ee_pose6: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    vel_A: np.ndarray
    gripper_pos: float | None
    force_K: np.ndarray | None = None
    torque_K: np.ndarray | None = None
    raw: dict[str, Any] | None = None

    @property
    def p_BA(self) -> np.ndarray:
        return self.pose_BA[:3]

    @property
    def q_BA(self) -> np.ndarray:
        return self.pose_BA[3:7]

    @property
    def rpy_BA(self) -> np.ndarray:
        return quat_to_euler_xyz(self.q_BA)

    @property
    def v_A(self) -> np.ndarray:
        return self.vel_A[:3]

    @property
    def w_A(self) -> np.ndarray:
        return self.vel_A[3:6]
