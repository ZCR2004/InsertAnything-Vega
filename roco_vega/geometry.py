from __future__ import annotations
import numpy as np
from . import dependencies
from .policy_runtime.transforms import quat_to_rotmat
from .task_spec import pose


def angle(a, b):
    return 2 * np.arccos(np.clip(abs(np.dot(a[3:], b[3:])), 0, 1))


class PolicyFrame:
    """Rigid mapping of real tip_r into an explicitly calibrated virtual tip frame.

    Position axes remain vertical; orientation offset is right-multiplied because
    Vega tip_r and Franka fingertip have different local tool conventions.
    No guessed successful-policy offset is baked into the controller.
    """
    def __init__(self, task):
        from .policy_runtime.transforms import quat_inv, quat_mul
        self.real_success = task.success_pose.copy()
        self.policy_success = pose(task.policy_mapping["success_pose_policy"], "policy success")
        # Optional base yaw maps task insertion XY into the learned task axes.
        yaw = float(task.policy_mapping.get("base_yaw_rad", 0))
        if not np.isfinite(yaw):
            raise ValueError("base_yaw_rad must be finite")
        from .policy_runtime.transforms import euler_xyz_to_quat
        self.q_base = euler_xyz_to_quat(0, 0, yaw)
        self.R = quat_to_rotmat(self.q_base)
        self.q_tool = quat_mul(quat_inv(quat_mul(self.q_base, self.real_success[3:])), self.policy_success[3:])

    def to_policy(self, real):
        from .policy_runtime.transforms import quat_mul
        real = pose(real, "real TCP")
        return np.r_[self.policy_success[:3] + self.R @ (real[:3]-self.real_success[:3]),
                     quat_mul(quat_mul(self.q_base, real[3:]), self.q_tool)]

    def to_real(self, policy):
        from .policy_runtime.transforms import quat_mul, quat_inv
        policy = pose(policy, "policy TCP")
        return np.r_[self.real_success[:3] + self.R.T @ (policy[:3]-self.policy_success[:3]),
                     quat_mul(quat_mul(quat_inv(self.q_base), policy[3:]), quat_inv(self.q_tool))]

    def twist_to_policy(self, twist):
        return np.r_[self.R @ twist[:3], self.R @ twist[3:]]


def constrain_target(current, candidate, task):
    """Project into the intersection of task box and per-command step box."""
    current, candidate = pose(current, "current"), pose(candidate, "candidate")
    a = task.success_pose
    lower = a[:3] + [-task.xy_workspace_m, -task.xy_workspace_m, -task.criteria.overshoot_m]
    upper = a[:3] + [task.xy_workspace_m, task.xy_workspace_m, task.entry_height_m + .002]
    if np.any(current[:3] < lower-1e-8) or np.any(current[:3] > upper+1e-8):
        raise RuntimeError("OUTSIDE_INSERTION_WORKSPACE")
    lo, hi = np.maximum(lower, current[:3]-task.step_m), np.minimum(upper, current[:3]+task.step_m)
    if np.any(lo > hi):
        raise RuntimeError("EMPTY_SAFE_TARGET_INTERSECTION")
    result = candidate.copy()
    result[:3] = np.clip(candidate[:3], lo, hi)
    # First integration locks the taught full orientation, including task yaw.
    result[3:] = a[3:]
    return result
