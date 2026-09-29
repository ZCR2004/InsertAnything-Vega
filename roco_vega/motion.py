"""Vertical Cartesian motion contract for the physically corrected tip_r frame.

This repository's corrected tip_r has local +Z along base +Z when the physical
claw points down. That tool convention must be verified on the actual robot.
Yaw may change; pitch/roll may not. Checks are not a collision model or a claim
about unsampled points in an undocumented SDK trajectory.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from . import dependencies
from transforms import quat_to_rotmat
from .task_spec import pose


@dataclass(frozen=True)
class VerticalMotion:
    max_translation_step_m: float = .01
    max_rotation_step_rad: float = .05
    command_tilt_tolerance_rad: float = .005
    feedback_tilt_tolerance_rad: float = .05
    position_tolerance_m: float = .004
    orientation_tolerance_rad: float = .04
    max_joint_step_rad: float = .10
    joint_path_sample_rad: float = .01

    @classmethod
    def parse(cls, value=None):
        result = cls(**(value or {}))
        if any(not np.isfinite(v) or v <= 0 for v in vars(result).values()):
            raise ValueError("Motion limits must be finite and positive")
        if not (result.command_tilt_tolerance_rad <= result.feedback_tilt_tolerance_rad <= .12):
            raise ValueError("Invalid vertical tilt limits")
        if (result.max_translation_step_m > .02 or result.max_rotation_step_rad > .10
                or result.max_joint_step_rad > .15 or result.joint_path_sample_rad > .02):
            raise ValueError("Cartesian/joint subdivision limits are too large")
        return result

    def check(self, value, floor_m, *, measured=False):
        p = pose(value, "vertical TCP")
        if p[2] < floor_m:
            raise RuntimeError("TCP_BELOW_FLOOR")
        tilt = float(np.arccos(np.clip(quat_to_rotmat(p[3:])[2, 2], -1, 1)))
        limit = self.feedback_tilt_tolerance_rad if measured else self.command_tilt_tolerance_rad
        if tilt > limit:
            raise RuntimeError(f"TCP_NOT_VERTICAL: tilt={tilt:.5f} rad, limit={limit:.5f}")
        return p

    def path(self, start, target, floor_m):
        from steadyhand.geometry import interpolate_pose
        from .session import to_pose, from_pose
        from .geometry import angle
        start = self.check(start, floor_m, measured=True)
        target = self.check(target, floor_m)
        count = max(1, math.ceil(np.linalg.norm(target[:3]-start[:3])/self.max_translation_step_m),
                    math.ceil(angle(start, target)/self.max_rotation_step_rad))
        if count > 2000:
            raise ValueError("Motion is too long; review the taught route")
        result = [from_pose(interpolate_pose(to_pose(start), to_pose(target), i/count))
                  for i in range(1, count+1)]
        for point in result:
            self.check(point, floor_m, measured=True)
        return result


def pick_heights(pick):
    """Offsets are between TCP poses, not an uncalibrated image-space object top."""
    grasp = float(pick["grasp_z_m"])
    hover_offset = float(pick.get("hover_height_m", .05))
    lift_offset = float(pick.get("lift_height_m", .10))
    if not np.isfinite([grasp, hover_offset, lift_offset]).all() or abs(hover_offset-.05) > 1e-9:
        raise ValueError("Pick hover must be 0.05 m above the calibrated grasp TCP pose")
    if lift_offset < hover_offset:
        raise ValueError("Pick lift must reach at least the 5 cm hover")
    hover = grasp + hover_offset
    # Refuse stale legacy absolute heights rather than silently changing a taught path.
    if pick.get("hover_z_m") is not None and abs(float(pick["hover_z_m"])-hover) > 1e-6:
        raise ValueError("Legacy hover_z_m disagrees with grasp_z_m + hover_height_m")
    return grasp, hover, grasp+lift_offset
