from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .geometry import angle


@dataclass
class Sample:
    pose: np.ndarray
    timestamp: float
    sequence: object
    velocity: np.ndarray


class InsertionMonitor:
    def __init__(self, task, started):
        self.task, self.started = task, started
        self.last = None
        self.last_sequence = None
        self.stable_since = None
        self.best_error = float("inf")
        self.progress_at = started

    def update(self, sample, now):
        c, a = self.task.criteria, self.task.success_pose
        if now-sample.timestamp < -1e-6 or now-sample.timestamp > c.max_sample_gap_s:
            raise RuntimeError("STALE_STATE")
        if self.last is not None:
            if sample.timestamp <= self.last or sample.sequence == self.last_sequence:
                raise RuntimeError("REPEATED_STATE")
            if sample.timestamp-self.last > c.max_sample_gap_s:
                self.stable_since = None
        self.last, self.last_sequence = sample.timestamp, sample.sequence
        if now-self.started >= c.timeout_s:
            raise RuntimeError("INSERT_TIMEOUT")
        z = float(sample.pose[2]-a[2])
        xy = float(np.linalg.norm(sample.pose[:2]-a[:2]))
        if z < -c.overshoot_m:
            raise RuntimeError("OVERSHOOT")
        if xy > self.task.xy_workspace_m or z > self.task.entry_height_m+.002:
            raise RuntimeError("OUTSIDE_INSERTION_WORKSPACE")
        error = max(z, 0)+xy
        if error < self.best_error-c.progress_m:
            self.best_error, self.progress_at = error, now
        candidate = bool(z <= c.z_tolerance_m and xy <= c.xy_tolerance_m and angle(sample.pose, a) <= c.orientation_tolerance_rad)
        if candidate:
            if self.stable_since is None:
                self.stable_since = sample.timestamp
            status = "success" if sample.timestamp-self.stable_since >= c.dwell_s else "hold"
        else:
            self.stable_since = None
            if now-self.progress_at >= c.no_progress_s:
                raise RuntimeError("NO_PROGRESS")
            status = "running"
        return status, {"remaining_z_m": z, "lateral_m": xy,
                        "depth_m": self.task.entry_height_m-z, "candidate": candidate}
