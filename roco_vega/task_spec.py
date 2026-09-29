from __future__ import annotations

from dataclasses import dataclass, replace
import json
import copy
from pathlib import Path
import numpy as np

ORDER = ("gear_60teeth", "gear_20teeth", "rod_16mm", "bolt_8mm", "usb_a",
         "hdmi", "pin", "battery_size1", "battery_size5")
BATTERIES = ("battery_size1", "battery_size5")
EXECUTION_ORDER = BATTERIES + tuple(name for name in ORDER if name not in BATTERIES)


def vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: expected {size} finite numbers")
    return a


def pose(value, name):
    a = vector(value, 7, name)
    if abs(np.linalg.norm(a[3:]) - 1) > 1e-3:
        raise ValueError(f"{name}: quaternion must be unit xyzw")
    return a.copy()


def positive(value, name):
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{name}: must be finite and positive")
    return x


@dataclass(frozen=True)
class SuccessCriteria:
    z_tolerance_m: float
    xy_tolerance_m: float
    orientation_tolerance_rad: float
    overshoot_m: float
    dwell_s: float
    timeout_s: float
    no_progress_s: float
    progress_m: float
    max_sample_gap_s: float

    @classmethod
    def parse(cls, value):
        result = cls(**{k: positive(value[k], f"success.{k}") for k in cls.__annotations__})
        if result.dwell_s >= result.timeout_s or result.no_progress_s >= result.timeout_s:
            raise ValueError("dwell/no_progress must be shorter than timeout")
        return result


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    strategy: str
    success_pose: np.ndarray
    entry_height_m: float
    criteria: SuccessCriteria
    xy_workspace_m: float
    step_m: float
    speed_scale: float
    pick: dict
    transfer_waypoints: tuple
    policy_mapping: dict | None
    verify: dict
    manual_load_waypoints: tuple = ()
    retreat_height_m: float = .15

    @property
    def entry_pose(self):
        p = self.success_pose.copy()
        p[2] += self.entry_height_m
        return p

    def translated(self, delta_xy):
        delta = vector(delta_xy, 2, "board translation")
        p = self.success_pose.copy()
        p[:2] += delta
        waypoints = []
        for v in self.transfer_waypoints:
            q = v.copy()
            q[:2] += delta
            waypoints.append(q)
        verify = copy.deepcopy(self.verify)
        if "view_pose_xyzw" in verify.get("placement", {}):
            view = pose(verify["placement"]["view_pose_xyzw"], "placement view")
            view[:2] += delta
            verify["placement"]["view_pose_xyzw"] = view.tolist()
        return replace(self, success_pose=p, transfer_waypoints=tuple(waypoints), verify=verify)

    @classmethod
    def parse(cls, value):
        name = value["task_id"]
        if name not in ORDER:
            raise ValueError(f"unknown task {name}")
        expected = "planned" if name.startswith("battery_") else "rl"
        if value["strategy"] != expected:
            raise ValueError(f"{name}: strategy must be {expected}")
        if value.get("calibrated") is not True:
            raise ValueError(f"{name}: physical calibration is incomplete")
        h = float(value["entry_height_m"])
        if not .04 <= h <= .05:
            raise ValueError(f"{name}: entry height must be 0.04..0.05 m")
        step = positive(value["step_m"], "step_m")
        if step > .005:
            raise ValueError("contact step must be <= 5 mm")
        speed = positive(value["speed_scale"], "speed_scale")
        if speed > 1:
            raise ValueError("speed_scale must be <= 1")
        criteria = SuccessCriteria.parse(value["success"])
        retreat = positive(value.get("retreat_height_m", .15), "retreat_height_m")
        if abs(retreat-.15) > 1e-9:
            raise ValueError("Post-release retreat must be 0.15 m")
        if criteria.z_tolerance_m >= h:
            raise ValueError("success tolerance must not include entry pose")
        mapping = value.get("policy_mapping")
        if expected == "rl":
            if not mapping or mapping.get("validated") is not True:
                raise ValueError(f"{name}: validated checkpoint frame mapping required")
            pose(mapping["success_pose_policy"], "success_pose_policy")
            pose(mapping["hole_pose_policy"], "hole_pose_policy")
            q = vector(mapping["relative_quat_reference_xyzw"], 4, "relative reference")
            if abs(np.linalg.norm(q)-1) > 1e-3:
                raise ValueError("relative reference quaternion must be unit")
        return cls(name, expected, pose(value["success_pose_xyzw"], name), h, criteria,
                   positive(value["xy_workspace_m"], "xy_workspace_m"), step, speed,
                   dict(value["pick"]), tuple(pose(v, "transfer waypoint") for v in value["transfer_waypoints_xyzw"]),
                   mapping, dict(value["verification"]),
                   tuple(pose(v, "manual load waypoint") for v in value.get("manual_load_waypoints_xyzw", [])),
                   retreat)


@dataclass(frozen=True)
class PickTaskSpec:
    task_id: str
    pick: dict
    verify: dict

    def translated(self, delta_xy):
        vector(delta_xy, 2, "board translation")
        return self  # pick XY is computed from the current board by the frontend


@dataclass(frozen=True)
class ApproachTaskSpec:
    task_id: str
    success_pose: np.ndarray
    entry_height_m: float
    speed_scale: float
    transfer_waypoints: tuple

    @property
    def entry_pose(self):
        p = self.success_pose.copy()
        p[2] += self.entry_height_m
        return p

    def translated(self, delta_xy):
        delta = vector(delta_xy, 2, "board translation")
        def shift(p):
            out = p.copy()
            out[:2] += delta
            return out
        return replace(self, success_pose=shift(self.success_pose),
                       transfer_waypoints=tuple(shift(p) for p in self.transfer_waypoints))


def parse_stage_task(value, stage):
    if stage in ("full", "insert"):
        return TaskSpec.parse(value)
    flag = {"pick": "pick_calibrated", "plan": "approach_calibrated"}[stage]
    if value.get(flag) is not True and value.get("calibrated") is not True:
        raise ValueError(f"{value['task_id']}: complete {flag} first")
    if stage == "pick":
        return PickTaskSpec(value["task_id"], dict(value["pick"]), dict(value["verification"]))
    h = float(value["entry_height_m"])
    if not .04 <= h <= .05:
        raise ValueError("entry height must be 0.04..0.05 m")
    speed = positive(value["speed_scale"], "speed_scale")
    if speed > 1:
        raise ValueError("speed_scale must be <= 1")
    return ApproachTaskSpec(value["task_id"], pose(value["success_pose_xyzw"], "A"), h, speed,
                            tuple(pose(v, "transfer waypoint") for v in value["transfer_waypoints_xyzw"]))


def load_config(path, *, task_id=None, stage="full", batteries_only=False):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if stage not in ("full", "plan", "pick", "insert"):
        raise ValueError("Unknown commissioning stage")
    if stage != "full" and task_id is None:
        raise ValueError("Stage tests require a single task_id")
    if batteries_only and (task_id is not None or stage != "full"):
        raise ValueError("batteries_only is a full-flow subset, incompatible with task/stage selection")
    ready = cfg.get("calibrated") is True or ((task_id is not None or batteries_only) and cfg.get("base_calibrated") is True)
    if cfg.get("schema_version") != 1 or not ready:
        raise ValueError("Configuration is a template: complete and validate physical calibration first")
    if cfg.get("working_arm") != "right" or cfg.get("tcp_frame") != "tip_r":
        raise ValueError("This integration is calibrated for right arm / tip_r only")
    if [v["task_id"] for v in cfg["tasks"]] != list(ORDER):
        raise ValueError("Configuration must contain the nine tasks in canonical order")
    if task_id is not None and task_id not in ORDER:
        raise ValueError(f"Unknown task: {task_id}")
    by_id = {v["task_id"]: v for v in cfg["tasks"]}
    selected = [task_id] if task_id else (BATTERIES if batteries_only else EXECUTION_ORDER)
    tasks = [parse_stage_task(by_id[name], stage) for name in selected]
    for key in ("robot_config", "board_calibration"):
        p = (path.parent / cfg[key]).resolve()
        if not p.is_file():
            raise ValueError(f"Missing {key}: {p}")
        cfg[key] = str(p)
    if stage in ("full", "insert") and any(t.strategy == "rl" for t in tasks):
        profile = cfg["policy"]
        if profile.get("task") != "InsertAnything-HexagonHole-III-Direct-v0" or profile.get("force_mode") != "zeros":
            raise ValueError("Expected Hexagon-III with raw force zeros")
        profile["checkpoint"] = str((path.parent / profile["checkpoint"]).resolve())
    cfg["config_dir"] = str(path.parent)
    return cfg, tasks
