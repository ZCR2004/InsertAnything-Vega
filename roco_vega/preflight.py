"""Validate selected stage inputs before SDK construction; full runs validate all tasks."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from . import dependencies
from steadyhand.config import read_json, missing_motion_setup
from steadyhand.board_calibration import load_board_calibration
from transforms import quat_to_rotmat
from .task_spec import pose, vector, positive
from .insertion import verify_checkpoint, HEXAGON_SHA256


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def speed(value, name):
    x = positive(value, name)
    if x > 1:
        raise ValueError(f"{name}: must be <= 1")
    return x


def check_preflight(cfg, tasks, *, check_weights=True, stage="full"):
    if stage not in ("full", "plan", "pick", "insert"):
        raise ValueError("Unknown commissioning stage")
    insertion = stage in ("full", "insert")
    picking = stage in ("full", "pick")
    approach = stage in ("full", "plan", "insert")
    uses_rl = insertion and any(t.strategy == "rl" for t in tasks)
    robot = read_json(cfg["robot_config"])
    missing = missing_motion_setup({"robot": robot})
    if missing:
        raise ValueError(f"Robot motion configuration missing: {missing}")
    if (robot["working_arm"] != "right" or robot["kinematics"]["ee_frame"] != "tip_r"
            or robot["gripper"]["scope"] != "right"):
        raise ValueError("Robot configuration must use right arm, tip_r and right gripper")
    if not robot["gripper"].get("home_on_connect", True):
        raise ValueError("Fresh CLI session must home an EMPTY gripper on connect")
    if robot.get("allow_sdk_version_mismatch", False):
        raise ValueError("SDK version mismatch override is not allowed in this entry point")
    wrists = robot["cameras"]["wrists"]["api_label_to_physical_mount"]
    if wrists.get("wrist_a") != "right_wrist":
        raise ValueError("Verify that wrist_a is the physical right camera")
    record = cfg["calibration_identity"]
    expected = {"robot_name": robot["robot_name"], "base_frame": robot["kinematics"]["base_frame"],
                "tcp_frame": "tip_r", "robot_config_sha256": digest(cfg["robot_config"]),
                "board_sha256": digest(cfg["board_calibration"])}
    if uses_rl:
        expected["checkpoint_sha256"] = HEXAGON_SHA256
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Calibration identity mismatch: {key}")
    if not record.get("measured_at_utc") or ((picking or insertion) and not record.get("grasp_convention")):
        raise ValueError("Record calibration date and the held-object/grasp convention")
    board = load_board_calibration(cfg["board_calibration"], robot)
    if board["schema_version"] != 2 or not board["camera_geometry_signature"]:
        raise ValueError("Use a completed five-point board calibration with camera geometry")
    raw_board = np.asarray(board["raw"]["camera_board_read"]["T_base_board_center"], dtype=float)
    if raw_board.shape != (4, 4) or not np.isfinite(raw_board).all():
        raise ValueError("Board calibration must retain its original camera-frame board transform")
    vector(robot["board_calibration"]["camera_target_corrections_m"]["CENTER"], 2, "camera correction")
    floor = float(cfg["tcp_floor_m"])
    if not np.isfinite(floor):
        raise ValueError("TCP floor must be finite")
    positive(cfg["command_timeout_s"], "command timeout")
    positive(cfg["board_max_translation_m"], "board translation")
    positive(cfg["board_max_rotation_deg"], "board rotation")
    speed(cfg["free_speed_scale"], "free speed")
    speed(cfg["head_speed_scale"], "head speed")
    head_q = vector(cfg["head_q_rad"], 3, "head q")
    calibrated_head = vector(board["raw"]["head_q_rad"], 3, "calibration head q")
    if np.max(np.abs(head_q-calibrated_head)) > .01:
        raise ValueError("Use the same head pose as board calibration for image-geometry comparison")

    def waypoint(value):
        p = pose(value, "waypoint")
        if p[2] < floor:
            raise ValueError("Waypoint below configured TCP floor")
        return p

    def route(values, name):
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name}: teach at least one collision-reviewed waypoint")
        for v in values:
            waypoint(v)

    route(cfg["camera_clear_waypoints_xyzw"], "camera clear")
    for task in tasks:
        if approach:
            margin = task.criteria.overshoot_m if insertion else 0
            if task.success_pose[2]-margin < floor:
                raise ValueError(f"{task.task_id}: insertion workspace below TCP floor")
            if quat_to_rotmat(task.success_pose[3:])[2, 2] < np.cos(.12):
                raise ValueError(f"{task.task_id}: A is not corrected vertical tip_r")
            route([v.tolist() for v in task.transfer_waypoints], "transfer")
            if any(p[2] < task.entry_pose[2]-1e-9 for p in task.transfer_waypoints):
                raise ValueError("Transfer waypoints must remain at or above A + entry_height")
        if stage == "plan":
            continue
        p = task.pick
        positive(p["current_a"], "grip current")
        positive(p["speed_dps"], "grip speed")
        low, high = vector(p["held_fraction_range"], 2, "held fraction")
        if not 0 < low < high < float(p["open_min_fraction"]) <= 1:
            raise ValueError("Grip/open fraction thresholds overlap or are invalid")
        if picking:
            vector(p["source_board_xy_m"], 2, "source position")
            positive(p["match_radius_m"], "match radius")
            positive(p["ambiguity_margin_m"], "ambiguity margin")
            size = np.asarray(p["size_range_m"], dtype=float)
            if size.shape != (2, 2) or not np.isfinite(size).all() or np.any(size <= 0) or np.any(size[0] >= size[1]):
                raise ValueError("Invalid component size range")
            q = waypoint([0, 0, p["hover_z_m"], *p["quaternion_xyzw"]])
            if q[2] < floor+.06 or quat_to_rotmat(q[3:])[2, 2] < np.cos(.12):
                raise ValueError("Wrist alignment requires vertical tip_r and 60 mm TCP-floor clearance")
            if not floor <= float(p["grasp_z_m"]) < q[2]:
                raise ValueError("Invalid grasp / hover heights")
            route(p["approach_waypoints_xyzw"], "pick approach")
            for key in ("feature_uv", "goal_uv"):
                if np.any(vector(p[key], 2, key) < 0):
                    raise ValueError("Pixel coordinates must be nonnegative")
            for key in ("descent_speed_scale", "lift_speed_scale"):
                speed(p[key], key)
            servo = p["servo"]
            allowed = {"probe_m", "gain", "max_step_m", "max_radius_m", "tolerance_px", "max_iterations", "speed_scale"}
            if set(servo) != allowed:
                raise ValueError("Specify exactly the documented wrist-servo parameters")
            for key, val in servo.items():
                positive(val, "servo."+key)
            if not (.006 <= servo["probe_m"] <= .015 and servo["max_step_m"] <= .02
                    and servo["probe_m"] <= servo["max_radius_m"] <= .1 and servo["gain"] <= 1
                    and .45 <= servo["speed_scale"] <= 1 and type(servo["max_iterations"]) is int
                    and 1 <= servo["max_iterations"] <= 20):
                raise ValueError("Invalid wrist-servo limits")
        if stage == "insert":
            route([v.tolist() for v in task.manual_load_waypoints], "manual load")
        for verification_stage in (("retention", "placement") if insertion else ("retention",)):
            v = task.verify[verification_stage]
            if verification_stage == "placement":
                waypoint(v["view_pose_xyzw"])
            if v["mode"] == "operator":
                if cfg.get("supervised") is not True:
                    raise ValueError("operator verification requires supervised=true")
            elif v["mode"] == "template":
                path = Path(cfg["config_dir"])/v["template"]
                import cv2
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None or image.std() < 5:
                    raise ValueError(f"Missing or uninformative {verification_stage} image template")
                roi = vector(v["roi_xyxy"], 4, "verification ROI")
                if np.any(roi != np.floor(roi)) or np.any(roi[:2] < 0) or np.any(roi[2:] <= roi[:2]):
                    raise ValueError("Invalid verification ROI")
                if not 0 < float(v["min_score"]) <= 1:
                    raise ValueError("Template threshold must be in (0, 1]")
            else:
                raise ValueError("Verification mode must be operator or template")
        if insertion and task.strategy == "rl":
            from .geometry import PolicyFrame
            PolicyFrame(task)  # validates yaw and both calibrated tool frames
    if check_weights and uses_rl:
        verify_checkpoint(cfg["policy"]["checkpoint"])
    return robot
