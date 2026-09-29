"""Generate the nine-task calibration contract without inventing robot coordinates."""
from .task_spec import ORDER
from .insertion import HEXAGON_SHA256
from .motion import VerticalMotion


def calibration_template():
    tasks = []
    for name in ORDER:
        rl = not name.startswith("battery_")
        tasks.append({"task_id": name, "strategy": "rl" if rl else "planned", "calibrated": False,
            "pick_calibrated": False, "approach_calibrated": False,
            "manual_load_waypoints_xyzw": [],
            "success_pose_xyzw": None, "entry_height_m": .045, "retreat_height_m": .15,
            "xy_workspace_m": None, "step_m": .001, "speed_scale": None,
            "success": {"z_tolerance_m": None, "xy_tolerance_m": None, "orientation_tolerance_rad": None,
                "overshoot_m": None, "dwell_s": .3, "timeout_s": 30, "no_progress_s": 5,
                "progress_m": .0002, "max_sample_gap_s": .5},
            "policy_mapping": {"validated": False, "success_pose_policy": None, "hole_pose_policy": None,
                "relative_quat_reference_xyzw": None, "base_yaw_rad": 0} if rl else None,
            "pick": {"source_board_xy_m": None, "match_radius_m": None, "ambiguity_margin_m": None,
                "size_range_m": None, "grasp_z_m": None, "hover_height_m": .05, "lift_height_m": .10,
                "quaternion_xyzw": None,
                "approach_waypoints_xyzw": [], "feature_uv": None, "goal_uv": None,
                "current_a": None, "speed_dps": None, "descent_speed_scale": None,
                "lift_speed_scale": None, "held_fraction_range": None, "open_min_fraction": None,
                "servo": {"probe_m": .012, "gain": .65, "max_step_m": .015, "max_radius_m": .06,
                    "tolerance_px": 5, "max_iterations": 8, "speed_scale": .45}},
            "transfer_waypoints_xyzw": [],
            "verification": {"retention": {"mode": "operator"},
                             "placement": {"mode": "operator", "view_pose_xyzw": None}},
        })
    return {"schema_version": 1, "calibrated": False, "base_calibrated": False,
        "working_arm": "right", "tcp_frame": "tip_r",
        "robot_config": "robot.local.json", "board_calibration": "board.local.json",
        "calibration_identity": {"robot_name": None, "base_frame": None, "tcp_frame": "tip_r",
            "robot_config_sha256": None, "board_sha256": None, "checkpoint_sha256": HEXAGON_SHA256,
            "measured_at_utc": None, "grasp_convention": None},
        "policy": {"task": "InsertAnything-HexagonHole-III-Direct-v0", "force_mode": "zeros",
            "checkpoint": "../.pretrained_checkpoints/hexagon_III.pth", "device": "cpu"},
        "supervised": True, "tcp_floor_m": None, "command_timeout_s": 30,
        "motion": vars(VerticalMotion()).copy(), "reset_waypoints_xyzw": [],
        "board_max_translation_m": None, "board_max_rotation_deg": None,
        "camera_clear_waypoints_xyzw": [], "free_speed_scale": None,
        "head_q_rad": None, "head_speed_scale": None, "tasks": tasks}
