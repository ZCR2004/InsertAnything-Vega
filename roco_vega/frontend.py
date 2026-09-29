from __future__ import annotations
import numpy as np
from . import dependencies
from steadyhand.vision.scene import detect_head_task_scene
from steadyhand.vision.wrist_servo import run_xy_servo
from steadyhand.board_calibration import (load_board_calibration, board_geometry_signature,
                                          compare_board_geometry)
from steadyhand.board_geometry import BOARD_SIZE_M, configured_board_plane_z
from .session import from_pose
from .geometry import angle
from .motion import pick_heights


def select_part(parts, expected_xy, radius, ambiguity_margin, size_range):
    """Known source-layout association, not a general semantic recognizer.

    Exactly one plausible dark component must match the taught region and size.
    An ambiguous merge/extra object is rejected rather than assigned by count.
    """
    ranked = []
    for p in parts:
        center = np.asarray(p["center_board_m"][:2], dtype=float)
        box = np.asarray(p["box_board_px"], dtype=float)
        size = (box[2:]-box[:2]) * BOARD_SIZE_M/799
        if np.any(size < np.asarray(size_range[0])) or np.any(size > np.asarray(size_range[1])):
            continue
        distance = float(np.linalg.norm(center-np.asarray(expected_xy)))
        if distance <= radius:
            ranked.append((distance, p))
    ranked.sort(key=lambda v:v[0])
    if not ranked:
        raise RuntimeError("NOT_FOUND")
    if len(ranked)>1 and ranked[1][0]-ranked[0][0] < ambiguity_margin:
        raise RuntimeError("WRONG_OR_AMBIGUOUS_PART")
    return ranked[0][1]


class SteadyHandFrontend:
    def __init__(self, session, cfg, robot_cfg, journal):
        self.s, self.cfg, self.robot_cfg, self.j = session, cfg, robot_cfg, journal
        self.board = load_board_calibration(cfg["board_calibration"], robot_cfg)
        if self.board.get("schema_version") != 2 or not self.board.get("camera_geometry_signature"):
            raise ValueError("Completed five-point calibration with camera geometry is required")
        self.last_head_stamp = None
        self.last_wrist_stamp = None

    def _scene(self):
        frame, head_q = self.s.adapter.capture_head()
        if frame.left_timestamp_ns is None or frame.left_timestamp_ns == self.last_head_stamp:
            raise RuntimeError("STALE_HEAD_IMAGE")
        self.last_head_stamp = frame.left_timestamp_ns
        fixed = self.robot_cfg["kinematics"]["fixed_joint_values"]
        scene = detect_head_task_scene(frame.left_rgb, frame.camera_info, head_q,
            plane_z_m=configured_board_plane_z(self.robot_cfg, self.s.floor_m),
            lift_m=fixed["Lift"], torso_flip_rad=fixed["torso_flip"], layout="unlabeled")
        signature = board_geometry_signature(scene["board"]["corners_px"])
        if signature is None:
            raise RuntimeError("BOARD_GEOMETRY_UNAVAILABLE")
        check = compare_board_geometry(self.board["camera_geometry_signature"], signature,
                                       max_angle_change_deg=self.cfg["board_max_rotation_deg"])
        if not check["valid"]:
            raise RuntimeError("BOARD_RECALIBRATION_REQUIRED")
        self.j.emit("scene", scene=scene, geometry_check=check)
        return scene

    def _wrist(self):
        frame = self.s.adapter.capture_wrist()
        # Arrival time alone cannot make a repeated sensor frame fresh.
        stamp = (frame.frame_id, frame.timestamp_ns)
        if all(v is None for v in stamp) or stamp == self.last_wrist_stamp:
            raise RuntimeError("STALE_WRIST_IMAGE")
        self.last_wrist_stamp = stamp
        rgb = np.asarray(frame.rgb)
        if rgb.ndim != 3 or rgb.dtype != np.uint8 or np.std(rgb) < 2:
            raise RuntimeError("INVALID_WRIST_IMAGE")
        return rgb

    def _retake_board(self):
        for waypoint in self.cfg["camera_clear_waypoints_xyzw"]:
            self.s.move(waypoint, self.cfg["free_speed_scale"])
        self.s.command(self.s.adapter.move_head, self.cfg["head_q_rad"], self.cfg["head_speed_scale"])
        scene = self._scene()
        # Follow the pinned fixed-plane / XY-translation model. Do not silently
        # rotate A while leaving its calibrated yaw/grasp convention unchanged.
        raw_center = np.asarray(scene["board"]["T_base_board_center"])[:2, 3]
        baseline = np.asarray(self.board["raw"]["camera_board_read"]["T_base_board_center"], dtype=float)[:2, 3]
        # The hand-corrected board center is authoritative. Camera translation
        # is a DELTA from its own calibration capture, not from a stale example
        # camera correction stored in the upstream robot configuration.
        delta = raw_center-baseline
        self.center = np.asarray(self.board["center_base_xy_m"])+delta
        if np.linalg.norm(delta) > self.cfg["board_max_translation_m"]:
            raise RuntimeError("BOARD_TRANSLATION_OUT_OF_RANGE")
        self.j.emit("board_retaken", board_delta_xy=delta)
        return scene, delta

    def prepare_approach(self, task):
        _, delta = self._retake_board()
        return task.translated(delta)

    def prepare(self, task):
        scene, delta = self._retake_board()
        part = select_part(scene["parts"], task.pick["source_board_xy_m"], task.pick["match_radius_m"],
                           task.pick["ambiguity_margin_m"], task.pick["size_range_m"])
        axes = np.column_stack([self.board["board_x_unit_base_xy"], self.board["board_y_unit_base_xy"]])
        self.pick_xy = self.center + axes @ np.asarray(part["center_board_m"][:2])
        self.j.emit("part_associated", task=task.task_id, method="taught_region_and_size",
                    board_delta_xy=delta, pick_xy=self.pick_xy)
        return task.translated(delta)

    def pick(self, task):
        p = task.pick
        grasp_z, hover_z, lift_z = pick_heights(p)
        hover = np.r_[self.pick_xy, hover_z, p["quaternion_xyzw"]]
        for waypoint in p["approach_waypoints_xyzw"]:
            self.s.move(waypoint, self.cfg["free_speed_scale"])
        # Approach XY above the object before any vertical descent.
        current = self.s.sample().pose
        high = current.copy()
        high[2] = max(current[2], hover_z)
        if high[2] > current[2]+1e-9:
            self.s.move(high, self.cfg["free_speed_scale"])
        above = hover.copy()
        above[2] = high[2]
        self.s.move(above, self.cfg["free_speed_scale"])
        self.s.move(hover, self.cfg["free_speed_scale"])
        self.j.emit("pick_hover_reached", task=task.task_id, hover_height_m=p.get("hover_height_m", .05))
        self.s.open_gripper(task.task_id)
        if not self.opened(task):
            raise RuntimeError("GRIPPER_NOT_OPEN")
        result = run_xy_servo(self.s, self._wrist, floor_m=self.s.floor_m,
            feature_uv=p["feature_uv"], goal_uv=p["goal_uv"], **p["servo"],
            event=lambda event, fields:self.j.emit("wrist_"+event, task=task.task_id, **fields))
        if not result or result.get("status") != "converged":
            raise RuntimeError("WRIST_ALIGNMENT_FAILED")
        aligned = from_pose(self.s.get_tcp_pose())
        grasp = aligned.copy()
        grasp[2] = grasp_z
        self.pick_hover_pose, self.pick_grasp_pose = aligned.copy(), grasp.copy()
        self.s.move(grasp, p["descent_speed_scale"])
        self.s.adapter.configure_grip(p["current_a"], p["speed_dps"])
        self.s.grip(task)
        lift = aligned.copy()
        lift[2] = lift_z
        self.pick_lift_pose = lift.copy()
        self.s.move(lift, p["lift_speed_scale"])
        self.j.emit("pick_lifted", task=task.task_id, lift_height_m=p.get("lift_height_m", .10))
        if not self.retained(task):
            raise RuntimeError("GRASP_LOST")
        if not self._verify("retention", task):
            raise RuntimeError("LIFT_RETENTION_UNVERIFIED")

    def return_pick(self, task):
        """Only for the isolated pick test, before any transfer/insertion."""
        if not self.retained(task):
            raise RuntimeError("GRASP_LOST")
        current = self.s.sample().pose
        if (np.linalg.norm(current[:3]-self.pick_lift_pose[:3]) > .004
                or angle(current, self.pick_lift_pose) > .03):
            raise RuntimeError("PICK_TEST_RETURN_REQUIRES_ORIGINAL_HOVER")
        self.s.move(self.pick_grasp_pose, task.pick["descent_speed_scale"])
        self.s.open_gripper(task.task_id)
        if not self.opened(task):
            raise RuntimeError("RELEASE_FAILED")
        self.s.move(self.pick_lift_pose, task.pick["lift_speed_scale"])

    def retained(self, task):
        status = self.s.adapter.gripper_status()
        position = status.get("position")
        low, high = task.pick["held_fraction_range"]
        return self.s.holding and position is not None and np.isfinite(position) and low <= position <= high

    def opened(self, task):
        value = self.s.adapter.gripper_position()
        return value is not None and np.isfinite(value) and value >= task.pick["open_min_fraction"]

    def _verify(self, stage, task):
        v = task.verify[stage]
        if v["mode"] == "operator":
            if not self.cfg.get("supervised", False):
                raise RuntimeError("OPERATOR_VERIFICATION_REQUIRES_SUPERVISED_MODE")
            self.j.emit("operator_verification_requested", task=task.task_id, stage=stage)
            return input(f"{task.task_id}: verify {stage}; type yes: ").strip().lower() == "yes"
        # Repeatable wrist pose + taught image template + an explicit region.
        # This is an appearance verifier, not a claim of electrical connectivity.
        import cv2
        from pathlib import Path
        template = cv2.imread(str(Path(self.cfg["config_dir"])/v["template"]), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(self._wrist(), cv2.COLOR_RGB2GRAY)
        x0,y0,x1,y1 = map(int, v["roi_xyxy"])
        if template is None or not 0 <= x0 < x1 <= image.shape[1] or not 0 <= y0 < y1 <= image.shape[0]:
            raise RuntimeError("INVALID_VERIFICATION_TEMPLATE_OR_ROI")
        roi = image[y0:y1,x0:x1]
        if np.std(template) < 5 or any(t>r for t,r in zip(template.shape,roi.shape)):
            raise RuntimeError("UNUSABLE_VERIFICATION_TEMPLATE")
        score = float(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED).max())
        self.j.emit("appearance_verification", task=task.task_id, stage=stage, score=score)
        return np.isfinite(score) and score >= v["min_score"]

    def transfer(self, task):
        current = self.s.sample().pose
        # Clear the source vertically, then traverse the taught high route.
        safe_z = max(current[2], task.entry_pose[2], *(p[2] for p in task.transfer_waypoints))
        if any(p[2] < max(current[2], task.entry_pose[2])-self.s.motion.position_tolerance_m
               for p in task.transfer_waypoints):
            raise RuntimeError("TRANSFER_ROUTE_BELOW_LIFT_OR_ENTRY")
        high = current.copy()
        high[2] = safe_z
        self.s.move(high, self.cfg["free_speed_scale"])
        for waypoint in task.transfer_waypoints:
            self.s.move(waypoint, self.cfg["free_speed_scale"])
        above = task.entry_pose.copy()
        above[2] = max(self.s.sample().pose[2], task.entry_pose[2])
        self.s.move(above, self.cfg["free_speed_scale"])
        self.s.move(task.entry_pose, task.speed_scale)

    def retreat(self, task):
        current = self.s.sample().pose
        target = current.copy()
        target[2] += task.retreat_height_m
        self.s.move(target, task.speed_scale)
        self.j.emit("post_release_lift", task=task.task_id, height_m=task.retreat_height_m,
                    start_xyzw=current, target_xyzw=target)

    def placed(self, task):
        # Operator verifies at the 15 cm retreat; no unnecessary downward return.
        if task.verify["placement"]["mode"] == "template":
            self.s.move(task.verify["placement"]["view_pose_xyzw"], task.speed_scale)
        return self._verify("placement", task)

    def finish_task(self, task):
        # Empty means stay at the high retreat/view, then take the next camera route.
        for waypoint in self.cfg.get("reset_waypoints_xyzw", []):
            self.s.move(waypoint, self.cfg["free_speed_scale"])
        self.j.emit("ready_for_next_task", task=task.task_id,
                    reset_used=bool(self.cfg.get("reset_waypoints_xyzw")))

    def verify_already_held(self, task):
        if not self.retained(task) or not self._verify("retention", task):
            raise RuntimeError("EXISTING_GRASP_UNVERIFIED")
