#!/usr/bin/env python3
"""Use pinned calibration tools with our LOCAL robot config, not upstream example coordinates."""
from __future__ import annotations
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
from roco_vega import dependencies
from roco_vega.task_spec import pose
from roco_vega.preflight import speed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, choices=("snapshot", "tcp-record", "tcp-analyze", "board"))
    parser.add_argument("--config", type=Path, required=True, help="roco config; may still be an incomplete template")
    parser.add_argument("--output", type=Path, required=True, help="new output file/directory")
    args, tool_args = parser.parse_known_args(argv)
    if tool_args[:1] == ["--"]:
        tool_args = tool_args[1:]
    if args.output.exists():
        parser.error("Output already exists; use a new filename")
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    from steadyhand.config import read_json
    robot = read_json((args.config.resolve().parent/cfg["robot_config"]).resolve())
    if os.environ.get("ROBOT_NAME") not in (None, robot["robot_name"]):
        parser.error("ROBOT_NAME disagrees with robot config")
    if args.tool != "tcp-analyze":
        os.environ["ROBOT_NAME"] = robot["robot_name"]
    names = {"snapshot": "vega_capture_snapshot", "tcp-record": "vega_tool_frame_record",
             "tcp-analyze": "vega_tool_frame_calibration", "board": "vega_board_five_point_calibrate"}
    module = importlib.import_module("tools."+names[args.tool])
    # Pinned tools bind load_bundle at import. Replace the loader in this
    # invocation only, without editing vendored or local JSON files.
    if hasattr(module, "load_bundle"):
        original = module.load_bundle
        def local_bundle(name):
            bundle = original(name)
            bundle["robot"] = copy.deepcopy(robot)
            return bundle
        module.load_bundle = local_bundle
    output_flag = "--json-output" if args.tool == "tcp-analyze" else "--output"
    forwarded = [*tool_args, output_flag, str(args.output.resolve())]
    if args.tool == "tcp-analyze" and "--urdf" not in tool_args:
        urdf = Path(robot["urdf_path"])
        if not urdf.is_absolute():
            urdf = dependencies.STEADYHAND/urdf
        forwarded.extend(["--urdf", str(urdf.resolve())])
    if args.tool == "board":
        # Override the upstream auto-generated camera-clear poses with our
        # explicitly taught route. RIGHT_READY still comes from local robot JSON.
        import math
        floor = float(cfg["tcp_floor_m"])
        if not math.isfinite(floor):
            raise ValueError("Complete tcp_floor_m first")
        waypoints = [pose(v, "camera clear waypoint") for v in cfg["camera_clear_waypoints_xyzw"]]
        if not waypoints or any(v[2] < floor for v in waypoints):
            raise ValueError("Teach camera-clear waypoints above the TCP floor first")
        free_speed = speed(cfg["free_speed_scale"], "free speed")
        from roco_vega.session import to_pose
        from steadyhand.executor import move_tcp_segmented
        def camera_clear(adapter, **kwargs):
            for point in waypoints:
                move_tcp_segmented(adapter, to_pose(point), speed_scale=free_speed,
                    max_translation_step_m=.05, max_orientation_step_rad=.15, min_tcp_z_m=floor)
        module.move_camera_clear_for_image = camera_clear
        skills = module.load_vega_skills()
        skills["safety"]["min_tcp_z_m"] = floor
        module.load_vega_skills = lambda: copy.deepcopy(skills)
        # Stop on first failure; do not re-enter motion automatically.
        return module._main_once(forwarded)
    return module.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
