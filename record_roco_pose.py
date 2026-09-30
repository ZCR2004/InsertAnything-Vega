#!/usr/bin/env python3
"""Record measured tip_r without Robot(), homing, motion or gripper commands."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from roco_vega import dependencies
from roco_vega.session import from_pose
from roco_vega.geometry import angle
from roco_vega.preflight import digest


def capture_record(reader, cfg, robot_file, label, *, sleep=time.sleep):
    samples = []
    for _ in range(5):
        samples.append(reader.capture())
        sleep(.05)
    poses = [from_pose(v["pose"]) for v in samples]
    if any(np.linalg.norm(p[:3]-poses[0][:3]) > .0005 or angle(p, poses[0]) > .005 for p in poses[1:]):
        raise RuntimeError("TCP_MOVING: hold the pose steady before recording")
    stamps = [v["joint_timestamp_ns"] for v in samples]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise RuntimeError("STALE_JOINT_STATE")
    return {"schema_version": 1, "label": label, "mode": "READ_ONLY_NO_MOTION_COMMANDS",
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "robot_name": cfg["robot_name"], "base_frame": cfg["kinematics"]["base_frame"],
            "tcp_frame": cfg["kinematics"]["ee_frame"], "robot_config_sha256": digest(robot_file),
            "pose_xyzw": poses[-1].tolist(), "joint_positions_rad": list(samples[-1]["joint_positions_rad"]),
            "joint_timestamp_ns": stamps[-1], "stable_samples": len(samples),
            "note": "Measured TCP only. Operator must verify held-object relationship and successful insertion."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", required=True, type=Path)
    parser.add_argument("--label", required=True, help="e.g. usb_a_success_A or usb_a_load")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; use a new filename")
    from steadyhand.config import read_json
    from tools.vega_tool_frame_record import ReadOnlyVegaTipReader
    cfg = read_json(args.robot_config)
    with ReadOnlyVegaTipReader(cfg) as reader:
        record = capture_record(reader, cfg, args.robot_config, args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
