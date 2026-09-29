#!/usr/bin/env python3
"""No robot import, construction or connection in template/check/simulate modes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("template", "check", "simulate", "live"), default="check")
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, help="new calibration template filename")
    p.add_argument("--log-dir", type=Path, help="new directory, never appended to an older run")
    from roco_vega.task_spec import ORDER
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--task", choices=ORDER, help="single task; default runs nine, batteries first")
    selection.add_argument("--batteries-only", action="store_true", help="complete both battery pick/insert cycles; no RL")
    p.add_argument("--confirm-empty-gripper", action="store_true", help="live connection homes gripper")
    p.add_argument("--confirm-calibrated-paths", action="store_true", help="waypoints were checked on this robot")
    args = p.parse_args(argv)
    if args.mode == "template":
        if not args.output:
            p.error("template requires --output")
        from roco_vega.template import calibration_template
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(calibration_template(), f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Created uncalibrated template: {args.output}")
        return 0

    if args.mode == "simulate":
        from roco_vega.orchestrator import Journal
        from roco_vega.simulation import simulate
        print(json.dumps(simulate(Journal(log_folder(args)), task_id=args.task,
                                  batteries_only=args.batteries_only), indent=2))
        return 0
    if not args.config:
        p.error("check/live require --config")
    from roco_vega.task_spec import load_config
    from roco_vega.preflight import check_preflight
    cfg, tasks = load_config(args.config, task_id=args.task, batteries_only=args.batteries_only)
    robot_cfg = check_preflight(cfg, tasks)
    actor = prepare_actor(cfg, tasks)
    if args.mode == "check":
        print("Selected task calibration, checkpoint (when RL is used) and forward checks passed. No SDK connection.")
        return 0
    if not args.confirm_empty_gripper or not args.confirm_calibrated_paths:
        p.error("live requires --confirm-empty-gripper and --confirm-calibrated-paths")
    if cfg.get("supervised") and not sys.stdin.isatty():
        p.error("supervised verification needs an interactive terminal")
    from roco_vega.hardware import FreshVegaAdapter
    from roco_vega.session import RobotSession
    from roco_vega.frontend import SteadyHandFrontend
    from roco_vega.orchestrator import Orchestrator, Journal
    adapter = FreshVegaAdapter(robot_cfg)
    adapter.prepare()  # URDF/IK setup only, before hardware connection
    session = RobotSession(adapter, floor_m=cfg["tcp_floor_m"], command_timeout_s=cfg["command_timeout_s"],
                           motion=cfg.get("motion"))
    journal = Journal(log_folder(args))
    journal.emit("configuration", config=cfg, selected_tasks=[t.task_id for t in tasks])
    frontend = SteadyHandFrontend(session, cfg, robot_cfg, journal)
    def cancel(signum, frame):
        session.abort()
        raise KeyboardInterrupt(f"Signal {signum}")
    for name in ("SIGINT", "SIGTERM"):
        signal.signal(getattr(signal, name), cancel)
    Orchestrator(session, frontend, actor, journal).run(tasks)
    return 0


def prepare_actor(cfg, tasks):
    """No Torch/checkpoint required for tests which only run planned motion."""
    if not any(t.strategy == "rl" for t in tasks):
        return None
    from roco_vega.insertion import HexagonActor, RLController
    from roco_vega.monitor import Sample
    import numpy as np
    actor = HexagonActor(cfg["policy"]["checkpoint"], cfg["policy"].get("device", "cpu"))
    for task in tasks:
        if task.strategy == "rl":
            ctrl = RLController(actor)
            ctrl.reset(task)
            ctrl.target(Sample(task.entry_pose, 0., 1, np.zeros(6)), task)
    return actor


def log_folder(args):
    return args.log_dir or Path("roco_runs")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
