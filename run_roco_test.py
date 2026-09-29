#!/usr/bin/env python3
"""Separate plan/pick/manual-load insertion tests. Defaults to no-motion checks."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import signal
import sys
from pathlib import Path
from run_roco_vega import log_folder, prepare_actor


def main(argv=None):
    from roco_vega.task_spec import ORDER, load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, choices=("plan", "pick", "insert"))
    parser.add_argument("--task", required=True, choices=ORDER)
    parser.add_argument("--mode", choices=("check", "simulate", "live"), default="check")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--confirm-empty-gripper", action="store_true")
    parser.add_argument("--confirm-calibrated-paths", action="store_true")
    args = parser.parse_args(argv)
    from roco_vega.orchestrator import Journal
    from roco_vega.commissioning import run_test
    from roco_vega.session import RobotSession
    if args.mode == "simulate":
        from roco_vega.simulation import FakeAdapter, FakeClock, FakeFrontend, FakeActor, synthetic_tasks
        clock, adapter, actor = FakeClock(), FakeAdapter(), FakeActor()
        session = RobotSession(adapter, floor_m=0, clock=clock)
        task = next(t for t in synthetic_tasks() if t.task_id == args.task)
        task = replace(task, manual_load_waypoints=(task.entry_pose,),
                       pick={**task.pick, "speed_dps": 60})
        run_test(args.test, session, FakeFrontend(session), task, actor, Journal(log_folder(args)),
                 prompt=lambda _: {"plan": "move", "pick": "return", "insert": "grip"}[args.test],
                 clock=clock, sleep=clock.sleep)
        print(json.dumps({"mode": "fake_hardware", "test": args.test, "actor_resets": actor.resets,
                          "connects": adapter.calls.count("connect"), "homes": adapter.calls.count("home")}))
        return 0
    if args.config is None:
        parser.error("--config is required")
    from roco_vega.preflight import check_preflight
    cfg, tasks = load_config(args.config, task_id=args.task, stage=args.test)
    robot_cfg = check_preflight(cfg, tasks, stage=args.test)
    actor = prepare_actor(cfg, tasks) if args.test == "insert" else None
    if args.mode == "check":
        print(f"{args.test}/{args.task}: selected-stage checks passed. No hardware connection.")
        return 0
    if not args.confirm_empty_gripper or not args.confirm_calibrated_paths:
        parser.error("live requires --confirm-empty-gripper and --confirm-calibrated-paths")
    if not sys.stdin.isatty():
        parser.error("These supervised stage tests require an interactive terminal")
    from roco_vega.hardware import FreshVegaAdapter
    from roco_vega.frontend import SteadyHandFrontend
    adapter = FreshVegaAdapter(robot_cfg)
    adapter.prepare()
    session = RobotSession(adapter, floor_m=cfg["tcp_floor_m"], command_timeout_s=cfg["command_timeout_s"],
                           motion=cfg.get("motion"))
    journal = Journal(log_folder(args))
    journal.emit("commissioning_configuration", stage=args.test, task=args.task, config=cfg)
    frontend = SteadyHandFrontend(session, cfg, robot_cfg, journal)
    def cancel(signum, frame):
        session.abort()
        raise KeyboardInterrupt(f"Signal {signum}")
    for name in ("SIGINT", "SIGTERM"):
        signal.signal(getattr(signal, name), cancel)
    run_test(args.test, session, frontend, tasks[0], actor, journal)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
