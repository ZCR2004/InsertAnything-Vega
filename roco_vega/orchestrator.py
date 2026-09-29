from __future__ import annotations
import json
import os
from pathlib import Path
import time
from .insertion import RLController, PlannedController, run_insertion


class Journal:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.completed = []

    def emit(self, event, **fields):
        row = {"event": event, "monotonic_s": time.monotonic(), **fields}
        with (self.folder/"events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=lambda x:x.tolist())+"\n")
        print(json.dumps(row, ensure_ascii=False, default=lambda x:x.tolist()), flush=True)

    def complete(self, task_id, result):
        completed = [*self.completed, task_id]
        temp = self.folder/"progress.tmp"
        with temp.open("w", encoding="utf-8") as stream:
            json.dump({"completed": completed, "last_result": result}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.folder/"progress.json")
        self.completed = completed


class Orchestrator:
    def __init__(self, session, frontend, actor, journal, *, clock=time.monotonic, sleep=time.sleep):
        self.session, self.frontend, self.journal = session, frontend, journal
        self.controllers = {"rl": RLController(actor), "planned": PlannedController()}
        self.clock, self.sleep = clock, sleep

    def run(self, tasks, *, already_held=False):
        s, f, j = self.session, self.frontend, self.journal
        if already_held and len(tasks) != 1:
            raise ValueError("already-held entry is single-task only")
        if already_held and not (s.connected and s.holding):
            raise ValueError("already-held requires an existing connected, verified session; no reconnect/home")
        try:
            s.connect()
            for original in tasks:
                s.check()
                j.emit("task_started", task=original.task_id, strategy=original.strategy)
                # A held-object single test uses the already-calibrated A and
                # must not move back to the camera-clear/source-recognition pose.
                task = original if already_held else f.prepare(original)
                if already_held:
                    # The frontend must verify a pre-existing shared calibrated session;
                    # reconnecting with an object and silently skipping home is forbidden.
                    f.verify_already_held(task)
                else:
                    f.pick(task)
                j.emit("grasp_verified", task=task.task_id)
                f.transfer(task)
                result = run_insertion(s, task, self.controllers[task.strategy], emit=j.emit,
                                       clock=self.clock, sleep=self.sleep, check_grip=f.retained)
                s.check()
                j.emit("release_started", task=task.task_id)
                s.open_gripper(task.task_id)
                if not f.opened(task):
                    raise RuntimeError("RELEASE_FAILED")
                j.emit("release_verified", task=task.task_id)
                f.retreat(task)
                j.emit("retreated", task=task.task_id)
                if not f.placed(task):
                    raise RuntimeError("PLACEMENT_UNVERIFIED")
                f.finish_task(task)
                result["status"] = "completed"
                j.complete(task.task_id, result)
                j.emit("task_completed", task=task.task_id, result=result)
            return list(j.completed)
        except BaseException as exc:
            try:
                s.abort()
            finally:
                j.emit("sequence_stopped", error=type(exc).__name__, reason=str(exc), completed=j.completed)
            raise
        finally:
            # No automatic release or retreat on error.
            s.close()
