from __future__ import annotations
import hashlib
import time
from pathlib import Path
import numpy as np
from . import dependencies
from .policy_runtime.action_postprocessor import ActionPostprocessor, ActionPostprocessorConfig
from .policy_runtime.observation_builder import CalibrationData, build_observation
from .policy_runtime.robot_state_adapter import RobotState
from .geometry import PolicyFrame, constrain_target, angle
from .monitor import InsertionMonitor

HEXAGON_SHA256 = "d11d9a4849238dbdec4602efb86df6d260118fb7ca59e0f159a05546d2b14c08"
OBS_ORDER = ["fingertip_pos_rel_fixed", "fingertip_quat", "fingertip_quat_rel_fixed",
             "ee_linvel", "ee_angvel", "contact_force"]


def verify_checkpoint(path, expected=HEXAGON_SHA256):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    if h.hexdigest() != expected.lower():
        raise ValueError("Hexagon-III checkpoint SHA256 mismatch")


class HexagonActor:
    def __init__(self, path, device="cpu"):
        verify_checkpoint(path)
        import torch
        from .policy_runtime.torch_policy import TorchPolicyConfig, TorchRecurrentActorPolicy
        self.torch, self.device = torch, device
        cfg = TorchPolicyConfig(obs_order=OBS_ORDER, device=device)
        self.model = TorchRecurrentActorPolicy.from_checkpoint(path, cfg)
        self.model.eval()
        self.reset()

    def reset(self):
        self.state = self.model.get_initial_state(batch_size=1, device=self.device)

    def act(self, obs):
        x = self.torch.as_tensor(obs, dtype=self.torch.float32, device=self.device).unsqueeze(0)
        with self.torch.no_grad():
            action, self.state = self.model.act(x, self.state)
        return action.squeeze(0).cpu().numpy()


class RLController:
    def __init__(self, actor):
        self.actor = actor

    def reset(self, task):
        self.actor.reset()
        self.frame = PolicyFrame(task)
        self.post = ActionPostprocessor(ActionPostprocessorConfig(
            pos_action_threshold=(.01,)*3, rot_action_threshold=(.045,)*3,
            raw_action_clip=1.0, yaw_enable=False))
        self.post_state = self.post.get_initial_state()
        self.previous = np.zeros(6)
        identity = np.array([0,0,0,0,0,0,1.], dtype=float)
        raw = {"policy_observation": {
            "obs_order": OBS_ORDER, "append_prev_actions": True, "quat_order": "wxyz",
            "requires_orientation_logic": True, "symmetry_angles_deg": [0.],
            "contact_force_mode": "zeros",
            "relative_quat_reference_quat_xyzw": task.policy_mapping["relative_quat_reference_xyzw"],
        }}
        self.calib = CalibrationData(identity, np.array(task.policy_mapping["hole_pose_policy"], dtype=float),
                                     self.frame.to_policy(task.entry_pose), raw)

    def target(self, sample, task):
        p = self.frame.to_policy(sample.pose)
        state = RobotState(p, np.zeros(6), np.zeros(7), np.zeros(7),
                           self.frame.twist_to_policy(sample.velocity), None)
        obs = build_observation(state, self.calib, prev_actions=self.previous)
        if obs.obs.shape != (26,) or not np.all(np.isfinite(obs.obs)) or np.any(obs.contact_force):
            raise RuntimeError("INVALID_POLICY_OBSERVATION")
        self.last_observation = obs.obs.copy()
        action = np.asarray(self.actor.act(obs.obs), dtype=float)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise RuntimeError("INVALID_POLICY_ACTION")
        out, self.post_state = self.post.process(p, self.calib.T_BH, self.calib.T_BPRE, action, self.post_state)
        self.previous = out.action_ema.copy()
        return constrain_target(sample.pose, self.frame.to_real(out.T_BT_target), task)


class PlannedController:
    def reset(self, task):
        pass

    def target(self, sample, task):
        return constrain_target(sample.pose, task.success_pose, task)


def run_insertion(session, task, controller, *, emit, period_s=8/120,
                  sleep=time.sleep, clock=time.monotonic, check_grip=None):
    sample = session.sample()
    delta = sample.pose[:3]-task.success_pose[:3]
    if not .04 <= delta[2] <= .05 or np.linalg.norm(delta[:2]) > task.criteria.xy_tolerance_m:
        raise RuntimeError("PREPOSE_NOT_REACHED")
    if angle(sample.pose, task.success_pose) > task.criteria.orientation_tolerance_rad:
        raise RuntimeError("PREPOSE_ORIENTATION_MISMATCH")
    if not session.holding:
        raise RuntimeError("GRASP_NOT_VERIFIED")
    # Readback after motion termination and a second low-velocity sample form the handoff.
    sleep(period_s)
    sample = session.sample()
    delta = sample.pose[:3]-task.success_pose[:3]
    if (not .04 <= delta[2] <= .05 or np.linalg.norm(delta[:2]) > task.criteria.xy_tolerance_m
            or angle(sample.pose, task.success_pose) > task.criteria.orientation_tolerance_rad
            or np.linalg.norm(sample.velocity[:3]) > .01 or np.linalg.norm(sample.velocity[3:]) > .05):
        raise RuntimeError("PREPOSE_NOT_SETTLED")
    controller.reset(task)
    monitor = InsertionMonitor(task, clock())
    steps = 0
    while True:
        began = clock()
        session.check()
        if check_grip is not None and not check_grip(task):
            raise RuntimeError("GRASP_LOST")
        status, metrics = monitor.update(sample, began)
        emit("insertion_sample", task=task.task_id, status=status, steps=steps, **metrics)
        if status == "success":
            return {"status": "depth_candidate", "steps": steps, **metrics}
        if status == "running":
            target = controller.target(sample, task)
            session.check()
            if clock()-sample.timestamp > task.criteria.max_sample_gap_s:
                raise RuntimeError("STATE_EXPIRED_DURING_INFERENCE")
            if clock()-monitor.started >= task.criteria.timeout_s:
                raise RuntimeError("INSERT_TIMEOUT")
            session.move(target, task.speed_scale)
            steps += 1
        # In the dwell window issue NO further motion command.
        session.check()  # unconditional, including overruns
        remaining = period_s-(clock()-began)
        if remaining > 0:
            sleep(remaining)
        session.check()
        sample = session.sample()
