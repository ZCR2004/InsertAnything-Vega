# RoCo Vega：SteadyHand 抓取 + InsertAnything 插入

实现分支：`feat/roco-vega-rl-sequence`。本入口面向右臂 `tip_r`，固定 SteadyHand `5efaa61368d6ad6777115c154b1fbe81e6db47b9`。采用官方比赛改装 Vega 的关节位置控制和 CAN 二指夹爪；不使用 Franka 控制服务。来源见 [`third_party/versions.json`](../third_party/versions.json)。

首次上机请按 **[现场操作顺序](现场操作顺序.md)** 操作。其中包含相机/TCP/板面标定、只读记录 A、空爪路径、夹取、手动装夹后单任务插入的独立命令。

## 已实现的流程

```text
板定位 / 零件区域关联 → 腕部视觉 XY 对齐 → 限流夹取 → 提起并验证仍持有
→ 已标定转运路点 → A 正上方 45 mm → 静止读回检查
→ RL（7 类）或小步规划下插（2 类电池）
→ 深度 + 横向 + 姿态容差持续满足 → 停止下压 → 松爪并确认
→ 垂直退离 → 放置验证 → 写入 progress.json → 下一任务
```

九任务顺序为 `gear_60teeth, gear_20teeth, rod_16mm, bolt_8mm, usb_a, hdmi, pin, battery_size1, battery_size5`。RL 使用同一份六边形 **Hexagon-III** checkpoint，每项任务重新清空 LSTM、动作 EMA、上一动作和成功判定状态。两种电池不调用 actor。

一个 `RobotSession` 持有一份机器人、相机和夹爪连接。夹爪仅在空爪启动时标定一次，抓住物体后不会重新连接或 home。超时、取消、状态陈旧、丢失抓取、超深、验证失败都会停止整个序列，不自动重试、不盲退、不自动张爪。Ctrl+C 停止活动命令并请求软件急停；日志不能替代现场确认。

## 环境和离线运行

从仓库根目录执行。需要 Python 3.10+；纯推理不需要安装整个 Isaac Lab 扩展。

```bash
python -m pip install -r requirements-roco.txt
python -m unittest -v test_roco_vega
python run_roco_vega.py --mode simulate
python run_roco_vega.py --mode template --output configs/roco.local.json
```

`simulate` 使用假机器人和解析动作器，经过实际的 26 维观测、动作后处理、深度监控与调度代码，输出到 `roco_runs/`。它验证接口与流程，**不是物理仿真，也不代表 checkpoint 能插入这些物体**。模板已另存为 [`configs/roco_vega.template.json`](../configs/roco_vega.template.json)，其中未知坐标为 `null`，默认拒绝真机启动。

真机需现场环境中的 `dexcontrol==0.5.0`、Pinocchio、`wrist_cameras`、CAN 驱动和机器人服务器 motion plugin。不要仅为兼容此脚本替换已调通的现场驱动；先在同一个 Python 解释器中确认这些依赖与 Torch 可以共存。

## 如何填写标定

1. 在 `configs/robot.local.json` 保存经过核对的 SteadyHand 机器人配置。可以参考 `third_party/steadyhand/configs/robots/vega.json` 的结构，但不能直接采用其他现场的机械臂位置、相机修正或夹爪电流。相对 URDF/driver 路径沿用 SteadyHand 的根目录解析方式，现场路径可改成绝对路径。
2. 用固定版本 SteadyHand 的五点标定工具得到 `configs/board.local.json`（schema 2，包含板轴、表面和相机角点）。该配置自带 12 小时时效检查。固定板面/头部姿态，只对允许范围内的板 XY 平移补偿；板旋转或高度变化需要重新标定。上游 fallback 文件是别人的测量记录，不是本机标定。
3. 每种物体保持一致夹持深度与朝向、末端垂直向下，记录**已成功插入时**的真实 `tip_r` 位姿 A 到 `success_pose_xyzw`，单位 m，顺序 `[x,y,z,qx,qy,qz,qw]`。这里既不能填孔中心，也不能用空爪 TCP 代替夹持状态 A。Vega 修正后 `tip_r` 的垂直约定与 Franka 局部工具轴不同，不能照抄 Franka 的 roll=π。
4. 分别标定抓取 hover/下降高度、图像 feature/goal 像素、物体区域与尺寸范围、夹持宽度、电流/速度、转运路点、验证视角和容差。`entry_height_m` 默认 0.045，必须在 0.04～0.05 内，切 RL 时再次检查实测位置。`step_m` 是每轴单次命令位移上限（默认 1 mm），不是力限制。转运采用已核查的路点加 IK/SDK 轨迹，不含全局碰撞规划。
5. 填 `calibration_identity` 的机器人名、base/TCP、时间、夹持约定以及机器人配置、板标定文件 SHA256；绑定实际 checkpoint SHA256。文件变化后应重核标定再更新哈希，而不是用哈希代替标定。Linux 可用 `sha256sum`，PowerShell 可用 `Get-FileHash -Algorithm SHA256`。
6. 确认所有任务和 policy mapping 后再置各项 `calibrated/validated=true`。`head_q_rad` 是 **3 维**，应与五点标定时相同。配置与测试路径必须核对整段机械臂运动；TCP 不低于 floor 不等于整臂无碰撞。

当前首次运行默认 `supervised=true`：提起后和放置后分别人工输入 `yes`。无人工模式可将两处 verifier 改成 `template`，配置 `template` 图片相对路径、`roi_xyxy` 和 `min_score`，在固定视角验证外观。模板示例结构：

```json
{"mode":"template","template":"verification_images/usb_placed.png",
 "roi_xyxy":[800,500,1100,900],"min_score":0.9,
 "view_pose_xyzw":["替换为7个实测数值"]}
```

以上像素只是字段示意，必须实测。放置验证的视角随板 XY 平移更新；抓取接近路点与 camera-clear 路点是固定 base 路点。外观验证不能证明 USB 电气连接或精确啮合。深度使用 TCP 相对 A 的垂直高度，必须与同一夹持关系配套，否则会出现物体滑脱但 TCP 到位的误判。

## 六边形策略的坐标与力输入

模型观测维度仍为 **26**：相对位置 3 + 绝对四元数 4 + 相对四元数 4 + 线速度 3 + 角速度 3 + 原始力 3 + 上一动作 6。三维原始力置零，之后仍执行 checkpoint 自带的归一化；不能删掉这三维，也不能把归一化后的力强制改零。Hexagon-III 保留原四元数参考、`symmetry_angles_deg=[0]`，没有套用其他六边形版本的对称设置。

每种物体的 `policy_mapping` 明确记录虚拟策略空间中的成功 TCP 位姿 `success_pose_policy`、孔位姿 `hole_pose_policy`、相对四元数参考和 base yaw。映射将真实 A 对应到策略的成功状态，并保持真实竖直方向；局部工具轴差用右乘四元数处理。**策略成功时的相对位置不一定是零**；必须从对应 checkpoint 的训练定义/标定获取，代码未硬编码一个通用孔深。`simulation.py` 中的 7.392 mm 仅为非零偏移回归用例。

第一版保持已标定的完整姿态（含 yaw），执行 actor 的平移动作，保留完整动作历史输入。不能宣称这是未经修改的仿真动作执行器：零力、姿态锁定、步长限制和 Vega 阻塞式位置控制都改变了部署条件。默认周期目标约 15 Hz，实际频率受 IK、SDK 轨迹完成和推理影响；过期观测不继续发动作，需要现场测时。

权重使用仓库公开的 Hexagon-III 单孔 checkpoint，校验值：

```text
d11d9a4849238dbdec4602efb86df6d260118fb7ca59e0f159a05546d2b14c08
```

请从项目原有 checkpoint 下载说明获取该文件，填入路径。该权重不是在 RoCo 九任务或零力条件下重新训练过的模型；跨物体能力要逐任务验证。

## 真机入口

先执行只读检查，校验配置、文件身份、checkpoint 和真实 actor 前向；此模式不连接 SDK。不指定 task 时校验全部九项，指定 `--task usb_a` 时只要求公共标定和该任务完成：

```bash
python run_roco_vega.py --mode check --config configs/roco.local.json
```

空爪、现场路点已核对后，先测试一项：

```bash
python run_roco_vega.py --mode live --config configs/roco.local.json \
  --task usb_a --confirm-empty-gripper --confirm-calibrated-paths
```

去掉 `--task usb_a` 且总 `calibrated=true` 才执行九任务。SDK `Robot()` 本身会引起头部回零，仍需在本机机器人配置中核实并设置 `allow_robot_init_head_motion=true`；不会偷偷覆盖这个上游开关。CLI 不提供“拿着物体重新连接后跳过 home”的选项。独立 `run_roco_test.py --test insert` 从空爪启动，在同一会话里提示手动装夹，然后进入该任务的插入流程。

失败后保留 `events.jsonl` 与原子更新的 `progress.json`。重启不会默默跳过旧任务，必须核实场景后明确选择单任务或重排执行。本轮调度不调用旧的 `run_multi_episode_closed_loop.py`；旧脚本仍是原单孔实验入口，不用于比赛串行任务。

## 模块与验证范围

| 模块 | 作用 |
|---|---|
| `frontend.py` | SteadyHand 板识别、已教区域关联、腕部 XY servo、抓取和放置验证 |
| `session.py` / `hardware.py` | 共享会话、有限等待、取消和新鲜状态读回 |
| `geometry.py` / `insertion.py` | Vega↔策略坐标、26D 零力观测、LSTM、RL/规划插入 |
| `monitor.py` / `orchestrator.py` | 深度驻留、失败分支、释放退离、九任务调度和日志 |
| `task_spec.py` / `preflight.py` | 标定合同、身份绑定、完整运行前检查 |
| `test_roco_vega.py` | 假硬件全流程与故障回归 |
| `../run_roco_test.py` / `commissioning.py` | 独立 plan、pick、insert 阶段测试 |
| `../record_roco_pose.py` | 无运动的实测 TCP 位姿记录 |
| `../run_roco_calibration.py` | 使用本机配置调用固定版 TCP/相机/板面标定工具 |

识别接入的是固定版本 SteadyHand 的几何/图像处理；零件通过已教区域和尺寸匹配，并拒绝歧义。它不是任意散乱、任意遮挡场景下的语义识别器。真机接触效果、零力策略成功率、视觉模板阈值与周期稳定性仍需现场验证。

2026-09-29 本地验证：Python 3.10 下 28 项新测试、5 项原 Vega backend 测试通过，九任务假硬件 CLI 完成（七次 actor reset、一次 robot connect、一次 gripper home），Python 编译检查与 `git diff --check` 通过。测试环境复用了前次审查的 PyYAML/requests 依赖副本，没有安装现场 SDK、Torch 或该 checkpoint；因此本次没有真实网络前向、视觉图像联调或真机运行。`--mode check` 已提供现场 checkpoint 前向检验，必须在配置完成后执行。
