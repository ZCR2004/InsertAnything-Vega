# RoCo Vega：策略接口与实现说明

实现分支：`feat/roco-vega-rl-sequence`。本入口面向右臂 `tip_r`，固定 SteadyHand `5efaa61368d6ad6777115c154b1fbe81e6db47b9`。采用官方比赛改装 Vega 的关节位置控制和 CAN 二指夹爪；不使用 Franka 控制服务。来源见 [`third_party/versions.json`](../third_party/versions.json)。安装、权重下载与入口总览见[仓库 README](../README.md)。本次仅清理无关代码，没有把上游最新模板识别流程合入；差异见[对比说明](../docs/SteadyHand抓取对比_20260929.md)。

首次上机请按 **[现场操作顺序](现场操作顺序.md)** 操作。其中包含相机/TCP/板面标定、只读记录 A、空爪路径、夹取、手动装夹后单任务插入的独立命令。

## 已实现的流程

```text
竖直初始姿态 → 板定位 / 零件区域关联 → 抓取TCP上方5 cm → 腕部视觉 XY 对齐
→ 竖直下降 → 限流夹取 → 默认抬升10 cm并验证仍持有
→ 已标定转运路点 → A 正上方 45 mm → 静止读回检查
→ RL（7 类）或小步规划下插（2 类电池）
→ 深度 + 横向 + 姿态容差持续满足 → 停止下压 → 松爪并确认
→ 原XY垂直上抬15 cm → 放置验证 → 可选复位 → 写入 progress.json → 下一任务
```

九任务执行顺序为 `battery_size1, battery_size5, gear_60teeth, gear_20teeth, rod_16mm, bolt_8mm, usb_a, hdmi, pin`，先完成两个无RL电池。配置存储列表仍采用原物体顺序。RL 使用同一份六边形 **Hexagon-III** checkpoint，每项任务重新清空 LSTM、动作 EMA、上一动作和成功判定状态。两种电池不调用 actor。

竖直约束覆盖起始读回、拍板、抓取、转运、插入和复位；yaw允许改变。所有移动经统一会话分成小段，真实adapter在发命令前求解全部段的IK并检查采样关节插值，执行时检查姿态读回。仍需确认校正后 `tip_r` 轴确实对应物理爪向下；不从未知倾斜姿态自动回正，也不宣称SDK连续轨迹或碰撞已经得到真机验证。默认限制和迁移说明见现场操作顺序。

一个 `RobotSession` 持有一份机器人、相机和夹爪连接。夹爪仅在空爪启动时标定一次，抓住物体后不会重新连接或 home。超时、取消、状态陈旧、丢失抓取、超深、验证失败都会停止整个序列，不自动重试、不盲退、不自动张爪。Ctrl+C 停止活动命令并请求软件急停；日志不能替代现场确认。

## 环境和离线运行

从仓库根目录执行。需要 Python 3.10+；纯推理不需要安装整个 Isaac Lab 扩展。

```bash
python -m pip install -r requirements-roco.txt
python -m unittest -v test_roco_vega test_roco_motion test_roco_runtime
python run_roco_vega.py --mode simulate --batteries-only
python run_roco_vega.py --mode simulate
python run_roco_vega.py --mode template --output configs/roco.local.json
```

`simulate` 使用假机器人和解析动作器，经过实际的 26 维观测、动作后处理、深度监控与调度代码，输出到 `roco_runs/`。它验证接口与流程，**不是物理仿真，也不代表 checkpoint 能插入这些物体**。模板已另存为 [`configs/roco_vega.template.json`](../configs/roco_vega.template.json)，其中未知坐标为 `null`，默认拒绝真机启动。

真机需现场环境中的 `dexcontrol==0.5.0`、Pinocchio、`wrist_cameras`、CAN 驱动和机器人服务器 motion plugin。不要仅为兼容此脚本替换已调通的现场驱动；先在同一个 Python 解释器中确认这些依赖与 Torch 可以共存。

## 如何填写标定

1. 在 `configs/robot.local.json` 保存经过核对的 SteadyHand 机器人配置。可以参考 `third_party/steadyhand/configs/robots/vega.json` 的结构，但不能直接采用其他现场的机械臂位置、相机修正或夹爪电流。相对 URDF/driver 路径沿用 SteadyHand 的根目录解析方式，现场路径可改成绝对路径。
2. 用固定版本 SteadyHand 的五点标定工具得到 `configs/board.local.json`（schema 2，包含板轴、表面和相机角点）。该配置自带 12 小时时效检查。固定板面/头部姿态，只对允许范围内的板 XY 平移补偿；板旋转或高度变化需要重新标定。上游 fallback 文件是别人的测量记录，不是本机标定。
3. 每种物体保持一致夹持深度与朝向、末端垂直向下，记录**已成功插入时**的真实 `tip_r` 位姿 A 到 `success_pose_xyzw`，单位 m，顺序 `[x,y,z,qx,qy,qz,qw]`。这里既不能填孔中心，也不能用空爪 TCP 代替夹持状态 A。Vega 修正后 `tip_r` 的垂直约定与 Franka 局部工具轴不同，不能照抄 Franka 的 roll=π。
4. 分别标定抓取TCP高度 `grasp_z_m`、图像feature/goal像素、物体区域与尺寸范围、夹持宽度、电流/速度、转运路点、验证视角和容差。抓取悬停为该TCP高度+`hover_height_m=0.05`，抬升默认+`lift_height_m=0.10`；转运路点必须高于抬升点和插入中间点。`entry_height_m` 默认0.045，必须在0.04～0.05内，切RL时再次检查实测位置。以上距离均相对TCP目标，不是未经夹持偏移换算的物体表面/孔口。`step_m` 是每轴单次命令位移上限（默认1 mm），不是力限制。松爪后退离15 cm，`reset_waypoints_xyzw=[]` 默认不复位。转运不含全局碰撞规划。
5. 填 `calibration_identity` 的机器人名、base/TCP、时间、夹持约定以及机器人配置、板标定文件 SHA256；绑定实际 checkpoint SHA256。文件变化后应重核标定再更新哈希，而不是用哈希代替标定。Linux 可用 `sha256sum`，PowerShell 可用 `Get-FileHash -Algorithm SHA256`。
6. 确认所有任务和 policy mapping 后再置各项 `calibrated/validated=true`。`head_q_rad` 是 **3 维**，应与五点标定时相同。配置与测试路径必须核对整段机械臂运动；TCP 不低于 floor 不等于整臂无碰撞。

当前首次运行默认 `supervised=true`：提起后和放置后分别人工输入 `yes`。无人工模式可将两处 verifier 改成 `template`，配置 `template` 图片相对路径、`roi_xyxy` 和 `min_score`，在固定视角验证外观。模板示例结构：

```json
{"mode":"template","template":"verification_images/usb_placed.png",
 "roi_xyxy":[800,500,1100,900],"min_score":0.9,
 "view_pose_xyzw":["替换为7个实测数值"]}
```

以上像素只是字段示意，必须实测。图像放置验证视角必须在15 cm退离高度以上，随板XY平移更新；人工验证留在退离位置，不另行移动。抓取接近、camera-clear和可选复位路点是固定base路点。外观验证不能证明USB电气连接或精确啮合。深度使用TCP相对A的垂直高度，必须与同一夹持关系配套，否则会出现物体滑脱但TCP到位的误判。

## 六边形策略的坐标与力输入

模型观测维度仍为 **26**：相对位置 3 + 绝对四元数 4 + 相对四元数 4 + 线速度 3 + 角速度 3 + 原始力 3 + 上一动作 6。三维原始力置零，之后仍执行 checkpoint 自带的归一化；不能删掉这三维，也不能把归一化后的力强制改零。Hexagon-III 保留原四元数参考、`symmetry_angles_deg=[0]`，没有套用其他六边形版本的对称设置。

每种物体的 `policy_mapping` 明确记录虚拟策略空间中的成功 TCP 位姿 `success_pose_policy`、孔位姿 `hole_pose_policy`、相对四元数参考和 base yaw。映射将真实 A 对应到策略的成功状态，并保持真实竖直方向；局部工具轴差用右乘四元数处理。**策略成功时的相对位置不一定是零**；必须从对应 checkpoint 的训练定义/标定获取，代码未硬编码一个通用孔深。`simulation.py` 中的 7.392 mm 仅为非零偏移回归用例。

第一版保持已标定的完整姿态（含 yaw），执行 actor 的平移动作，保留完整动作历史输入。不能宣称这是未经修改的仿真动作执行器：零力、姿态锁定、步长限制和 Vega 阻塞式位置控制都改变了部署条件。默认周期目标约 15 Hz，实际频率受 IK、SDK 轨迹完成和推理影响；过期观测不继续发动作，需要现场测时。

权重使用仓库公开的 Hexagon-III 单孔 checkpoint，校验值：

```text
d11d9a4849238dbdec4602efb86df6d260118fb7ca59e0f159a05546d2b14c08
```

下载命令和文件路径见[仓库 README 的权重部分](../README.md#权重)。该权重不是在 RoCo 九任务或零力条件下重新训练过的模型；跨物体能力要逐任务验证。

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

只连跑两电池的完整自动抓取/插入流程，不加载RL模型：

```bash
python run_roco_vega.py --mode check --config configs/roco.local.json --batteries-only
python run_roco_vega.py --mode live --config configs/roco.local.json --batteries-only --confirm-empty-gripper --confirm-calibrated-paths
```

该子集仅要求公共标定和两个电池完成；七个RL任务可留空。所有真机启动仍要求现场竖直起始姿态和已验证路径。

失败后保留 `events.jsonl` 与原子更新的 `progress.json`。重启不会默默跳过旧任务，必须核实场景后明确选择单任务或重排执行。旧的单孔执行入口已删除；所有现场执行使用本页与根目录 README 列出的 RoCo 入口。

## 模块与验证范围

| 模块 | 作用 |
|---|---|
| `frontend.py` | SteadyHand 板识别、已教区域关联、腕部 XY servo、抓取和放置验证 |
| `motion.py` / `session.py` / `hardware.py` | 全程竖直约束、笛卡尔分段、IK路径检查、取消和新鲜状态读回 |
| `policy_runtime/` | 从旧工程抽出的 checkpoint、观测、动作后处理与数学核心；不依赖 Isaac Lab 或 Franka HTTP |
| `geometry.py` / `insertion.py` | Vega↔策略坐标、26D 零力观测、LSTM、RL/规划插入 |
| `monitor.py` / `orchestrator.py` | 深度驻留、失败分支、释放退离、九任务调度和日志 |
| `task_spec.py` / `preflight.py` | 标定合同、身份绑定、完整运行前检查 |
| `test_roco_vega.py` / `test_roco_motion.py` | 假硬件全流程、生产frontend和姿态故障回归 |
| `../run_roco_test.py` / `commissioning.py` | 独立 plan、pick、insert 阶段测试 |
| `../record_roco_pose.py` | 无运动的实测 TCP 位姿记录 |
| `../run_roco_calibration.py` | 使用本机配置调用固定版 TCP/相机/板面标定工具 |

识别接入的是固定版本 SteadyHand 的几何/图像处理；零件通过已教区域和尺寸匹配，并拒绝歧义。它不是任意散乱、任意遮挡场景下的语义识别器。真机接触效果、零力策略成功率、视觉模板阈值与周期稳定性仍需现场验证。

2026-09-29 清理后验证：60项离线测试覆盖生产frontend上的双电池抓取/规划插入、释放后15 cm退离与复位、命令前/运动中的倾斜拒绝，以及标定工具动态导入和运行依赖。删除了旧独立执行器的专用测试，保留当前执行路径的测试。本机没有现场SDK、Torch或该checkpoint，因此未进行真实网络前向、视觉图像联调或真机运行。`--mode check` 提供现场checkpoint前向检验，须在配置完成后执行。
