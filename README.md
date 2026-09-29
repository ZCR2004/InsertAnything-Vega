# InsertAnything-Vega · RoCo 九任务执行

本分支面向 **RoCo 比赛改装版 DexMate Vega 的右臂、`tip_r` 和 CAN 二指夹爪**：头部视觉定位任务板和物体区域，腕部视觉对齐后抓取，再执行规划插入或 InsertAnything RL 插入。

**两种电池先做规划抓取/插入，其余七种物体使用 Hexagon-III 策略。** 力观测暂时置零，任务过程中机械爪保持经本机验证的竖直朝下姿态；不同任务允许不同 yaw。代码已通过离线流程和假硬件测试，真机抓取、插入成功率仍需逐任务验证。

- [现场操作顺序：标定、规划测试、夹取测试、单任务插入](roco_vega/现场操作顺序.md)
- [策略接口、坐标约定与实现说明](roco_vega/README.md)
- [与最新 SteadyHand 抓取流程的比较](docs/SteadyHand抓取对比_20260929.md)
- [第三方来源、固定版本与清理范围](third_party/README.md)

## 执行流程

```text
已验证的竖直初始姿态
  → 拍摄任务板，关联物体区域
  → 抓取 TCP 位姿上方 5 cm，腕部视觉 XY 对齐
  → 竖直下降，限流夹取，检查持有
  → 竖直抬升（默认 10 cm），经过已教高位转运路点
  → 成功插入 TCP 位姿 A 正上方 4–5 cm（默认 4.5 cm）
  → 电池：规划小步下插；其他：Hexagon-III RL
  → 深度、XY、姿态误差持续达标，停止下压
  → 松爪并确认，从实测释放点沿原 XY 上抬 15 cm
  → 放置验证，可选复位，进入下一任务
```

这里的 5 cm、4–5 cm 分别相对**抓取 TCP 位姿**、**成功插入 TCP 位姿 A**，不是直接相对物体表面或孔口。A 必须在固定夹持深度与方向下记录；从视觉孔位生成 A 还需工具/物体偏移与插入深度，当前没有自动孔位检测或换算模块。

| 顺序 | task ID | 插入方式 |
|---:|---|---|
| 1 | `battery_size1` | 规划 |
| 2 | `battery_size5` | 规划 |
| 3 | `gear_60teeth` | RL |
| 4 | `gear_20teeth` | RL |
| 5 | `rod_16mm` | RL |
| 6 | `bolt_8mm` | RL |
| 7 | `usb_a` | RL |
| 8 | `hdmi` | RL |
| 9 | `pin` | RL |

执行顺序由 `EXECUTION_ORDER` 决定；配置的 `tasks` 列表保留固定字段顺序，不要靠随意重排 JSON 改执行顺序。

## 安装与离线检查

Python **3.10+**，以下命令从仓库根目录执行。现场须使用能够访问机器人 SDK、相机和 CAN 的 Linux Python 环境。

```bash
python -m pip install -r requirements-roco.txt
python -m unittest -q test_roco_vega test_roco_motion test_roco_runtime
python run_roco_vega.py --mode simulate --batteries-only
python run_roco_vega.py --mode simulate
python run_roco_vega.py --mode template --output configs/roco.local.json
```

`simulate` 使用假机器人和解析动作器，验证真实调度、观测构造、动作后处理和成功判定。它不连接机器人、不加载训练权重，也不是接触物理仿真。日志写入 `roco_runs/`。

现场另需 `dexcontrol==0.5.0`、Pinocchio、`wrist_cameras`、CAN 驱动及机器人端 motion plugin；这些由现场部署提供，不包含在 pip 依赖文件中。先确认 SDK/相机能工作，再在同一解释器配置 Torch。纯电池流程不加载 RL actor。

## 权重

七个 RL 任务共用公开的 **InsertAnything-HexagonHole-III-Direct-v0**，并非前面 Sharpa/π0.5 训练得到的策略，也未在本项目中重新训练成九任务策略。

从[原作者权重仓库](https://huggingface.co/tOMRmA/InsertAnything-checkpoints)下载：

```bash
python -m pip install huggingface_hub
hf download tOMRmA/InsertAnything-checkpoints InsertAnything-HexagonHole-III-Direct-v0.pth --local-dir .pretrained_checkpoints
```

将 `configs/roco.local.json` 中 `policy.checkpoint` 设置为以下路径（相对配置文件目录）：

```text
../.pretrained_checkpoints/InsertAnything-HexagonHole-III-Direct-v0.pth
```

运行器校验的 SHA256：

```text
d11d9a4849238dbdec4602efb86df6d260118fb7ca59e0f159a05546d2b14c08
```

观测维度为 26：相对位置 3、绝对四元数 4、相对四元数 4、线速度 3、角速度 3、原始力 3、上一动作 6。力在原始观测中置零后仍经过 checkpoint 的归一化。每任务重置 LSTM 与动作历史；实际执行平移并锁定该任务的完整姿态。零力、锁姿态、限步长和 Vega 阻塞式关节控制改变了部署条件，不能把原论文结果直接当成本机成功率。

## 现场标定与分阶段测试

请按[现场操作顺序](roco_vega/现场操作顺序.md)完成：

1. 核对本机机器人名、右臂、TCP 轴和夹爪，必要时记录/分析工具坐标。
2. 检查头部与物理右腕 `wrist_a` 图像，标定任务板到 base 的坐标与表面高度。
3. 为每种物体教抓取区域、尺寸范围、腕部特征/目标像素、抓取高度、yaw 和夹持参数。
4. 在一致夹持关系下记录成功插入 TCP 位姿 A，填写转运路点、策略坐标映射与成功容差。
5. 依次验证空爪路径、夹取、单任务插入、单任务全流程，最后连跑两个电池和九任务。

模板中的未知坐标为 `null`，默认拒绝真机运行。上游配置仅供结构参考，不是本机标定结果。抓取测试可先只填写抓取相关标定；电池子集可保留七个 RL 任务未标定。

| 入口 | 用途 |
|---|---|
| `run_roco_calibration.py --tool snapshot` | 只采集相机图像 |
| `run_roco_calibration.py --tool tcp-record / tcp-analyze` | 只读工具位姿记录与分析 |
| `run_roco_calibration.py --tool board` | 交互板面标定，会运动 |
| `record_roco_pose.py` | 只读记录稳定 TCP 位姿 A/路点 |
| `run_roco_test.py --test plan` | 空爪分段到 A 上方，不下插 |
| `run_roco_test.py --test pick` | 抓取、提起检查，再由操作员确认放回 |
| `run_roco_test.py --test insert` | 同一会话内手动装夹，单独测规划/RL 插入 |
| `run_roco_vega.py` | 单任务、双电池或完整九任务 |

完成标定后，先检查两电池配置，再启动现场流程：

```bash
python run_roco_vega.py --mode check --config configs/roco.local.json --batteries-only
python run_roco_vega.py --mode live --config configs/roco.local.json --batteries-only --confirm-empty-gripper --confirm-calibrated-paths
```

单项完整流程把 `--batteries-only` 换为 `--task usb_a`；全部九项去掉这个选择参数，并完成全部任务标定。`check` 不连接 SDK；选中 RL 任务时会验证权重并执行真实 actor 前向。

## 当前能力边界

- **视觉版本固定**：使用 SteadyHand `5efaa613` 的板定位、几何区域关联和局部腕部跟踪，尚未合入上游最新 `a6fed00` 按物体保存模板并全图重定位的流程。不是开放场景语义识别器。
- **插入目标依赖标定**：当前使用每任务 A 和板面 XY 平移补偿；板旋转/高度或夹持关系改变后要重标定。没有视觉自动精确检测九个孔位。
- **竖直约束**：检查起始、命令、采样关节插值与运动读回；不从任意倾斜姿态自动翻转。底层分段 IK/关节位置控制不是全局碰撞规划，也没有保证未采样的连续轨迹完全无误差。
- **完成判定**：深度与误差持续达标后才松爪，另有持有/放置验证。默认由操作员确认；外观和 TCP 深度不能替代电气连通、实际啮合等功能评测。
- **失败行为**：整轮共用一个硬件会话，空爪启动时只 home 一次。超时、陈旧状态、掉物或判定失败会停止序列，不自动试下一项。重启不会盲目按旧 `progress.json` 跳过任务。

## 仓库结构

```text
run_roco_vega.py            九任务主入口
run_roco_test.py            plan / pick / insert 分阶段入口
run_roco_calibration.py     标定工具包装
record_roco_pose.py         只读 TCP 记录
configs/                   未标定配置模板；本机 *.local.json 不进 Git
roco_vega/                 抓取、运动约束、插入、判定与任务调度
  policy_runtime/          精简保留的 InsertAnything 推理核心
third_party/steadyhand/    固定版 Vega 运行与标定依赖
third_party/versions.json  来源提交与实际保留文件清单
test_roco_*.py             离线回归与依赖完整性检查
docs/                     上游比较说明
```

已移除本分支不使用的 Isaac Lab 训练环境、USD 资源、Franka HTTP/旧单孔执行入口、Sharpa 后端、旧示例/部署工具和旧论文 README。原实现仍可从 Git 历史或[InsertAnything 上游](https://github.com/mzhsoul/InsertAnything)查阅；当前分支专注 Vega 比赛执行。许可证及第三方来源见 [LICENSE](LICENSE) 和[来源说明](third_party/README.md)。
