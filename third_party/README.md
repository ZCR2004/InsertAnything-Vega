# 来源与保留范围

本分支只保留 RoCo Vega 执行和标定需要的依赖。精确来源提交与 SteadyHand 文件清单见 [versions.json](versions.json)。

## SteadyHand 与比赛硬件

- 来源：[matveho/ROCO-SteadyHand](https://github.com/matveho/ROCO-SteadyHand/tree/5efaa61368d6ad6777115c154b1fbe81e6db47b9)，固定提交 `5efaa61368d6ad6777115c154b1fbe81e6db47b9`。
- 保留 Vega adapter、相机、CAN gripper、Pinocchio IK、任务板/腕视觉，以及四种标定入口的传递依赖。
- 动态加载的 `vendor/dexmate_setup/gripper.py` 也在保留清单中；只按静态 Python import 删文件会误删它。
- `configs/task_coordinates.json` 是 `task_board.json` 引用的上游坐标来源记录；不是本分支用于自动生成成功 A 的接口。
- 删除 Sharpa/mock 后端、上游独立比赛/电池执行器、不使用的现场调试和部署工具、旧上游文档。保留的上游代码及示例配置内容未修改，所有本项目包装位于 `roco_vega` 和根目录入口。
- 比赛硬件改装来源：[dexmate-setup](https://github.com/intelligent-control-lab/dexmate-setup/tree/4be4d16140b25f730673d664b49c58280b07ef46)。当前通过固定版 SteadyHand 使用其 CAN 夹爪驱动及比赛 URDF 约定；不是另启动一个名为 dexmate-setup 的控制服务。

上游没有在这个固定提交中提供顶层 LICENSE。保留现有作者/来源标记；本清单不额外授予上游材料的再分发权利。示例机器人的 IP、姿态、夹爪参数和板标定是来源记录，不能视为本机的有效标定。

## InsertAnything 推理核心

来源参考：[mzhsoul/InsertAnything](https://github.com/mzhsoul/InsertAnything/tree/f033da5ccbc35a34369a0ac86e04c3e74b1408a0)，以及本仓库清理前提交 `48a34827a997df801d976583d0080576e425c39f`。仓库原有 [BSD-3-Clause LICENSE](../LICENSE) 已保留。

旧 `source/InsertAnything/.../sim2real/` 中以下实现迁入 [roco_vega/policy_runtime](../roco_vega/policy_runtime)：

| 文件 | 保留用途 / 修改 |
|---|---|
| `torch_policy.py` | checkpoint 架构、观测归一化、LSTM actor 前向；本次未改算法 |
| `observation_builder.py` | 观测及策略姿态约定；仅改为包内 import |
| `action_postprocessor.py` | 动作 EMA、限幅和目标变换；仅改为包内 import |
| `transforms.py` | 位姿、四元数和速度变换；本次未改算法 |
| `robot_state_adapter.py` | 仅保留 `RobotState` 数据结构，删除 Franka HTTP 读取/转换 |

旧 `vega_backend.py` 仅仍在使用的 `twist_between` 移入 `roco_vega/session.py`，不再保留重复的机器人连接/执行器。旧 safety guard 及其专用测试随旧入口删除；当前运动约束、插入边界和错误处理仍由现有测试覆盖。

训练环境、Isaac Lab 扩展、USD、旧独立执行器及旧配置在此分支删除，仍可从历史提交获取。当前 README 的权重下载和本地路径不依赖被删除目录。

## 为什么没有直接升级到最新 SteadyHand

本次清理调整目录与文档，没有把上游最新任务坐标、控制入口或抓取算法混入当前流程。`a6fed00` 的模板重定位、标注坐标归一化等值得后续单独迁移和测试；比较见 [SteadyHand 抓取对比](../docs/SteadyHand抓取对比_20260929.md)。
