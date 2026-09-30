# 最新 SteadyHand 与本分支的抓取流程

核查日期：2026-09-29。通过 `git fetch origin main` 核对的最新提交为 **`a6fed00294eb8dfe20eecfc61e768a3f362c27ed`**，北京时间 12:37:26。本文固定引用该提交，后续上游变化不自动反映在本文中。

本分支集成的 SteadyHand 仍固定在 **`5efaa613`**。下面区分“用户希望的完整流程”“本分支已经实现的流程”和“上游最新源码”，避免把上游功能当成本地已经具备。

## 结论

**抓取前半段思路一致：粗定位 → 上方悬停 → 腕视觉对齐 → 下降 → 闭合 → 抬升。** 上游最新版本对“保存每种物体的特征并在下一次运行时重新找到它”支持更完整；本分支则增加了竖直运动检查、A 上方切换 RL、持续深度判定以及成功后 15 cm 退离。

两者尚不等价：上游的放置完成并不判定装配成功；本分支也还没有把上游最新模板重定位和标注坐标导入接进来。

## 上游最新实现怎么抓

当前主入口是 [vega_competition_pipeline.py](https://github.com/matveho/ROCO-SteadyHand/blob/a6fed00294eb8dfe20eecfc61e768a3f362c27ed/tools/vega_competition_pipeline.py)，按物体教学与执行位于 [vega_wrist_part_calibrate.py](https://github.com/matveho/ROCO-SteadyHand/blob/a6fed00294eb8dfe20eecfc61e768a3f362c27ed/tools/vega_wrist_part_calibrate.py)。

1. 头部相机重拍任务板，将已知物体的板内坐标注册到机器人 base，移动到本机测得的 RIGHT_READY。
2. 按所选 task 的粗抓取坐标移到板面上方 **100 mm**。该抓取执行器用板表面函数 `surface(x,y)` 加净空计算高度。
3. 教学时选物体上的纹理/边缘特征和“正确对准夹爪时应出现的位置”；保存对齐前、未被爪遮住的图像块、锚点、图像尺寸、yaw、抓取深度及文件哈希。
4. 下次运行加载该物体模板，先重新定位，再局部跟踪。使用腕相机 `wrist_a`，通过探测 X/Y 小位移估计像素雅可比，闭环对齐。它是针对已选物体的模板与几何流程，不是自动把任意散乱九类物体全部分类的语义检测器。
5. 从悬停高度向下到 `surface(x,y)+grasp_clearance_m`。`depth N` 定义的是从 100 mm 悬停位向下多少毫米，而不是相对物体表面的距离。
6. 使用 CAN 夹爪限流闭合，核对 `last_grip_result()['gripped']`，再回到刚才的悬停位。失败或可能仍持物时，不盲目重试。
7. 放置时，按已知目标板内 XY 加该物体教过的偏移移动到目标上方，下降到教过的放置净空，打开夹爪，回到板上方 100 mm。

源码明确说明，完成释放不推断已经插入或装配。没有 Hexagon-III actor，也没有本分支的 A 深度、XY、姿态误差持续达标判定。

## 和我们的流程逐项比较

| 项目 | SteadyHand 最新 `a6fed00` | 本分支 |
|---|---|---|
| 粗定位 | 拍板注册 + 审核后的板内物体坐标 | 拍板 + 已教区域/尺寸关联可见物体 |
| 腕部精定位 | 每物体持久模板、重新定位、局部跟踪 | 配置 `feature_uv/goal_uv`，运行时局部跟踪；尚未接入持久模板的重新定位 |
| 抓取悬停 | 抓取执行器为板表面 +10 cm | 已教抓取 TCP 高度 +5 cm |
| 下降高度 | 已教板面净空 | 已教 `grasp_z_m`，随板 XY 补偿 |
| 姿态 | 以实测 RIGHT_READY 四元数为参考，叠加 base Z 轴 yaw | 校正后 TCP 必须符合本机竖直约定；检查命令、路径采样与读回，yaw 可变 |
| 抬升与转运 | 回抓取悬停位，再到目标上方 | 默认抬升10 cm、已教高位路径，再到 A+4–5 cm |
| 插入 | 下降到已教放置位置并释放 | 两电池规划；其余七种切换六边形 RL，原始力置零 |
| 成功判定 | 夹取/释放有教学验证标记；没有运行时插入深度成功判定 | TCP 深度 + XY + 姿态 + 驻留，另加持有/放置验证 |
| 释放后 | 回到板面 +10 cm | 从实际释放位置原XY抬升15 cm，可选复位 |
| 任务顺序 | 可配置得分优先顺序，跳过未验证物体 | 固定先两电池，再七个RL任务；完整九任务必须全部标定 |
| 失败恢复 | 操作员确认的有限重试/跳过/停止；疑似持物时禁止重试 | 整个序列停止，现场确认后另行启动，不自动重试或跳过 |

这里的“竖直”必须绑定实际工具坐标。上游参考姿态平放并不等于在所有起始姿态和每段实际关节轨迹上都执行了本分支的竖直检查；本分支的采样检查也不等于全程连续运动已获真机证明。

## 这次新增的坐标与调度更新

[最新提交](https://github.com/matveho/ROCO-SteadyHand/commit/a6fed00294eb8dfe20eecfc61e768a3f362c27ed) 已把审核后的 initial/final 图像标注接入任务坐标：

- [导入工具](https://github.com/matveho/ROCO-SteadyHand/blob/a6fed00294eb8dfe20eecfc61e768a3f362c27ed/tools/vega_import_task_annotations.py) 使用各自矫正图像的像素比例，统一归一化到 **386 mm × 386 mm**，并配置板内 180° 旋转、不做 X 镜像。此前两幅导出图的尺度差异，现在有显式归一化处理。
- 标注里的 Z 被忽略；实际 Z 来自板面标定和已教净空。因此“有 final XY”仍不等于“已有成功插入 TCP 位姿 A”。
- 旧 organizer 坐标另存参考；connect/grade 次级点不当作已验证的实时插入目标。
- [competition_plan.json](https://github.com/matveho/ROCO-SteadyHand/blob/a6fed00294eb8dfe20eecfc61e768a3f362c27ed/configs/competition_plan.json) 的当前顺序是 `battery_size1 → gear_20teeth → gear_60teeth → pin → bolt_8mm → battery_size5 → rod_16mm → usb_a → hdmi`，并非两个电池一起先做。
- 只有通过 `grasp_verified/place_verified` 等门槛的任务进入对应运行计划；新增持物不确定性摘要及默认一次、须操作员确认的重试。

在该提交的公开 `calibration/` 树中，没有提交 `wrist_part_profiles.json` 和 `wrist_templates/` 的实际九任务教学结果。**有教学和执行代码，不代表公开仓库已经给出九类物体可直接复用的完整实机标定或成功证据。**

## 一处需先修复的上游问题

该版本 [return_part()](https://github.com/matveho/ROCO-SteadyHand/blob/a6fed00294eb8dfe20eecfc61e768a3f362c27ed/tools/vega_wrist_part_calibrate.py#L280) 在执行 `open_gripper()` 后，记录事件时使用了当前函数未定义的 `release` 和 `settings`。从源码判断，走到此分支会出现 `NameError`，其后的回升和持物标记更新不能正常完成。

这是源码检查发现，未执行机器人验证。它属于上游“放回原处”的路径，不是说每次 `pick_place` 都会触发。当前分支没有调用这一最新版工具，不能直接整体替换后宣称无需测试。

## 后续适合迁移什么

优先迁移“每物体模板教学/重定位 + 图像/配置身份检查”，让粗定位后更稳健地找到同一个特征；其次是审核后的板内 XY 数据导入。保留本分支的单硬件会话、竖直约束、两电池先做、A/RL/成功判定及15 cm退离。

迁移时要在我们的 **抓取TCP+5 cm** 视角重新教学模板与目标像素，不能直接拿上游 **板面+10 cm** 视角的数据用。目标坐标仍须换算并校准成功 TCP 位姿，不能只对齐 yaw 就省掉工具/夹持偏移和深度。
