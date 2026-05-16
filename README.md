# PM01 Motion Imitation — 项目文档

## 1. 项目概述

本项目是一个 **EngineAI PM01 人形机器人的运动模仿（Motion Imitation）强化学习系统**。

- **仿真平台**：NVIDIA Isaac Lab（Isaac Sim 4.5.0）
- **RL 算法**：PPO（RSL-RL 库）
- **目标**：训练 PM01 机器人复现预录制的人体动作捕捉数据（舞蹈、行走等）
- **并行环境**：4096 个
- **控制频率**：50 Hz（sim dt=0.005s, decimation=4）
- **观测维度**：129 维（策略网络输入）
- **动作维度**：24 个关节位置目标

### 技术路线

```
视频 → 人体姿态估计(GVHMR等) → 重定向到PM01关节 → CSV文件
                                                          ↓
                                               csv_to_npz_pm01.py
                                                          ↓
                                                    NPZ动作文件
                                                          ↓
                                          Isaac Lab + PPO 训练 (4096并行环境)
                                                          ↓
                                               ONNX / JIT 策略模型
                                                          ↓
                                                实机部署 (deploy.yaml)
```

---

## 2. 文件功能说明

### 2.1 顶层文件

| 文件 | 作用 |
|------|------|
| `engineai_rl_lab.sh` | 主入口脚本：`-i` 安装、`-t` 训练、`-p` 推理回放、`-l` 列出环境 |
| `pyproject.toml` | Python 构建配置（setuptools、isort、pyright） |
| `serial_links.xml` | MuJoCo 格式的 PM01 完整运动学模型（24 关节 J00-J23，含碰撞体和力传感器） |

### 2.2 核心包 — source/engineai_rl_lab/

#### 2.2.1 机器人资产 — assets/robots/

| 文件 | 作用 |
|------|------|
| `pm01_engineai.py` | PM01 机器人完整定义：USD 模型路径、初始站立姿态（z=0.9m）、两种电机参数、PD 控制增益、动作缩放系数 |
| `serial_pm_v2/` | PM01 的 USD 模型文件目录（~97MB 基础网格 + 物理/传感器层） |

**电机参数**：
- **Q90**（高扭矩）：用于 HIP_PITCH、HIP_ROLL、KNEE_PITCH — 力矩上限 164 Nm
- **Q25**（低扭矩）：用于其余所有关节 — 力矩上限 52 Nm

#### 2.2.2 环境配置 — tasks/mimic/robots/pm01/

| 文件 | 作用 |
|------|------|
| `__init__.py` | 注册 Gym 环境 `"EngineAI-PM01-Mimic"`，关联训练/播放配置和 PPO agent |
| `pm01_minic.py` | **核心环境配置文件**，定义以下所有 MDP 组件 |

**pm01_minic.py 详细配置**：

| 模块 | 内容 |
|------|------|
| `RobotSceneCfg` | 地面（平面地形 + 摩擦材质）、PM01 机器人、灯光、接触传感器（history=3, threshold=10N） |
| `CommandsCfg` | 运动指令：加载 NPZ 文件，锚点为 `LINK_TORSO_YAW`，跟踪 12 个身体部位，位姿/速度/关节位置随机化范围 |
| `ActionsCfg` | 关节位置动作：24 个关节，使用默认偏移 |
| `ObservationsCfg` | 策略观测（129维，带噪声）+ Critic 特权观测（无噪声，额外包含线速度等） |
| `RewardsCfg` | 6 个指数跟踪奖励 + 4 个正则化/安全惩罚 |
| `TerminationsCfg` | 超时(30s)、锚点 z 漂移(0.25m)、锚点朝向偏差(0.8)、末端偏离(0.25m) |
| `EventCfg` | 摩擦力随机化、关节默认位置偏移、质心随机化、周期性推扰 |

**观测空间详情（129维）**：

| 观测项 | 维度 | 噪声 | 说明 |
|--------|------|------|------|
| motion_command | 48 | 无 | 目标关节位置(24) + 目标关节速度(24) |
| motion_anchor_ori_b | 6 | ±0.05 | 参考动作锚点朝向（6D旋转矩阵） |
| base_ang_vel | 3 | ±0.2 | 基座角速度 |
| joint_pos_rel | 24 | ±0.01 | 当前关节位置（相对默认值） |
| joint_vel_rel | 24 | ±0.5 | 当前关节速度 |
| last_action | 24 | 无 | 上一步动作 |

**奖励函数详情**：

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| motion_global_anchor_pos | 0.5 | 锚点位置跟踪（std=0.3） |
| motion_global_anchor_ori | 0.5 | 锚点朝向跟踪（std=0.4） |
| motion_body_pos | 1.0 | 身体位置跟踪（std=0.3） |
| motion_body_ori | 1.0 | 身体朝向跟踪（std=0.4） |
| motion_body_lin_vel | 1.0 | 身体线速度跟踪（std=1.0） |
| motion_body_ang_vel | 1.0 | 身体角速度跟踪（std=3.14） |
| action_rate_l2 | -0.1 | 动作变化率惩罚 |
| joint_limit | -10.0 | 关节极限惩罚 |
| joint_acc | -2.5e-7 | 关节加速度惩罚 |
| joint_torque | -1e-5 | 关节力矩惩罚 |
| feet_collision | -1.5 | 双脚碰撞惩罚（x<20cm 或 y<10cm） |
| undesired_contacts | -0.1 | 非期望接触惩罚（排除脚踝和手腕） |

**域随机化详情**：

| 项目 | 范围 | 时机 |
|------|------|------|
| 静摩擦系数 | 0.3 ~ 1.6 | 启动时 |
| 动摩擦系数 | 0.3 ~ 1.2 | 启动时 |
| 恢复系数 | 0.0 ~ 0.5 | 启动时 |
| 关节默认位置偏移 | ±0.01 rad | 启动时 |
| 躯干质心偏移 | x:±0.025, y:±0.05, z:±0.05 m | 启动时 |
| 速度推扰 | x/y:±0.5, z:±0.2 m/s, roll/pitch:±0.52, yaw:±0.78 rad/s | 每 1~3 秒 |

#### 2.2.3 Agent 配置 — tasks/mimic/agents/

| 文件 | 作用 |
|------|------|
| `rsl_rl_ppo_cfg.py` | PPO 超参数配置 |

**PPO 参数**：

| 参数 | 值 |
|------|-----|
| Actor 网络 | MLP (512→256→128, ELU) |
| Critic 网络 | MLP (512→256→128, ELU) |
| 分布类型 | 高斯分布（标量标准差，初始值 1.0） |
| 学习率 | 1e-3（自适应调度） |
| gamma | 0.99 |
| lambda (GAE) | 0.95 |
| clip_param | 0.2 |
| entropy_coef | 0.005 |
| num_learning_epochs | 5 |
| num_mini_batches | 4 |
| max_grad_norm | 1.0 |
| 每环境步数 | 24 |
| 最大迭代次数 | 30000 |
| 保存间隔 | 每 1000 步 |

#### 2.2.4 MDP 模块 — tasks/mimic/mdp/

| 文件 | 作用 |
|------|------|
| `__init__.py` | 统一导出所有 MDP 函数 |
| `commands.py` | **运动指令系统**：`MotionLoader` 加载 NPZ；`MotionCommand` 管理参考动作跟踪，实现自适应采样（困难片段过采样）、重置时加随机扰动、坐标系变换 |
| `observations.py` | 观测函数：锚点朝向(6D)、身体位置/朝向(体坐标系)、参考动作锚点位置/朝向 |
| `rewards.py` | 奖励函数：6 个指数核跟踪奖励 + 脚部碰撞/非期望接触惩罚 |
| `events.py` | 域随机化：关节默认位置偏移、刚体质心随机化 |
| `terminations.py` | 终止条件：锚点 3D/z 轴距离、锚点朝向、跟踪体 z 轴偏离 |

#### 2.2.5 工具 — utils/

| 文件 | 作用 |
|------|------|
| `export_deploy_cfg.py` | 训练后导出 `deploy.yaml`：身体名称、控制频率、默认关节位置、动作缩放/偏移/裁剪、观测维度/缩放 |
| `parser_cfg.py` | 从 Gym 注册表加载环境配置，覆盖 device/num_envs |

### 2.3 运行脚本 — scripts/

#### 2.3.1 训练与推理 — scripts/rsl_rl/

| 文件 | 作用 |
|------|------|
| `train.py` | 训练入口：创建 Isaac Sim 环境 → RSL-RL PPO 训练 → 保存 checkpoint + 配置文件 + deploy.yaml |
| `play.py` | 推理入口：加载 checkpoint → 仿真回放 → 导出 ONNX + JIT 策略 |
| `cli_args.py` | CLI 参数定义（实验名、恢复训练、checkpoint 路径、日志类型） |
| `export_IODescriptors.py` | 导出观测/动作的 IO 描述到 YAML（部署工具用） |

#### 2.3.2 动作数据处理 — scripts/mimic/

| 文件 | 作用 |
|------|------|
| `csv_to_npz.py` | CSV → NPZ 转换（G1-29dof 机器人用，默认 50 FPS） |
| `csv_to_npz_pm01.py` | CSV → NPZ 转换（PM01 专用，默认 30 FPS 输入 → 50 FPS 输出） |
| `replay_npz.py` | NPZ 可视化回放（G1-29dof） |
| `replay_npz_pm01.py` | NPZ 可视化回放（PM01 专用） |

#### 2.3.3 其他

| 文件 | 作用 |
|------|------|
| `list_envs.py` | 列出所有已注册的 EngineAI Gym 环境 |

### 2.4 动作数据 — motion_data/

| 目录/文件 | 来源 | 内容 |
|-----------|------|------|
| `minic/dance1_subject1.csv/.npz` | minic 动捕 | 舞蹈动作 1（受试者 1） |
| `minic/dance1_subject2.npz` | minic 动捕 | 舞蹈动作 1（受试者 2） |
| `minic/dance2_subject3.csv/.npz` | minic 动捕 | 舞蹈动作 2（受试者 3） |
| `minic/walk.csv/.npz` | minic 动捕 | 行走动作 |
| `gvhmr/mywalk.csv/.npz` | GVHMR 视频动捕 | 行走 |
| `gvhmr/sharkHand.csv/.npz` | GVHMR 视频动捕 | 鲨鱼手动作 |
| `walk_0226/test.csv/.npz` | — | 测试行走数据 |

**CSV 格式**：每行 31 列 = `base_pos_xyz(3) + base_rot_quat_xyzw(4) + dof_pos(24)`

### 2.5 训练产物 — logs/

| 路径 | 说明 |
|------|------|
| `model_*.pt` | PPO checkpoint（每 1000 步一个） |
| `exported/policy.onnx` | 导出的 ONNX 策略（部署用） |
| `exported/policy.pt` | 导出的 JIT 策略 |
| `params/env.yaml` | 环境配置快照 |
| `params/agent.yaml` | PPO agent 配置快照 |
| `params/deploy.yaml` | 实机部署配置 |

---

## 3. 服务器部署

### 3.1 环境要求

| 依赖 | 版本要求 |
|------|----------|
| NVIDIA Isaac Sim | 4.5.0 |
| Isaac Lab | 对应 Isaac Sim 4.5.0 版本 |
| RSL-RL | >= 2.3.1 |
| Python | >= 3.10 |
| CUDA GPU | 训练需要（4096 并行环境） |
| conda | 需要激活 conda 环境 |

### 3.2 安装步骤

```bash
# 1. 激活 conda 环境（已安装 Isaac Sim + Isaac Lab）
conda activate <your_isaac_env>

# 2. 设置 ISAACLAB_PATH 环境变量
export ISAACLAB_PATH=/path/to/isaaclab

# 3. 进入项目目录
cd /path/to/pm01_minic

# 4. 一键安装（pip editable + conda 环境配置 + argcomplete）
source engineai_rl_lab.sh -i
```

安装脚本会自动执行：
- `git lfs install`（拉取 USD 大文件）
- `pip install -e source/engineai_rl_lab/`（editable 模式安装本项目包）
- 配置 conda activate.d 脚本（自动设置 `ISAACLAB_PATH` 和 `ENGINEAI_RL_LAB_PATH`）
- 注册 argcomplete（命令行自动补全）

### 3.3 验证安装

```bash
# 列出已注册的 Gym 环境
source engineai_rl_lab.sh -l
```

应看到 `"EngineAI-PM01-Mimic"` 环境。

---

## 4. 训练流程

### 4.1 准备动作数据

如果有原始 CSV 动捕数据，先转换为 NPZ：

```bash
python scripts/mimic/csv_to_npz_pm01.py \
    --input_file motion_data/minic/dance1_subject1.csv \
    --input_fps 30 \
    --output_fps 50
```

参数说明：
- `--input_file`：输入 CSV 文件路径
- `--input_fps`：原始动捕帧率（默认 30）
- `--output_fps`：输出帧率（默认 50，需与训练控制频率一致）
- `--frame_range START END`：可选，只截取指定帧范围

该脚本会：
1. 加载 CSV（基座位置 + 四元数 + 关节角度）
2. 插值到目标 FPS
3. 计算线速度和角速度
4. 在 Isaac Sim 中回放，记录全身运动学（29 个 link 的位置/朝向/速度 + 24 个关节位置/速度）
5. 输出 `.npz` 文件

### 4.2 配置动作文件路径

修改 `source/engineai_rl_lab/engineai_rl_lab/tasks/mimic/robots/pm01/pm01_minic.py` 第 34 行：

```python
# 将路径改为你实际的 NPZ 文件路径
motion_file_ = "/your/path/to/motion_data/minic/dance1_subject1.npz"
```

### 4.3 启动训练

```bash
# 方式一：通过入口脚本
source engineai_rl_lab.sh -t EngineAI-PM01-Mimic

# 方式二：直接运行 Python（可传更多参数）
python scripts/rsl_rl/train.py \
    --task EngineAI-PM01-Mimic \
    --headless \
    --num_envs 4096 \
    --max_iterations 30000
```

可选参数：
- `--num_envs N`：并行环境数量（默认 4096）
- `--max_iterations N`：最大训练轮数（默认 30000）
- `--device cuda:0`：指定 GPU
- `--seed N`：随机种子
- `--distributed`：多 GPU 分布式训练
- `--video`：训练过程中录制视频

### 4.4 训练过程

训练过程中会自动：
- 每 1000 步保存一个 checkpoint（`model_*.pt`）
- 输出到 `logs/rsl_rl/engineai_pm01_mimic/<timestamp>/`
- 训练结束后自动导出 `deploy.yaml` 到 `params/` 目录

**自适应采样机制**：`MotionCommand` 会将动作序列分 bin，跟踪每个 bin 的失败率，对困难片段进行过采样，加速学习。

### 4.5 恢复训练

```bash
python scripts/rsl_rl/train.py \
    --task EngineAI-PM01-Mimic \
    --headless \
    --resume \
    --load_run <run_dir_name> \
    --load_checkpoint model_20000.pt
```

### 4.6 推理回放与策略导出

```bash
# 方式一：通过入口脚本
source engineai_rl_lab.sh -p EngineAI-PM01-Mimic \
    --checkpoint logs/rsl_rl/engineai_pm01_mimic/<timestamp>/model_30000.pt

# 方式二：直接运行
python scripts/rsl_rl/play.py \
    --task EngineAI-PM01-Mimic \
    --checkpoint logs/rsl_rl/engineai_pm01_mimic/<timestamp>/model_30000.pt
```

可选参数：
- `--real-time`：实时运行
- `--video`：录制回放视频
- `--num_envs N`：回放环境数

该脚本会：
1. 加载 checkpoint
2. 运行仿真回放
3. 自动导出策略到 `<checkpoint_dir>/exported/`：
   - `policy.onnx` — ONNX 格式（通用部署）
   - `policy.pt` — JIT 格式（PyTorch 部署）

### 4.7 可视化动作数据（不训练）

```bash
python scripts/mimic/replay_npz_pm01.py \
    --motion_file motion_data/minic/dance1_subject1.npz
```

### 4.8 实机部署

训练完成后，`logs/rsl_rl/engineai_pm01_mimic/<timestamp>/params/deploy.yaml` 包含部署所需的所有信息：

| 字段 | 说明 |
|------|------|
| `body_names` | 身体部位名称列表 |
| `step_dt` | 控制时间步（0.02s = 50Hz） |
| `default_joint_pos` | 默认关节位置（弧度） |
| `actions.*.scale` | 每个关节的动作缩放系数 |
| `actions.*.offset` | 每个关节的动作偏移量 |
| `actions.*.clip` | 动作裁剪范围 |
| `observations.*.scale` | 每个观测项的缩放系数 |

部署时将 `policy.onnx`（或 `policy.pt`）+ `deploy.yaml` 加载到实机控制器即可。

---

## 5. 目录结构总览

```
pm01_minic/
├── engineai_rl_lab.sh                  # 主入口脚本
├── pyproject.toml                      # 构建配置
├── serial_links.xml                    # MuJoCo 运动学模型
├── PROJECT_DOC.md                      # 本文档
│
├── source/engineai_rl_lab/             # 核心 Python 包
│   ├── config/extension.toml           # Isaac Lab 扩展元数据
│   ├── setup.py                        # pip 安装配置
│   └── engineai_rl_lab/
│       ├── assets/robots/
│       │   ├── pm01_engineai.py        # PM01 机器人定义
│       │   └── serial_pm_v2/           # USD 模型文件
│       ├── tasks/mimic/
│       │   ├── agents/
│       │   │   └── rsl_rl_ppo_cfg.py   # PPO 超参数
│       │   ├── robots/pm01/
│       │   │   ├── __init__.py         # 环境注册
│       │   │   └── pm01_minic.py       # 核心环境配置
│       │   └── mdp/
│       │       ├── commands.py         # 运动指令 + 自适应采样
│       │       ├── observations.py     # 观测函数
│       │       ├── rewards.py          # 奖励函数
│       │       ├── events.py           # 域随机化
│       │       └── terminations.py     # 终止条件
│       └── utils/
│           ├── export_deploy_cfg.py    # 导出部署配置
│           └── parser_cfg.py           # 配置解析
│
├── scripts/
│   ├── list_envs.py                    # 列出环境
│   ├── rsl_rl/
│   │   ├── train.py                    # 训练入口
│   │   ├── play.py                     # 推理 + 导出
│   │   ├── cli_args.py                 # CLI 参数
│   │   └── export_IODescriptors.py     # IO 描述导出
│   └── mimic/
│       ├── csv_to_npz.py              # CSV→NPZ (G1)
│       ├── csv_to_npz_pm01.py         # CSV→NPZ (PM01)
│       ├── replay_npz.py              # NPZ 回放 (G1)
│       └── replay_npz_pm01.py         # NPZ 回放 (PM01)
│
├── motion_data/                        # 动作数据
│   ├── minic/                          # minic 动捕数据
│   ├── gvhmr/                          # GVHMR 视频动捕数据
│   └── walk_0226/                      # 测试数据
│
└── logs/rsl_rl/engineai_pm01_mimic/    # 训练产物
    └── <timestamp>/
        ├── model_*.pt                  # checkpoints
        ├── exported/
        │   ├── policy.onnx             # ONNX 策略
        │   └── policy.pt               # JIT 策略
        └── params/
            ├── env.yaml                # 环境配置
            ├── agent.yaml              # PPO 配置
            └── deploy.yaml             # 部署配置
```
