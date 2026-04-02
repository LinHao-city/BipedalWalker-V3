# 🤖 BipedalWalker Reinforcement Learning Project

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28%2B-green.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于深度强化学习的双足步行机器人控制项目，使用SAC和TD3算法解决BipedalWalker环境挑战。

## 📋 项目概述

本项目实现了一个完整的强化学习系统，专门用于解决**BipedalWalker-v3**和**BipedalWalkerHardcore-v3**环境。项目采用了较为前沿的深度强化学习算法和创新的神经网络架构，实现了在复杂地形下的双足行走控制。

### 🎯 核心目标
- 解决BipedalWalkerHardcore-v3环境的挑战
- 比较不同强化学习算法的性能表现
- 探索时序建模在机器人控制中的应用
- 提供可复现的实验结果和详细的实现

## ✨ 项目特色

### 🧠 先进的强化学习算法
- **SAC (Soft Actor-Critic)**: 基于最大熵的演员-评论家算法
- **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**: 改进的确定性策略梯度算法

### 🏗️ 创新的神经网络架构
- **FeedForward网络**: 基础前馈神经网络架构
- **Transformer模型**: 利用注意力机制处理时序信息
- **LSTM**: 长短期记忆网络处理历史观察
- **时序建模**: 支持6步或12步历史观察输入

## 📁 项目结构

```
BipedalWalker-V3/
├── 📂 src/                          # 核心源代码（模型训练）
│   ├── 📄 agents/                    # 强化学习智能体
│   │   ├── sac_agent.py             # SAC算法实现
│   │   └── td3_agent.py             # TD3算法实现
│   ├── 📄 architectures/            # 神经网络架构
│   │   ├── feedforward.py           # 前馈网络
│   │   ├── transformer.py           # Transformer模型
│   │   └── lstm.py                  # LSTM模型
│   ├── 📄 utils/                     # 工具模块
│   │   ├── replay_buffer.py         # 经验回放缓冲区
│   │   └── noise.py                 # 噪声策略
│   ├── 📄 training.py               # 训练逻辑
│   └── 📄 evaluation.py             # 评估逻辑
├── 📂 scripts/                      # 现代化脚本
│   └── 📂 visualization/            # 可视化和生成工具
│       ├── simple_plot.py           # 主训练曲线图生成
│       ├── plot_results.py          # 专门对比图生成
│       ├── generate_tables.py       # 性能表格生成
│       ├── record_ff_final.py       # FeedForward模型视频生成
│       └── record_original_wrappers_fixed.py # 全模型视频生成
├── 📂 models/                       # 预训练模型
├── 📂 results/                      # 原始实验结果
│   ├── 📄 logs/                     # 训练日志
│   └── 📄 plots/                    # 原始结果图表
├── 📂 evaluation_results/           # 现代化评估结果
│   ├── 📄 plots/                    # 生成的对比图表
│   ├── 📄 tables/                   # 性能统计表格
│   └── 📄 videos/                   # 生成的视频文件
├── 📄 requirements.txt              # 依赖包列表
├── 📄 main_script.py                # 训练脚本
└── 📄 README.md                    # 项目说明
```

## 🚀 快速开始

### 环境要求

直接安装项目依赖：
```bash
pip install -r requirements.txt
```

### 📹 视频生成（推荐使用）

生成所有模型的H264 MP4视频：

#### 生成所有模型视频
```bash
# 生成所有8个模型（16个视频文件，每个模型2个episodes）
python scripts/visualization/record_original_wrappers_fixed.py
```

#### 生成特定类型模型视频
```bash
# 生成FeedForward模型视频
python scripts/visualization/record_ff_final.py

# 生成LSTM和Transformer模型视频
python scripts/visualization/record_original_wrappers_fixed.py
```

### 📊 数据分析和可视化

#### 生成主要训练曲线图
```bash
# 生成包含8种模型的完整训练对比图
python scripts/visualization/simple_plot.py
```

#### 生成专门对比分析图
```bash
# 生成所有专门的对比图
python scripts/visualization/plot_results.py

# 或生成特定类型的对比图
python scripts/visualization/plot_results.py --sac_vs_td3      # SAC vs TD3算法对比
python scripts/visualization/plot_results.py --sac_archs      # SAC架构对比
python scripts/visualization/plot_results.py --td3_archs      # TD3架构对比
```

#### 生成性能统计表格
```bash
# 生成完整的性能对比表格（CSV和Markdown格式）
python scripts/visualization/generate_tables.py
```

### 🏃‍♂️ 模型训练

模型训练使用代码：
```bash
# 训练SAC + FeedForward
python main_script.py -f train -r sac -m ff

# 训练SAC + LSTM (6步历史)
python main_script.py -f train -r sac -m lstm -hl 6

# 训练SAC + LSTM (12步历史)
python main_script.py -f train -r sac -m lstm -hl 12

# 训练SAC + Transformer (6步历史)
python main_script.py -f train -r sac -m trsf -hl 6

# 训练TD3 + FeedForward
python main_script.py -f train -r td3 -m ff

# 训练TD3 + LSTM (6步历史)
python main_script.py -f train -r td3 -m lstm -hl 6

# 训练TD3 + Transformer (6步历史)
python main_script.py -f train -r td3 -m trsf -hl 6
```

## 📊 实验结果

### 生成的分析文件

所有现代化分析结果都保存在 `evaluation_results/` 目录下：

#### 📈 图表文件 (`evaluation_results/plots/`)
- `training_plot.png` - 8种模型的完整训练对比图
- `SAC_vs_TD3_comparison.png` - SAC vs TD3算法性能对比
- `SAC_Architectures.png` - SAC不同架构性能对比
- `TD3_Architectures.png` - TD3不同架构性能对比

#### 📋 表格文件 (`evaluation_results/tables/`)
- `performance_summary_YYYYMMDD_HHMMSS.csv` - 性能对比表格（CSV格式）
- `performance_summary_YYYYMMDD_HHMMSS.md` - 性能对比报告（Markdown格式）

#### 📹 视频文件 (`evaluation_results/videos/`)
- **格式**: H264 MP4，50 FPS，yuv420p像素格式
- **兼容性**: 可在VSCode中直接播放
- **命名规则**: `{ModelName}_episode_{1|2}.mp4`

### 🏆 性能对比结果

根据最新实验结果（8个模型）：

| 排名 | 模型 | 算法 | 架构 | 最终分数 | 最高分数 |
|------|------|------|------|----------|----------|
| 1 | SAC_Transformer-12 | SAC | Transformer | 303.6 | 309.3 |
| 2 | TD3_LSTM-6 | TD3 | LSTM | 303.0 | 313.0 |
| 3 | SAC_LSTM-12 | SAC | LSTM | 302.4 | 314.6 |
| 4 | TD3_FeedForward | TD3 | FeedForward | 282.4 | 296.4 |
| 5 | TD3_Transformer-6 | TD3 | Transformer | 276.9 | 303.6 |
| 6 | SAC_FeedForward | SAC | FeedForward | 252.1 | 308.1 |
| 7 | SAC_LSTM-6 | SAC | LSTM | 213.9 | 322.2 |
| 8 | SAC_Transformer-6 | SAC | Transformer | 73.7 | 303.4 |

### 🔍 关键发现

1. **算法对比**: TD3算法平均表现更稳定 (287.5 ± 13.7)，SAC算法方差较大 (229.2 ± 94.7)
2. **架构效果**: LSTM-12和Transformer-12架构表现最佳
3. **时序建模**: 12步历史观察在LSTM架构中表现最好
4. **训练稳定性**: TD3_LSTM-6收敛最快 (708 episodes)

## 🔧 技术实现

### 核心算法特点
- **SAC**: 结合最大熵强化学习，提高探索效率
- **TD3**: 双Q网络和延迟更新，稳定训练过程
- **优先级回放**: 基于TD-error的经验回放策略

### 现代化分析工具
- **数据可视化**: matplotlib + seaborn 高级图表
- **性能分析**: pandas 数据处理和统计分析
- **视频生成**: FFmpeg H264编码，VSCode兼容

## 🔧 故障排除

### 常见问题

1. **FFmpeg未找到错误**
   ```bash
   conda install -c conda-forge ffmpeg -y
   ```

2. **Gymnasium版本问题**
   ```bash
   pip uninstall gym
   pip install gymnasium
   pip install Box2D
   ```

3. **视频无法在VSCode中播放**
   - 确保视频使用H264编码
   - 检查像素格式是否为yuv420p

4. **依赖包问题**
   ```bash
   pip install -r requirements.txt
   ```

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件