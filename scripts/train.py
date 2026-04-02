#!/usr/bin/env python3
"""
BipedalWalker Training Script - Enhanced Wrapper
双足步行机器人训练脚本 - 增强版包装器

这是一个包装脚本，简化了训练流程，同时保持原始训练逻辑不变。
支持更友好的命令行参数和可选的配置文件。

Usage:
    # 基础训练
    python scripts/train.py --algorithm sac --architecture feedforward

    # 使用历史序列模型
    python scripts/train.py --algorithm sac --architecture lstm --history_length 12

    # TD3算法训练
    python scripts/train.py --algorithm td3 --architecture transformer --history_length 6

    # 原始参数映射到新参数
    python scripts/train.py -r sac -m ff  # 映射到 --algorithm sac --architecture feedforward
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import original components
import gymnasium as gym
from training import train, test
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from agents.ddpg_agent import DDPGAgent
from utils.env_wrappers import MyWalkerWrapper, BoxToHistoryBox

def get_architecture_imports(model_type):
    """根据模型类型动态导入对应的Actor和Critic类"""
    import_map = {
        'ff': ('archs.ff_models', 'Actor', 'Critic'),
        'feedforward': ('archs.ff_models', 'Actor', 'Critic'),
        'mlp': ('archs.mlp_models', 'Actor', 'Critic'),
        'rnn': ('archs.rnn_models', 'Actor', 'Critic'),
        'lstm': ('archs.lstm_models', 'Actor', 'Critic'),
        'bilstm': ('archs.bilstm_models', 'Actor', 'Critic'),
        'trsf': ('archs.trsf_models', 'Actor', 'Critic'),
        'transformer': ('archs.trsf_models', 'Actor', 'Critic'),
    }

    if model_type not in import_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(import_map.keys())}")

    module_name, actor_name, critic_name = import_map[model_type]

    # 动态导入
    module = __import__(module_name, fromlist=[actor_name, critic_name])
    Actor = getattr(module, actor_name)
    Critic = getattr(module, critic_name)

    return Actor, Critic

def create_environment(env_type, model_type, history_length):
    """创建环境，与原始逻辑完全相同"""
    if env_type == 'classic':
        env = gym.make('BipedalWalker-v3')
        env = MyWalkerWrapper(env, skip=2)
    elif env_type == 'hardcore':
        env = gym.make('BipedalWalkerHardcore-v3')
        env = MyWalkerWrapper(env, skip=2)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    # 为序列模型添加历史包装器
    if model_type in ['rnn', 'lstm', 'bilstm', 'trsf', 'transformer']:
        env = BoxToHistoryBox(env, h=history_length)
        env_type_str = f"{env_type}-{history_length}"
    else:
        env_type_str = env_type

    return env, env_type_str

def create_agent(rl_type, Actor, Critic, env, args):
    """创建智能体，与原始逻辑完全相同"""
    state_size = env.observation_space.shape[-1]
    action_size = env.action_space.shape[-1]

    device = args.device if args.device else None

    if rl_type == 'ddpg':
        agent = DDPGAgent(Actor, Critic,
                         state_size=state_size,
                         action_size=action_size,
                         lr=args.lr,
                         weight_decay=args.wd,
                         gamma=args.gamma,
                         batch_size=args.batch_size,
                         device=device)
    elif rl_type == 'td3':
        agent = TD3Agent(Actor, Critic, clip_low=-1, clip_high=+1,
                        state_size=state_size,
                        action_size=action_size,
                        lr=args.lr,
                        weight_decay=args.wd,
                        gamma=args.gamma,
                        batch_size=args.batch_size,
                        device=device)
    elif rl_type == 'sac':
        agent = SACAgent(Actor, Critic, clip_low=-1, clip_high=+1,
                        state_size=state_size,
                        action_size=action_size,
                        lr=args.lr,
                        weight_decay=args.wd,
                        gamma=args.gamma,
                        alpha=args.alpha,
                        batch_size=args.batch_size,
                        device=device)
    else:
        raise ValueError(f"Unknown RL algorithm type: {rl_type}")

    return agent

def print_training_info(args, env, agent):
    """打印训练信息"""
    print("=" * 70)
    print("🚀 BipedalWalker Enhanced Training Started")
    print("=" * 70)
    print(f"📊 Algorithm: {args.rl_type.upper()}")
    print(f"🧠 Architecture: {args.model_type}")
    print(f"🎮 Environment: {args.env}")
    if hasattr(args, 'history_length') and args.history_length:
        print(f"📚 History Length: {args.history_length}")
    print(f"💻 Device: {agent.device}")
    print(f"🎯 Learning Rate: {args.lr}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"🎲 Explore Episodes: {args.explore_episode}")
    print("=" * 70)
    print(f"🔧 Action Space: {env.action_space.shape}")
    print(f"🔍 State Space: {env.observation_space.shape}")
    print(f"📈 Checkpoint: {args.ckpt}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced BipedalWalker Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --algorithm sac --architecture feedforward
  %(prog)s -r sac -m ff  # Short form
  %(prog)s --algorithm td3 --architecture lstm --history_length 12
  %(prog)s -r td3 -m lstm -hl 12  # Short form
        """
    )

    # 保持与原始参数的兼容性
    parser.add_argument("-f", "--flag", type=str, choices=['train', 'test', 'test-record', 'test-exp', 'test-100'],
                        default='train', help="train or test? (原始参数兼容)")

    # 新的友好参数
    parser.add_argument("--algorithm", "-r", type=str,
                        choices=['ddpg', 'td3', 'sac'],
                        default='sac', help='RL algorithm')
    parser.add_argument("--architecture", "-m", type=str,
                        choices=['ff','mlp','rnn','lstm','bilstm','trsf','feedforward','transformer'],
                        default='ff', help='Model type')
    parser.add_argument("--environment", "-e", type=str, choices=['classic', 'hardcore'],
                        default='hardcore', help="Environment type")

    # 训练参数
    parser.add_argument("--learning_rate", "-l", type=float, default=4e-4, help='Learning Rate')
    parser.add_argument("--weight_decay", "-w", type=float, default=0, help='Weight Decay')
    parser.add_argument("--batch_size", "-b", type=int, default=64, help='Batch size')
    parser.add_argument("--checkpoint", "-c", type=str, default='seed', help='Checkpoint to start with')
    parser.add_argument("--explore_episode", "-x", type=int, default=50, help='Number of exploration steps')
    parser.add_argument("--gamma", "-g", type=float, default=0.98, help='Discount rate')
    parser.add_argument("--alpha", "-a", type=float, default=0.01, help='Entropy regularization term in SAC')
    parser.add_argument("--history_length", "-hl", type=int, default=12, help='History length for sequential models')
    parser.add_argument("--device", "-d", type=str, default=None, help='Device to train on')

    # 新增的便利参数
    parser.add_argument("--max_episodes", type=int, default=8000, help='Maximum training episodes')
    parser.add_argument("--test_frequency", type=int, default=200, help='Test evaluation frequency')
    parser.add_argument("--save_results", action='store_true', help='Save training results to file')
    parser.add_argument("--verbose", "-v", action='store_true', help='Verbose output')

    args = parser.parse_args()

    # 为了向后兼容，设置原始参数名
    args.rl_type = args.algorithm
    args.model_type = args.architecture
    args.env = args.environment
    args.lr = args.learning_rate
    args.wd = args.weight_decay
    args.ckpt = args.checkpoint

    try:
        # 动态导入架构
        Actor, Critic = get_architecture_imports(args.model_type)

        # 创建环境
        env, env_type_str = create_environment(args.env, args.model_type, args.history_length)

        # 创建智能体
        agent = create_agent(args.rl_type, Actor, Critic, env, args)

        # 加载检查点
        agent.load_ckpt(args.model_type, env_type_str, args.ckpt)

        # 打印训练信息
        if args.verbose:
            print_training_info(args, env, agent)

        # 如果是训练模式
        if args.flag == 'train':
            agent.train_mode()

            # 使用原始训练逻辑
            scores, test_scores = train(
                env, agent,
                n_episodes=args.max_episodes,
                model_type=args.model_type,
                env_type=env_type_str,
                explore_episode=args.explore_episode,
                test_f=args.test_frequency
            )

            if args.save_results:
                # 保存结果（保持原始格式）
                results_dir = Path("results")
                results_dir.mkdir(exist_ok=True)

                train_file = results_dir / f"train-{env_type_str}-{args.model_type}-{args.rl_type}.txt"
                test_file = results_dir / f"test-{env_type_str}-{args.model_type}-{args.rl_type}.txt"

                import numpy as np
                np.savetxt(train_file, scores, fmt="%.6e")
                np.savetxt(test_file, test_scores, fmt="%.6e")

                print(f"\n📊 Results saved:")
                print(f"   📄 Training: {train_file}")
                print(f"   📄 Testing: {test_file}")

        elif args.flag == 'test':
            # 测试模式（保持原始逻辑）
            agent.eval_mode()
            scores = test(env, agent, explore=False)

        elif args.flag == 'test-exp':
            # 测试模式，带探索
            agent.eval_mode()
            scores = test(env, agent, explore=True)

        elif args.flag == 'test-100':
            # 100轮测试
            agent.eval_mode()
            scores = test(env, agent, render=False, explore=False, n_times=100)

        else:
            print(f"❌ Unknown flag: {args.flag}")
            return 1

        print("\n✅ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        # 保存当前状态
        agent.save_ckpt(args.model_type, env_type_str, "interrupted")
        print("💾 Current state saved as 'interrupted'")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        env.close()

    return 0

if __name__ == "__main__":
    exit(main())