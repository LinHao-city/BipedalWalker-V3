#!/usr/bin/env python3
"""
BipedalWalker Specialized Results Visualization Tool

This script creates specialized comparison plots for the BipedalWalker project.
Focus: Creates algorithm and architecture comparison charts in evaluation_results/plots/

Usage:
    python scripts/visualization/plot_results.py                    # Generate all comparison plots
    python scripts/visualization/plot_results.py --sac_vs_td3      # Generate SAC vs TD3 comparison
    python scripts/visualization/plot_results.py --architectures   # Generate architecture comparison
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_training_data(file_path):
    """Load training data from special format file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Line 1: episodes, Line 2: scores
    episodes = [float(x) for x in lines[0].strip().split()]
    scores = [float(x) for x in lines[1].strip().split()]

    return np.array(episodes), np.array(scores)

def create_sac_vs_td3_comparison():
    """Create SAC vs TD3 algorithm comparison plot"""
    plt.figure(figsize=(12, 8))

    # Select representative models for comparison
    sac_files = [
        "results/logs/train-hardcore-12-lstm-sac.txt",
        "results/logs/train-hardcore-6-lstm-sac.txt",
        "results/logs/train-hardcore-ff-sac.txt"
    ]

    td3_files = [
        "results/logs/train-hardcore-6-lstm-td3.txt",
        "results/logs/train-hardcore-ff-td3.txt"
    ]

    colors_sac = ['red', 'darkred', 'salmon']
    colors_td3 = ['blue', 'darkblue', 'lightblue']

    # Plot SAC models
    for i, file_path in enumerate(sac_files):
        if Path(file_path).exists():
            episodes, scores = load_training_data(file_path)
            model_name = Path(file_path).stem.replace('train-hardcore-', '').replace('-sac', '')

            # Calculate moving average
            window_size = min(100, len(scores))
            if window_size > 0:
                ma_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
                ma_episodes = episodes[window_size-1:]
                plt.plot(ma_episodes, ma_scores, color=colors_sac[i], linewidth=2,
                        label=f'SAC-{model_name}', alpha=0.8)

    # Plot TD3 models
    for i, file_path in enumerate(td3_files):
        if Path(file_path).exists():
            episodes, scores = load_training_data(file_path)
            model_name = Path(file_path).stem.replace('train-hardcore-', '').replace('-td3', '')

            # Calculate moving average
            window_size = min(100, len(scores))
            if window_size > 0:
                ma_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
                ma_episodes = episodes[window_size-1:]
                plt.plot(ma_episodes, ma_scores, color=colors_td3[i], linewidth=2,
                        label=f'TD3-{model_name}', alpha=0.8, linestyle='--')

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('SAC vs TD3 Algorithm Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save to evaluation_results
    output_dir = Path("evaluation_results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'SAC_vs_TD3_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SAC vs TD3 comparison saved: {output_dir / 'SAC_vs_TD3_comparison.png'}")

def create_sac_architectures_plot():
    """Create SAC architecture comparison plot"""
    plt.figure(figsize=(12, 8))

    sac_files = [
        "results/logs/train-hardcore-ff-sac.txt",
        "results/logs/train-hardcore-6-lstm-sac.txt",
        "results/logs/train-hardcore-12-lstm-sac.txt"
    ]

    colors = ['green', 'darkgreen', 'olive']

    for i, file_path in enumerate(sac_files):
        if Path(file_path).exists():
            episodes, scores = load_training_data(file_path)
            model_name = Path(file_path).stem.replace('train-hardcore-', '').replace('-sac', '')

            # Calculate moving average
            window_size = min(100, len(scores))
            if window_size > 0:
                ma_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
                ma_episodes = episodes[window_size-1:]
                plt.plot(ma_episodes, ma_scores, color=colors[i], linewidth=2,
                        label=f'SAC-{model_name}', alpha=0.8)

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('SAC: Architecture Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save to evaluation_results
    output_dir = Path("evaluation_results/plots")
    plt.savefig(output_dir / 'SAC_Architectures.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SAC architectures comparison saved: {output_dir / 'SAC_Architectures.png'}")

def create_td3_architectures_plot():
    """Create TD3 architecture comparison plot"""
    plt.figure(figsize=(12, 8))

    td3_files = [
        "results/logs/train-hardcore-ff-td3.txt",
        "results/logs/train-hardcore-6-lstm-td3.txt",
        "results/logs/train-hardcore-6-trsf-td3.txt"
    ]

    colors = ['purple', 'darkmagenta', 'violet']

    for i, file_path in enumerate(td3_files):
        if Path(file_path).exists():
            episodes, scores = load_training_data(file_path)
            model_name = Path(file_path).stem.replace('train-hardcore-', '').replace('-td3', '')

            # Calculate moving average
            window_size = min(100, len(scores))
            if window_size > 0:
                ma_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
                ma_episodes = episodes[window_size-1:]
                plt.plot(ma_episodes, ma_scores, color=colors[i], linewidth=2,
                        label=f'TD3-{model_name}', alpha=0.8)

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('TD3: Architecture Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save to evaluation_results
    output_dir = Path("evaluation_results/plots")
    plt.savefig(output_dir / 'TD3_Architectures.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"TD3 architectures comparison saved: {output_dir / 'TD3_Architectures.png'}")

def main():
    parser = argparse.ArgumentParser(description='BipedalWalker Specialized Visualization Tool')

    parser.add_argument('--sac_vs_td3', action='store_true',
                       help='Generate SAC vs TD3 comparison plot')
    parser.add_argument('--sac_archs', action='store_true',
                       help='Generate SAC architectures comparison plot')
    parser.add_argument('--td3_archs', action='store_true',
                       help='Generate TD3 architectures comparison plot')

    args = parser.parse_args()

    # If no specific plot requested, generate all
    if not any([args.sac_vs_td3, args.sac_archs, args.td3_archs]):
        print("Generating all specialized comparison plots...")
        create_sac_vs_td3_comparison()
        create_sac_architectures_plot()
        create_td3_architectures_plot()
        print("All specialized plots generated!")
    else:
        if args.sac_vs_td3:
            create_sac_vs_td3_comparison()
        if args.sac_archs:
            create_sac_architectures_plot()
        if args.td3_archs:
            create_td3_architectures_plot()

if __name__ == "__main__":
    main()