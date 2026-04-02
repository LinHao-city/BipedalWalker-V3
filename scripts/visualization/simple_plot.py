#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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

def create_simple_plot():
    """Create a simple plot with available training data"""
    plt.figure(figsize=(15, 10))

    # Find training result files - use only 'train-' files for consistency
    result_files = list(Path("results/logs").glob("train-hardcore-*.txt"))

    if not result_files:
        print("No training result files found!")
        return

    print(f"Found {len(result_files)} training files")

    # Define colors for different model types
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    plt.figure(figsize=(15, 10))

    all_scores = []
    all_episodes = []

    for i, file_path in enumerate(result_files):
        try:
            episodes, scores = load_training_data(file_path)

            # Get model name from filename
            model_name = file_path.stem.replace('train-hardcore-', '')

            print(f"Processing: {model_name}")
            print(f"  Episodes range: {episodes[0]:.0f} - {episodes[-1]:.0f}")
            print(f"  Score range: {np.min(scores):.1f} - {np.max(scores):.1f}")

            # Calculate moving average
            window_size = min(100, len(scores))
            if window_size > 0:
                ma_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
                ma_episodes = episodes[window_size-1:]

                plt.plot(ma_episodes, ma_scores, color=colors[i % len(colors)],
                        linewidth=2, label=model_name, alpha=0.8)

                all_scores.extend(scores)
                all_episodes.extend(episodes)

                print(f"  Final Score: {scores[-1]:.1f}, Max: {np.max(scores):.1f}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if all_scores:
        # Set reasonable axis limits
        plt.xlim(0, max(all_episodes) * 1.05)
        plt.ylim(min(all_scores) * 1.1, max(all_scores) * 1.1)

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('BipedalWalker Training Results', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)

    # Save to evaluation_results
    output_dir = Path("evaluation_results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'training_plot.png', dpi=300, bbox_inches='tight')
    plt.close()  # Don't show, just save

    print(f"\nPlot saved to: {output_dir / 'training_plot.png'}")
    print(f"Total models plotted: {len(result_files)}")

if __name__ == "__main__":
    create_simple_plot()