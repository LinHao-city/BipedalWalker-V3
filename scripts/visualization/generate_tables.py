#!/usr/bin/env python3
"""
BipedalWalker Performance Tables Generator

This script creates comprehensive performance comparison tables
from all training result files in evaluation_results/tables/

Usage:
    python scripts/visualization/generate_tables.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_training_data(file_path):
    """Load training data from special format file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Line 1: episodes, Line 2: scores
    episodes = [float(x) for x in lines[0].strip().split()]
    scores = [float(x) for x in lines[1].strip().split()]

    return np.array(episodes), np.array(scores)

def parse_filename_info(filename):
    """Parse model information from filename"""
    parts = filename.replace('train-hardcore-', '').replace('.txt', '').split('-')

    info = {
        'algorithm': 'unknown',
        'architecture': 'unknown',
        'history_length': 0
    }

    for part in parts:
        if 'sac' in part:
            info['algorithm'] = 'SAC'
        elif 'td3' in part:
            info['algorithm'] = 'TD3'
        elif 'lstm' in part:
            info['architecture'] = 'LSTM'
        elif 'trsf' in part:
            info['architecture'] = 'Transformer'
        elif 'ff' in part:
            info['architecture'] = 'FeedForward'
        elif part.isdigit():
            info['history_length'] = int(part)

    return info

def find_convergence_episode(scores, window=100, threshold=10.0):
    """Find the episode where training stabilizes"""
    if len(scores) < window:
        return len(scores)

    rolling_std = pd.Series(scores).rolling(window=window).std()
    stable_episodes = np.where(rolling_std < threshold)[0]

    if len(stable_episodes) > 0:
        return stable_episodes[0] + window

    return len(scores)

def generate_performance_table():
    """Generate complete performance comparison table"""
    print("Generating comprehensive performance tables...")

    # Find all training files
    training_files = list(Path("results/logs").glob("train-hardcore-*.txt"))

    results = []

    for file_path in training_files:
        try:
            # Load and analyze data
            episodes, scores = load_training_data(file_path)
            file_info = parse_filename_info(file_path.name)

            # Calculate metrics
            final_score = float(scores[-1])
            max_score = float(np.max(scores))
            mean_last_100 = float(np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores))
            convergence_ep = int(find_convergence_episode(scores))
            total_episodes = int(len(episodes))

            # Create model name
            model_name = f"{file_info['algorithm']}_{file_info['architecture']}"
            if file_info['history_length'] > 0:
                model_name += f"-{file_info['history_length']}"

            results.append({
                'Model': model_name,
                'Algorithm': file_info['algorithm'],
                'Architecture': file_info['architecture'],
                'History_Length': file_info['history_length'],
                'Final_Score': final_score,
                'Max_Score': max_score,
                'Mean_Last_100': mean_last_100,
                'Convergence_Episode': convergence_ep,
                'Total_Episodes': total_episodes
            })

            print(f"Processed: {model_name} - Final: {final_score:.1f}, Max: {max_score:.1f}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if not results:
        print("No valid results found!")
        return

    # Create DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values('Final_Score', ascending=False)
    df.insert(0, 'Rank', range(1, len(df) + 1))

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV table
    output_dir = Path("evaluation_results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"performance_summary_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV table saved: {csv_path}")

    # Save Markdown table
    md_path = output_dir / f"performance_summary_{timestamp}.md"
    save_markdown_table(df, md_path)
    print(f"Markdown table saved: {md_path}")

    # Print summary
    print(f"\nGenerated performance table with {len(df)} models")
    print(f"Best model: {df.iloc[0]['Model']} (Score: {df.iloc[0]['Final_Score']:.1f})")
    print(f"Average score: {df['Final_Score'].mean():.1f} ± {df['Final_Score'].std():.1f}")

def save_markdown_table(df, file_path):
    """Save DataFrame as formatted Markdown table"""
    with open(file_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# BipedalWalker Performance Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Performance Ranking
        f.write("## Performance Ranking\n\n")
        f.write("| Rank | Model | Algorithm | Architecture | History Length | Final Score | Max Score | Mean Last 100 | Convergence Episode | Total Episodes |\n")
        f.write("|------|-------|-----------|-------------|---------------|-------------|-----------|---------------|-------------------|---------------|\n")

        for _, row in df.iterrows():
            f.write(f"| {row['Rank']} | {row['Model']} | {row['Algorithm']} | {row['Architecture']} | ")
            f.write(f"{row['History_Length']} | {row['Final_Score']:.1f} | {row['Max_Score']:.1f} | ")
            f.write(f"{row['Mean_Last_100']:.1f} | {row['Convergence_Episode']} | {row['Total_Episodes']} |\n")

        # Statistics Summary
        f.write("\n## Statistics Summary\n\n")
        f.write(f"- **Total Models**: {len(df)}\n")
        f.write(f"- **Average Final Score**: {df['Final_Score'].mean():.2f} ± {df['Final_Score'].std():.2f}\n")
        f.write(f"- **Average Max Score**: {df['Max_Score'].mean():.2f} ± {df['Max_Score'].std():.2f}\n")
        f.write(f"- **Highest Score**: {df['Max_Score'].max():.2f}\n")
        f.write(f"- **Best Model**: {df.iloc[0]['Model']} ({df.iloc[0]['Final_Score']:.2f})\n")

        # Algorithm Comparison
        f.write("\n### Algorithm Comparison\n\n")
        algo_stats = df.groupby('Algorithm')['Final_Score'].agg(['mean', 'std', 'count'])
        for algo in algo_stats.index:
            mean_score = algo_stats.loc[algo, 'mean']
            std_score = algo_stats.loc[algo, 'std']
            count = algo_stats.loc[algo, 'count']
            f.write(f"- **{algo}**: {mean_score:.2f} ± {std_score:.2f} ({count} models)\n")

        # Architecture Comparison
        f.write("\n### Architecture Comparison\n\n")
        arch_stats = df.groupby('Architecture')['Final_Score'].agg(['mean', 'std', 'count'])
        for arch in arch_stats.index:
            mean_score = arch_stats.loc[arch, 'mean']
            std_score = arch_stats.loc[arch, 'std']
            count = arch_stats.loc[arch, 'count']
            f.write(f"- **{arch}**: {mean_score:.2f} ± {std_score:.2f} ({count} models)\n")

        # Top Models
        f.write("\n### Top 3 Models\n\n")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            f.write(f"{i+1}. **{row['Model']}**: {row['Final_Score']:.1f} (max: {row['Max_Score']:.1f})\n")

if __name__ == "__main__":
    generate_performance_table()