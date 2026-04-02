#!/usr/bin/env python3
# Record FF models only with fixed wrapper reset issue

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import subprocess
from collections import deque

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import gymnasium as gym
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from architectures.feedforward import Actor, Critic

# FF models only
FF_MODELS = [
    {"algorithm": "sac", "architecture": "ff", "checkpoint": "ep7600", "name": "SAC_FeedForward"},
    {"algorithm": "td3", "architecture": "ff", "checkpoint": "ep6600", "name": "TD3_FeedForward"},
]

# Original MyWalkerWrapper for FF models
class OriginalMyWalkerWrapper:
    def __init__(self, env, skip=2):
        self.env = env
        self._obs_buffer = deque(maxlen=skip)
        self._skip = skip
        self._max_episode_steps = 750
        self._is_gymnasium = True

    def reset(self):
        reset_result = self.env.reset()
        # Handle new version gym return format (obs, info)
        if isinstance(reset_result, tuple):
            return reset_result[0]  # Only return observation part
        else:
            return reset_result  # Old version returns observation directly

    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False

        for i in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                terminated = False
                truncated = False
                done = done

            if self.env.game_over:
                reward = -10.0
                info["dead"] = True
            else:
                info["dead"] = False

            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        # Return correct format based on environment type
        if self._is_gymnasium:
            return obs, total_reward, terminated, truncated, info
        else:
            return obs, total_reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    # Keep all original environment attributes
    def __getattr__(self, name):
        return getattr(self.env, name)

def create_agent(model_config):
    """Create FF agent"""
    ActorClass, CriticClass = Actor, Critic
    state_size = 24  # BipedalWalkerHardcore-v3 original state dimension
    action_size = 4  # BipedalWalkerHardcore-v3 action dimension

    if model_config["algorithm"] == "sac":
        agent = SACAgent(ActorClass, CriticClass, clip_low=-1, clip_high=+1,
                       state_size=state_size, action_size=action_size)
    elif model_config["algorithm"] == "td3":
        agent = TD3Agent(ActorClass, CriticClass, clip_low=-1, clip_high=+1,
                       state_size=state_size, action_size=action_size)
    else:
        raise ValueError(f"Unknown algorithm: {model_config['algorithm']}")

    # Load checkpoint
    env_type = "hardcore"
    agent.load_ckpt(model_config["architecture"], env_type, model_config["checkpoint"])
    agent.eval_mode()

    return agent

def convert_frames_to_mp4_h264(frames_dir, output_path, fps=50):
    """Convert image sequence to H264 MP4 using FFmpeg"""
    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")

    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Overwrite output file
        '-framerate', str(fps),
        '-i', frame_pattern,
        '-c:v', 'libx264',  # H.264 encoder
        '-preset', 'medium',   # Encoding speed vs quality balance
        '-pix_fmt', 'yuv420p',  # Most compatible pixel format
        '-crf', '25',  # Quality factor
        '-r', str(fps),  # Output frame rate
        output_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"    [SUCCESS] FFmpeg H264 conversion completed")

        # Verify generated file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_fps = test_cap.get(cv2.CAP_PROP_FPS)
                fourcc = int(test_cap.get(cv2.CAP_PROP_FOURCC))
                codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                test_cap.release()
                print(f"    [VERIFIED] {test_frames} frames, {test_fps:.1f}FPS, {codec} codec")
                return True
            else:
                print(f"    [FAILED] Created video cannot be opened")
                return False
        else:
            print(f"    [FAILED] No output file created")
            return False

    except subprocess.CalledProcessError as e:
        print(f"    [ERROR] FFmpeg failed: {e}")
        return False
    except FileNotFoundError:
        print(f"    [ERROR] FFmpeg not found")
        return False

def record_ff_models():
    """Record FF model videos"""
    print("=" * 70)
    print("Recording FF Models with Fixed Wrappers")
    print("=" * 70)

    results = {}
    video_dir = "evaluation_results/videos/real_bipedalwalker_all"

    for model_config in FF_MODELS:
        model_name = model_config["name"]
        print(f"\nProcessing: {model_name}")
        print("-" * 50)

        try:
            # Create agent
            agent = create_agent(model_config)
            print(f"Agent loaded successfully")

            # Create video directory
            model_video_dir = os.path.join(video_dir, model_name)
            os.makedirs(model_video_dir, exist_ok=True)

            scores = []

            # Run 2 episodes and record
            for episode in range(2):
                # Create original Box2D environment
                base_env = gym.make('BipedalWalkerHardcore-v3', render_mode='rgb_array')

                # Add Walker wrapper only (no history for FF models)
                env = OriginalMyWalkerWrapper(base_env, skip=2)

                # Create frame save directory
                frames_dir = os.path.join(model_video_dir, f"frames_episode_{episode+1}")
                os.makedirs(frames_dir, exist_ok=True)

                print(f"  Episode {episode + 1}:")
                print(f"  Saving frames to: {frames_dir}")

                # Reset environment (FF models return only state)
                state = env.reset()
                info = {}
                print(f"  Initial state shape: {state.shape}")

                total_reward = 0
                done = False
                step = 0
                frame_count = 0

                while not done and step < 2000:
                    action = agent.get_action(state, explore=False)
                    action = np.clip(action, env.action_space.low, env.action_space.high)

                    # Environment step
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        state, reward, done, info = step_result
                    total_reward += reward

                    # Render and save frame
                    frame = env.render()
                    if frame is not None and frame.size > 0:
                        # Ensure frame is correct format
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            if frame.dtype != np.uint8:
                                frame = (frame * 255).astype(np.uint8)

                            # Save frame
                            frame_filename = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
                            cv2.imwrite(frame_filename, frame)
                            frame_count += 1

                    step += 1

                    # Print progress every 100 steps
                    if step % 100 == 0:
                        print(f"  Step {step:4d}, Score: {total_reward:6.2f}, Frames: {frame_count}")

                scores.append(total_reward)
                print(f"  Episode {episode + 1} completed! Score: {total_reward:.2f}, Frames: {frame_count}")

                # Convert to H264 MP4 using FFmpeg
                video_path = os.path.join(model_video_dir, f"{model_name}_episode_{episode+1}.mp4")
                print(f"  Converting to MP4 with FFmpeg...")

                if convert_frames_to_mp4_h264(frames_dir, video_path, fps=50):
                    print(f"  [SUCCESS] Video created: {video_path}")

                    # Clean up frame files
                    import shutil
                    shutil.rmtree(frames_dir)
                    print(f"  [INFO] Cleaned up frame files")
                else:
                    print(f"  [FAILED] Failed to create video, frames saved at: {frames_dir}")

                # Close environment
                env.close()

            avg_score = sum(scores) / len(scores)
            print(f"[COMPLETED] {model_name} recording completed! Average Score: {avg_score:.2f}")

            results[model_name] = avg_score

        except Exception as e:
            print(f"[ERROR] Failed to record video for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = None

    # Print summary
    print("\n" + "=" * 70)
    print("FF Models Recording Summary:")
    print("=" * 70)

    successful_results = {k: v for k, v in results.items() if v is not None}
    failed_results = {k: v for k, v in results.items() if v is None}

    if successful_results:
        print("\nSuccessfully recorded videos:")
        for model, avg_score in successful_results.items():
            print(f"   {model}: {avg_score:.2f}")

    if failed_results:
        print("\nFailed to record:")
        for model in failed_results:
            print(f"   {model}")

    print(f"\nSuccess rate: {len(successful_results)}/{len(FF_MODELS)}")
    print(f"Videos saved to: {video_dir}")

if __name__ == "__main__":
    record_ff_models()