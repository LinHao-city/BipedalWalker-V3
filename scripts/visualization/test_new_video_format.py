#!/usr/bin/env python3
"""
快速测试新的视频格式生成
Quick test for new video format generation
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "scripts"))

from record_real_videos import convert_frames_to_mp4
import cv2
import numpy as np

def test_new_format():
    """测试新的H264格式生成"""
    print("=" * 60)
    print("测试新的H264视频格式（匹配原版）")
    print("=" * 60)

    # 创建测试帧目录
    test_frames_dir = "test_frames_h264"
    test_output = "test_h264_output.mp4"

    os.makedirs(test_frames_dir, exist_ok=True)

    # 生成50个测试帧（模拟BipedalWalker）
    fps = 50
    frame_size = (600, 400)

    print(f"生成测试帧: {len(frame_size)}, {fps} FPS")

    for i in range(50):
        # 创建模拟BipedalWalker环境的帧
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        # 添加背景（天空蓝）
        frame[:, :] = (135, 206, 235)  # Sky blue

        # 添加地面（棕色）
        frame[frame_size[1]//2:, :] = (139, 69, 19)  # Brown ground

        # 添加移动的机器人（简化表示）
        robot_x = int(frame_size[0] * (0.2 + i * 0.01))
        robot_y = frame_size[1]//2 - 20
        cv2.rectangle(frame, (robot_x-20, robot_y-40), (robot_x+20, robot_y), (255, 0, 0), -1)
        cv2.circle(frame, (robot_x, robot_y-50), 10, (255, 255, 0), -1)  # Head

        # 添加文字
        cv2.putText(frame, f'Test Frame {i+1}/50', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'FPS: {fps}, Codec: H264', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 保存帧
        frame_path = os.path.join(test_frames_dir, f"frame_{i:06d}.png")
        cv2.imwrite(frame_path, frame)

    print(f"生成了50个测试帧在 {test_frames_dir}")

    try:
        # 转换为MP4
        print(f"开始转换为新格式H264 MP4...")
        convert_frames_to_mp4(test_frames_dir, test_output, fps, frame_size)

        if os.path.exists(test_output):
            print(f"✅ 视频生成成功: {test_output}")

            # 分析生成的视频
            cap = cv2.VideoCapture(test_output)
            if cap.isOpened():
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

                file_size = os.path.getsize(test_output) / (1024 * 1024)
                duration = frame_count / actual_fps if actual_fps > 0 else 0

                print(f"📹 视频规格:")
                print(f"   分辨率: {width}x{height}")
                print(f"   帧率: {actual_fps:.1f} FPS")
                print(f"   帧数: {frame_count}")
                print(f"   编码器: {codec}")
                print(f"   时长: {duration:.2f}秒")
                print(f"   文件大小: {file_size:.2f} MB")

                cap.release()

                # 对比原版规格
                print(f"\n🎯 与原版对比:")
                print(f"   原版: 600x400, 50.0 FPS, h264编码器")
                print(f"   我们: {width}x{height}, {actual_fps:.1f} FPS, {codec}编码器")

                if width == 600 and height == 400 and abs(actual_fps - 50.0) < 1.0:
                    print(f"✅ 规格匹配！应该能在VSCode中正常显示")
                else:
                    print(f"⚠️  规格不匹配，可能需要调整")

                # 清理测试文件
                import shutil
                shutil.rmtree(test_frames_dir)
                print(f"\n🧹 清理了测试文件")

                return True
            else:
                print(f"❌ 无法打开生成的视频文件")
                return False
        else:
            print(f"❌ 视频文件未生成")
            return False

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

if __name__ == "__main__":
    success = test_new_format()
    if success:
        print(f"\n🎉 测试成功！新的H264格式应该与原版兼容")
    else:
        print(f"\n💥 测试失败，需要进一步调试")