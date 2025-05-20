# mono_to_stereo.py

import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import subprocess
import argparse


def run_stereo_pipeline(video_file, depth_file, output_dir, baseline=25):
    # STEP 1: Create folders for frame extraction and output
    folders = {
        "video_frames": "video_frames",
        "depth_frames": "depth_frames",
        "left_frames": "left_frames",
        "right_frames": "right_frames",
        "ou_frames": "ou_frames"
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # STEP 2: Extract frames
    print("🎞️ Extracting frames...")

    subprocess.run([
        "ffmpeg", "-i", video_file, "-q:v", "1",
        os.path.join(folders["video_frames"], "frame_%04d.png")
    ])

    subprocess.run([
        "ffmpeg", "-i", depth_file, "-q:v", "1",
        os.path.join(folders["depth_frames"], "frame_%04d.png")
    ])

    # STEP 3: Generate stereo output with progress bar
    def generate_stereo_all(video_dir, depth_dir, left_dir, right_dir, ou_dir, baseline=25):
        frame_names = sorted(os.listdir(video_dir))
        print(f"🔧 Generating stereo frames ({len(frame_names)} total)...")

        for fname in tqdm(frame_names, desc="Stereo synthesis", unit="frame"):
            color_path = os.path.join(video_dir, fname)
            depth_path = os.path.join(depth_dir, fname)

            color = cv2.imread(color_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

            if color is None or depth is None:
                continue

            h, w = depth.shape
            norm_depth = cv2.normalize(depth.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
            disparity = (1 - norm_depth) * baseline

            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_left = (map_x + disparity / 2).astype(np.float32)
            map_right = (map_x - disparity / 2).astype(np.float32)

            left_eye = cv2.remap(color, map_left, map_y.astype(np.float32), cv2.INTER_LINEAR)
            right_eye = cv2.remap(color, map_right, map_y.astype(np.float32), cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(left_dir, fname), left_eye)
            cv2.imwrite(os.path.join(right_dir, fname), right_eye)
            ou_frame = np.vstack((left_eye, right_eye))
            cv2.imwrite(os.path.join(ou_dir, fname), ou_frame)

    generate_stereo_all(
        folders["video_frames"],
        folders["depth_frames"],
        folders["left_frames"],
        folders["right_frames"],
        folders["ou_frames"],
        baseline
    )

    # STEP 4: Encode final videos
    print("🎞️ Encoding final videos...")

    subprocess.run([
        "ffmpeg", "-framerate", "25", "-i",
        os.path.join(folders["left_frames"], "frame_%04d.png"),
        "-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
        "-pix_fmt", "yuv420p", os.path.join(output_dir, "left_eye.mp4")
    ])

    subprocess.run([
        "ffmpeg", "-framerate", "25", "-i",
        os.path.join(folders["right_frames"], "frame_%04d.png"),
        "-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
        "-pix_fmt", "yuv420p", os.path.join(output_dir, "right_eye.mp4")
    ])

    subprocess.run([
        "ffmpeg", "-framerate", "25", "-i",
        os.path.join(folders["ou_frames"], "frame_%04d.png"),
        "-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
        "-pix_fmt", "yuv420p", os.path.join(output_dir, "stereo_over_under.mp4")
    ])

    # STEP 5: Cleanup
    print("🧹 Cleaning up temporary folders...")
    for folder in folders.values():
        shutil.rmtree(folder, ignore_errors=True)

    print(f"✅ All done! Videos saved to:\n📁 {output_dir}")


# Optional CLI front-end
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stereo video from mono + depth input")
    parser.add_argument("--video", required=True, help="Path to mono RGB video")
    parser.add_argument("--depth", required=True, help="Path to depth video")
    parser.add_argument("--out", required=True, help="Directory to save stereo output")
    parser.add_argument("--baseline", type=float, default=25, help="Disparity baseline")
    args = parser.parse_args()

    run_stereo_pipeline(args.video, args.depth, args.out, baseline=args.baseline)