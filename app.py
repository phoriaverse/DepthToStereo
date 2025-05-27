# mono_to_stereo.py
from pathlib import Path
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import subprocess
import argparse


def run_stereo_pipeline(video_name, video_file, depth_file, output_dir, baseline=25):
    
    video_file = Path(video_file).resolve()
    depth_file = Path(depth_file).resolve()
    output_dir = Path(output_dir).resolve()

    if not video_file.exists():
        raise FileNotFoundError(f"🎥 Video file not found: {video_file}")
    if not depth_file.exists():
        raise FileNotFoundError(f"🧠 Depth file not found: {depth_file}")

    # STEP 1: Create folders for frame extraction and output
    output_dir = Path(output_dir)
    folders = {
        "video_frames": output_dir / "video_frames",
        "depth_frames": output_dir / "depth_frames",
        "left_frames": output_dir / "left_frames",
        "right_frames": output_dir / "right_frames",
        "ou_frames": output_dir / "ou_frames"
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
        # STEP 2: Extract frames
        print("🎞️ Extracting frames...")
    subprocess.run([
        "ffmpeg", "-i", video_file, "-q:v", "1",
        str(folders["video_frames"] / "frame_%04d.png")
    ], check = True)

    subprocess.run([
        "ffmpeg", "-i", depth_file, "-q:v", "1",
        str(folders["depth_frames"] / "frame_%04d.png")
    ], check = True)
    
    video_frames = sorted(os.listdir(folders["video_frames"]))
    depth_frames = sorted(os.listdir(folders["depth_frames"]))
    print(f"🧪 Found {len(video_frames)} video frames and {len(depth_frames)} depth frames")
    # STEP 3: Generate stereo output with progress bar

    def generate_stereo_all(video_dir, depth_dir, left_dir, right_dir, ou_dir, baseline=15):
        frame_names = sorted(os.listdir(video_dir))
        print(f"🔧 Generating stereo frames ({len(frame_names)} total)...")

        for fname in tqdm(frame_names, desc="Stereo synthesis", unit="frame"):
            color_path = Path(video_dir) / fname
            depth_path = Path(depth_dir) / fname

            color = cv2.imread(str(color_path))
            depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)

            if color is None or depth is None:
                print(f"⚠️ Skipping {fname} due to read error")
                continue

            # ✅ Blur to reduce harsh depth jumps
            smoothed_depth = cv2.medianBlur(depth, 5)

            # ✅ Normalize to 0-1 range for stable disparity
            norm_depth = cv2.normalize(smoothed_depth.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

            # ✅ Safe disparity calculation
            disparity = np.nan_to_num((1 - norm_depth) * baseline)

            h, w = depth.shape
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

            map_left = (map_x + disparity / 2).astype(np.float32)
            map_right = (map_x - disparity / 2).astype(np.float32)

            # ✅ Border mode protects against edge bleeding
            left_eye = cv2.remap(color, map_left, map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            right_eye = cv2.remap(color, map_right, map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # ✅ Fallback if remap somehow fails
            if left_eye is None or right_eye is None:
                print(f"⚠️ Frame {fname} remap failed")
                continue

            cv2.imwrite(str(Path(left_dir) / fname), left_eye)
            cv2.imwrite(str(Path(right_dir) / fname), right_eye)

            ou_frame = np.vstack((left_eye, right_eye))
            cv2.imwrite(str(Path(ou_dir) / fname), ou_frame)

    generate_stereo_all(
        folders["video_frames"],
        folders["depth_frames"],
        folders["left_frames"],
        folders["right_frames"],
        folders["ou_frames"],
        baseline
    )

    print("🎞️ Encoding final videos...")

    def encode_video(frames_dir, output_name):
        subprocess.run([
            "ffmpeg", "-framerate", "25", "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx265", "-crf", "20", "-preset", "veryslow", "-tag:v", "hvc1",
            "-pix_fmt", "yuv420p", str(output_dir / output_name)
        ], check=True)

    # encode_video(folders["left_frames"], f"{video_name}_left_eye.mp4")
    # encode_video(folders["right_frames"], f"{video_name}_right_eye.mp4")
    encode_video(folders["ou_frames"], f"{video_name}_stereo_over_under.mp4")

    print("🧹 Cleaning up temporary folders...")
    for folder in folders.values():
        shutil.rmtree(folder, ignore_errors=True)

    print(f"✅ All done! Final outputs saved to: {output_dir}")

    # STEP 5: Cleanup
    print("🧹 Cleaning up temporary folders...")
#    for folder in folders.values():
#        shutil.rmtree(folder, ignore_errors=True)

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
