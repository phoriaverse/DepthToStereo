import os
import argparse
import subprocess
from pathlib import Path
from app import run_stereo_pipeline  # make sure this is importable!
import sys
# === CONFIG: Adjust this if needed ===
VIDEO_DEPTH_ANYTHING_PATH = Path("C:/Users/Phoria - WS01/sam/third-party/Video-Depth-Anything").resolve()
DEPTH_SCRIPT = VIDEO_DEPTH_ANYTHING_PATH / "run.py"
ENCODER = "vitl"

def generate_depth(input_video: Path, output_dir: Path) -> Path:
    video_name = input_video.stem
    depth_out = output_dir / "depth"
    depth_out.mkdir(parents=True, exist_ok=True)

    interpreter = str(sys.executable)
    script = str((VIDEO_DEPTH_ANYTHING_PATH / "run.py").resolve())
    cwd = str(VIDEO_DEPTH_ANYTHING_PATH.resolve())

    print(f"🧪 Executing: {interpreter} {script}")
    print(f"📁 Absolute cwd: {cwd}")
    print(f"📄 Script exists? {os.path.exists(script)}")
    print(f"📂 CWD exists? {os.path.isdir(cwd)}")

    result = subprocess.run([
        interpreter, script,
        "--input_video", str(input_video),
        "--output_dir", str(depth_out),
        "--encoder", ENCODER,
        #"--fp32", "--grayscale"
    ], cwd=cwd, check=True)

    return depth_out / f"{video_name}_vis.mp4"
def orchestrate_pipeline(input_video: Path, output_dir: Path, baseline: float = 25.0):
    input_video = input_video.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎬 Starting full depth + stereo pipeline")
    print(f"🎥 Input video: {input_video}")
    print(f"📦 Output folder: {output_dir}")

    # 1. Generate depth
    depth_video = generate_depth(input_video, output_dir)

    # 2. Generate stereo
    stereo_out = output_dir / "stereo"
    stereo_out.mkdir(exist_ok=True)
    run_stereo_pipeline(str(input_video), str(depth_video), str(stereo_out), baseline=baseline)

    print(f"✅ All done! Stereo results in: {stereo_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator: Mono video ➝ Depth ➝ Stereo 180")
    parser.add_argument("--video", required=True, help="Path to input mono video")
    parser.add_argument("--out", required=True, help="Path to output folder")
    parser.add_argument("--baseline", type=float, default=25.0, help="Stereo disparity baseline (default=25)")
    args = parser.parse_args()

    orchestrate_pipeline(Path(args.video), Path(args.out), baseline=args.baseline)
