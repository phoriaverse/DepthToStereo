import os
import argparse
import subprocess
from pathlib import Path
from app import run_stereo_pipeline  # make sure this is importable!
import sys
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os

# === CONFIG: Adjust this if needed ===
CURRENT_DIR = Path(__file__).parent.resolve()
DEPTH_ANYTHING_PATH = CURRENT_DIR.parent / "VideoDepthAnythingFork"
DEPTH_SCRIPT = DEPTH_ANYTHING_PATH / "run.py"
ENCODER = "vits"

def generate_depth(input_video: Path, output_dir: Path) -> Path:
    video_name = input_video.stem
    depth_out = output_dir 
    depth_out.mkdir(parents=True, exist_ok=True)

    interpreter = str(sys.executable)
    script = str((DEPTH_ANYTHING_PATH / "run.py").resolve())
    cwd = str(DEPTH_ANYTHING_PATH.resolve())
    max_res = get_video_resolution(str(input_video))
    print(f"🧪 Max resolution detected: {max_res}")

    print(f"🧪 Executing: {interpreter} {script}")
    print(f"📁 Absolute cwd: {cwd}")
    print(f"📄 Script exists? {os.path.exists(script)}")
    print(f"📂 CWD exists? {os.path.isdir(cwd)}")

    result = subprocess.run([
        interpreter, script,
        "--input_video", str(input_video),
        "--output_dir", str(depth_out),
        "--encoder", ENCODER,
        "--max_res", str(max_res)
        #"--fp32", "--grayscale"
    ], cwd=cwd, check=True)

    return depth_out / f"{video_name}_vis.mp4"

def get_video_resolution(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    return int(max(width, height))  # depth script uses max_res based on larger dimension

def split_video_by_scene(video_path: Path, output_dir: Path) -> list[Path]:
    from scenedetect.video_splitter import split_video_ffmpeg
    scene_output_dir = output_dir / "scenes"
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()
    split_video_ffmpeg([str(video_path)], scene_list, str(scene_output_dir), show_progress=True)

    return sorted(scene_output_dir.glob("*.mp4"))

def concatenate_clips(clip_paths: list[Path], output_path: Path):
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip.as_posix()}'\n")
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(txt_path),
        "-c", "copy", str(output_path)
    ])
    
def orchestrate_pipeline(input_video: Path, output_root: Path, baseline: float = 15.0, precomputed_depth: Path = None):
    input_video = input_video.resolve()
    video_stem = input_video.stem
    output_root = output_root.resolve()
    
    job_dir = output_root / f"{video_stem}_outputs"
    depth_out = job_dir
    stereo_out = job_dir

    depth_out.mkdir(parents=True, exist_ok=True)
    stereo_out.mkdir(parents=True, exist_ok=True)

    print(f"🎬 Starting full depth + stereo pipeline")
    print(f"🎥 Input video: {input_video}")
    print(f"📦 Job folder: {job_dir}")

    # 1. Generate depth
    if precomputed_depth:
        print(f"📥 Using precomputed depth map: {precomputed_depth}")
        depth_video = Path(precomputed_depth).resolve()
    else:
        print("🧠 Running depth generation stage...")
        depth_video = generate_depth(input_video, depth_out).resolve()
    # 2. Generate stereo
    run_stereo_pipeline(str(video_stem), str(input_video), str(depth_video), str(stereo_out), baseline=baseline)

    print(f"✅ All done! Outputs in: {job_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator: Mono video ➝ Depth ➝ Stereo 180")
    parser.add_argument("--video", required=True, help="Path to input mono video")
    parser.add_argument("--input_dir", type=str, help="Optional directory to process all .mp4 files")
    parser.add_argument("--scenedetect", action="store_true", help="Use PySceneDetect to split video into scenes before processing")
   
    parser.add_argument("--out", required=True, help="Path to output folder")
    parser.add_argument("--baseline", type=float, default=25.0, help="Stereo disparity baseline (default=25)")
    parser.add_argument("--depth", type=str, help="Optional precomputed depth .mp4")
    args = parser.parse_args()
    # 1. Determine input mode
    if args.input_dir:
        video_paths = sorted(Path(args.input_dir).glob("*.mp4"))
    elif args.video:
        video_paths = [Path(args.video)]
    else:
        raise ValueError("Provide --video or --input_dir")

    # 2. Process each video (split if flagged)
    for video_path in video_paths:
        if args.scenedetect:
            print(f"🎬 Splitting: {video_path.name}")
            split_clips = split_video_by_scene(video_path, Path(args.out) / video_path.stem)

            for clip in split_clips:
                orchestrate_pipeline(clip, Path(args.out))

            print(f"🎞️ Reassembling {video_path.name}")
            concatenate_clips(
                sorted(Path(args.out, video_path.stem, "scenes").glob("*_stereo_over_under.mp4")),
                Path(args.out, video_path.stem + "_stereo_combined.mp4")
            )
        else:
            orchestrate_pipeline(video_path, Path(args.out))
    orchestrate_pipeline(Path(args.video), Path(args.out), baseline=args.baseline, precomputed_depth=Path(args.depth) if args.depth else None)
