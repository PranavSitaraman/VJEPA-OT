import os
import argparse
from pathlib import Path

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

from PIL import Image
import numpy as np


def extract_frames(video_path: str, output_dir: str, num_frames: int = 6, prefix: str = "frame"):
    """
    Extract evenly-spaced frames from a video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
        prefix: Prefix for output filenames
    """
    # Open video with imageio
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    
    # Get video properties
    total_frames = reader.count_frames()
    fps = meta.get('fps', 30)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}, Duration: {duration:.1f}s")
    
    # Calculate frame indices to extract (evenly spaced, excluding first and last)
    # Skip first and last frames to avoid initialization/termination artifacts
    start_frame = 1
    end_frame = total_frames - 2  # Exclude last frame
    usable_frames = end_frame - start_frame + 1
    
    if usable_frames < num_frames:
        frame_indices = list(range(start_frame, end_frame + 1))
    else:
        frame_indices = [start_frame + int(i * (usable_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    print(f"Extracting frames at indices: {frame_indices}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames
    extracted = []
    for i, frame_idx in enumerate(frame_indices):
        try:
            frame = reader.get_data(frame_idx)
            output_path = os.path.join(output_dir, f"{prefix}_{i:02d}.png")
            # Convert to PIL Image and save
            img = Image.fromarray(frame)
            img.save(output_path)
            extracted.append(output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Could not read frame {frame_idx}: {e}")
    
    reader.close()
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract frames from test-place.py evaluation videos")
    parser.add_argument("--results-dir", type=str, default="../results",
                        help="Directory containing result videos")
    parser.add_argument("--output-dir", type=str, default="figures/frames/test-place",
                        help="Output directory for extracted frames")
    parser.add_argument("--num-frames", type=int, default=6,
                        help="Number of frames to extract per video")
    args = parser.parse_args()
    
    # Define videos to process (test-place.py hybrid placement evaluation)
    videos = [
        ("vjepa2ac-baseline_all_scenes_mpc.mp4", "baseline"),
        ("vjepa2ac-ot_all_scenes_mpc.mp4", "ot"),
        ("vjepa2ac-continued_all_scenes_mpc.mp4", "continued"),
        ("vjepa2ac-unfreeze_all_scenes_mpc.mp4", "unfreeze")
    ]
    
    # Also check for other variants if they exist
    optional_videos = [
    ]
    
    for video_name, variant in optional_videos:
        video_path = os.path.join(args.results_dir, video_name)
        if os.path.exists(video_path):
            videos.append((video_name, variant))
        
    for video_name, variant in videos:
        video_path = os.path.join(args.results_dir, video_name)
        
        if not os.path.exists(video_path):
            print(f"\nSkipping {video_name} (not found)")
            continue
        
        output_subdir = os.path.join(args.output_dir, variant)
        print(f"\nProcessing {variant}...")
        
        try:
            extract_frames(
                video_path=video_path,
                output_dir=output_subdir,
                num_frames=args.num_frames,
                prefix="frame"
            )
        except Exception as e:
            print(f"{e}")
    
    print(f"\n" + "=" * 60)
    print("Done! Frames saved to:", args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
