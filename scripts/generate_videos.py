from __future__ import annotations
import os, argparse, json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FanimWriter, FFMpegWriter
from PIL import Image
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_episode_images(episode_dir: str, camera: str = "front"):
    """Load all images from an episode directory."""
    pattern = os.path.join(episode_dir, f"{camera}_*.png")
    files = sorted(glob.glob(pattern))
    images = [np.array(Image.open(f)) for f in files]
    return np.stack(images, axis=0)  # (T, H, W, 3)

def create_comparison_video(
    episode_dirs: list[str],
    output_path: str,
    labels: list[str],
    fps: int = 4,
    max_frames: int = 40,
):
    """Create side-by-side comparison video of multiple episodes.
    
    Args:
        episode_dirs: List of episode directory paths
        output_path: Output MP4 file path
        labels: Labels for each episode (e.g., ["Baseline", "OT-Train"])
        fps: Frames per second
        max_frames: Maximum number of frames to render
    """
    # Load images from all episodes
    episodes = []
    for ep_dir in episode_dirs:
        front_imgs = load_episode_images(ep_dir, "front")
        wrist_imgs = load_episode_images(ep_dir, "wrist")
        episodes.append({"front": front_imgs, "wrist": wrist_imgs})
    
    # Determine minimum length
    min_len = min(len(ep["front"]) for ep in episodes)
    min_len = min(min_len, max_frames)
    
    # Setup figure
    n_eps = len(episodes)
    fig, axes = plt.subplots(n_eps, 2, figsize=(10, 4*n_eps))
    if n_eps == 1:
        axes = axes.reshape(1, -1)
    
    # Initialize image placeholders
    im_objs = []
    for i, (ep, label) in enumerate(zip(episodes, labels)):
        # Front camera
        im_front = axes[i, 0].imshow(ep["front"][0])
        axes[i, 0].set_title(f"{label} - Front Camera")
        axes[i, 0].axis('off')
        
        # Wrist camera
        im_wrist = axes[i, 1].imshow(ep["wrist"][0])
        axes[i, 1].set_title(f"{label} - Wrist Camera")
        axes[i, 1].axis('off')
        
        im_objs.append((im_front, im_wrist))
    
    plt.tight_layout()
    
    # Animation update function
    def update(frame):
        for i, (ep, (im_front, im_wrist)) in enumerate(zip(episodes, im_objs)):
            im_front.set_data(ep["front"][frame])
            im_wrist.set_data(ep["wrist"][frame])
        return [obj for pair in im_objs for obj in pair]
    
    # Create animation
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=min_len, blit=True, interval=1000//fps)
    
    # Save
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved comparison video to {output_path}")

def plot_coupling_matrix(coupling: np.ndarray, output_path: str, title: str = "OT Coupling Matrix"):
    """Plot and save coupling matrix heatmap.
    
    Args:
        coupling: (B, B) numpy array
        output_path: Output PNG file path
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(coupling, cmap='viridis', aspect='auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Index', fontsize=12)
    ax.set_ylabel('Context Index', fontsize=12)
    
    # Annotate diagonal
    B = coupling.shape[0]
    for i in range(B):
        ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=2))
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coupling Weight', fontsize=12)
    
    # Stats text
    diag_mean = np.diag(coupling).mean()
    off_diag_mean = coupling[~np.eye(B, dtype=bool)].mean()
    stats_text = f"Diagonal: {diag_mean:.4f}\nOff-Diagonal: {off_diag_mean:.4f}\nRatio: {diag_mean/off_diag_mean:.2f}×"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved coupling matrix to {output_path}")

def plot_trajectory_metrics(results_json: str, output_dir: str):
    """Plot success rates, collision rates, and clearances from evaluation results.
    
    Args:
        results_json: Path to results JSON file
        output_dir: Output directory for plots
    """
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Success rate by scene
    scenes = list(results.keys())
    success_rates = [results[s]['success_rate'] for s in scenes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(scenes, success_rates, color='steelblue', alpha=0.8)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_xlabel('Scene', fontsize=12)
    ax.set_title('Success Rate by Scene', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rates.png'), dpi=150)
    plt.close()
    
    # Collision rates
    collision_rates = [results[s]['collision_rate'] for s in scenes]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(scenes, collision_rates, color='coral', alpha=0.8)
    ax.set_ylabel('Collision Rate', fontsize=12)
    ax.set_xlabel('Scene', fontsize=12)
    ax.set_title('Collision Rate by Scene', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'collision_rates.png'), dpi=150)
    plt.close()
    
    print(f"Saved trajectory metrics to {output_dir}")

def main():
    ap = argparse.ArgumentParser(description="Generate visualization videos from evaluation results")
    ap.add_argument("--results_dir", type=str, default="./results", help="Directory containing evaluation results")
    ap.add_argument("--output_dir", type=str, default="./videos", help="Output directory for videos")
    ap.add_argument("--variants", nargs="+", default=["baseline", "continued", "unfreeze", "ot"],
                    help="Variants to visualize")
    ap.add_argument("--scenes", nargs="+", default=["scene_d", "scene_e", "scene_f"],
                    help="Scenes to visualize")
    ap.add_argument("--fps", type=int, default=10, help="Video frames per second")
    args = ap.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate comparison videos for each scene
    for scene in args.scenes:
        episode_dirs = []
        labels = []
        for variant in args.variants:
            pattern = os.path.join(args.results_dir, f"{variant}_{scene}_ep*")
            dirs = sorted(glob.glob(pattern))
            if dirs:
                episode_dirs.append(dirs[0])  # Take first episode
                labels.append(variant.capitalize())
        
        if episode_dirs:
            output_path = os.path.join(args.output_dir, f"comparison_{scene}.mp4")
            create_comparison_video(episode_dirs, output_path, labels, fps=args.fps)
    
    # Plot trajectory metrics
    for variant in args.variants:
        results_json = os.path.join(args.results_dir, f"{variant}_results.json")
        if os.path.exists(results_json):
            plot_dir = os.path.join(args.output_dir, f"{variant}_plots")
            plot_trajectory_metrics(results_json, plot_dir)
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
