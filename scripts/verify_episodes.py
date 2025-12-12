from __future__ import annotations
import os, sys, json, glob
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def analyze_episode(episode_dir: str) -> dict:
    """Analyze a single episode directory and return diagnostics."""
    meta_path = os.path.join(episode_dir, "meta.json")
    parquet_path = os.path.join(episode_dir, "episode.parquet")
    
    if not os.path.exists(meta_path) or not os.path.exists(parquet_path):
        return {"error": "Missing meta.json or episode.parquet"}
    
    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Load trajectory
    df = pd.read_parquet(parquet_path)
    
    # Check for required 7D end-effector columns
    ee_state_cols = [c for c in df.columns if c.startswith("state_ee_state")]
    ee_delta_cols = [c for c in df.columns if c.startswith("act_ee_delta")]
    
    # Legacy columns (should exist for backward compatibility but not be primary)
    joint_state_cols = [c for c in df.columns if c.startswith("state_q") and not c.startswith("state_qdot")]
    joint_action_cols = [c for c in df.columns if c.startswith("act_qdot")]
    
    # Determine if episode has correct 7D format
    has_ee_state = len(ee_state_cols) > 0
    has_ee_delta = len(ee_delta_cols) > 0
    has_legacy = len(joint_action_cols) > 0
    
    # Extract dimensions
    ee_state_dim = 0
    ee_delta_dim = 0
    joint_state_dim = 0
    joint_action_dim = 0
    
    if has_ee_state and len(df) > 0:
        first_row = df[ee_state_cols[0]].iloc[0]
        if isinstance(first_row, (list, np.ndarray)):
            ee_state_dim = len(first_row)
        else:
            ee_state_dim = len(ee_state_cols)
    
    if has_ee_delta and len(df) > 0:
        first_row = df[ee_delta_cols[0]].iloc[0]
        if isinstance(first_row, (list, np.ndarray)):
            ee_delta_dim = len(first_row)
    else:
            ee_delta_dim = len(ee_delta_cols)
    
    if has_legacy and len(df) > 0:
        first_row = df[joint_action_cols[0]].iloc[0]
        if isinstance(first_row, (list, np.ndarray)):
            joint_action_dim = len(first_row)
        else:
            joint_action_dim = len(joint_action_cols)
    
    if len(joint_state_cols) > 0 and len(df) > 0:
        first_row = df[joint_state_cols[0]].iloc[0]
        if isinstance(first_row, (list, np.ndarray)):
            joint_state_dim = len(first_row)
    else:
            joint_state_dim = len(joint_state_cols)
    
    # Analyze gripper behavior from 7D end-effector state
    gripper_analysis = {}
    if has_ee_state and ee_state_dim == 7:
        # Extract end-effector states
        if df[ee_state_cols[0]].dtype == object:
            # Stored as serialized arrays
            ee_states = np.array([np.asarray(row, dtype=np.float32) for row in df[ee_state_cols[0]].to_numpy()])
        else:
            # Stored as separate columns
            ee_states = df[ee_state_cols].to_numpy(dtype=np.float32)
        
        # Gripper is dimension 6 (0-indexed): [pos(3), rot(3), gripper(1)]
        gripper_pos = ee_states[:, 6]  # Single value: 0=closed, ~0.08=open
        
        # Extract gripper deltas from actions (if available)
        if has_ee_delta and ee_delta_dim == 7:
            if df[ee_delta_cols[0]].dtype == object:
                # Stored as serialized arrays
                ee_deltas = np.array([np.asarray(row, dtype=np.float32) for row in df[ee_delta_cols[0]].to_numpy()])
            else:
                # Stored as separate columns
                ee_deltas = df[ee_delta_cols].to_numpy(dtype=np.float32)
            
            gripper_delta = ee_deltas[:, 6]  # Gripper velocity/delta
        
        gripper_analysis = {
                "opening_min": float(gripper_pos.min()),
                "opening_max": float(gripper_pos.max()),
                "opening_mean": float(gripper_pos.mean()),
                "opening_std": float(gripper_pos.std()),
                "delta_mean": float(np.abs(gripper_delta).mean()),
                "delta_max": float(np.abs(gripper_delta).max()),
                "has_closing": bool((gripper_pos < 0.01).any()),  # Gripper closed
                "has_opening": bool((gripper_pos > 0.04).any()),  # Gripper opened (half-open for Panda)
                "has_motion": bool((np.abs(gripper_delta) > 0.001).any()),  # Gripper moved
                "gripper_range": float(gripper_pos.max() - gripper_pos.min()),
            }
    
    # Check for image files
    front_images = glob.glob(os.path.join(episode_dir, "front_*.png"))
    wrist_images = glob.glob(os.path.join(episode_dir, "wrist_*.png"))
    
    # Format validation
    format_ok = (
        has_ee_state and 
        has_ee_delta and 
        ee_state_dim == 7 and 
        ee_delta_dim == 7
    )
    
    return {
        "episode_id": os.path.basename(episode_dir),
        "format_ok": format_ok,
        "has_ee_state": has_ee_state,
        "has_ee_delta": has_ee_delta,
        "has_legacy": has_legacy,
        "ee_state_dim": ee_state_dim,
        "ee_delta_dim": ee_delta_dim,
        "joint_state_dim": joint_state_dim,
        "joint_action_dim": joint_action_dim,
        "trajectory_length": len(df),
        "num_front_images": len(front_images),
        "num_wrist_images": len(wrist_images),
        "images_match_trajectory": len(front_images) == len(df),
        "robot_config": meta.get("robot_config", "unknown"),
        "gripper": gripper_analysis,
        "columns": list(df.columns),
    }

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Verify episode structure for 7D end-effector format (VJEPA2-AC)")
    ap.add_argument("--episodes-dir", type=str, default="./episodes", help="Root episodes directory")
    ap.add_argument("--sample-size", type=int, default=1, help="Number of episodes to sample per scene")
    ap.add_argument("--verbose", action="store_true", help="Print detailed per-episode stats")
    ap.add_argument("--show-columns", action="store_true", help="Show column names from first episode")
    args = ap.parse_args()
    
    # Find all episode directories
    episode_dirs = []
    for parquet_file in glob.glob(os.path.join(args.episodes_dir, "**", "episode.parquet"), recursive=True):
        episode_dirs.append(os.path.dirname(parquet_file))
    
    if not episode_dirs:
        print(f"No episodes found in {args.episodes_dir}")
        print("Run: python data.py --episodes 10 --scene-set abc --out ./episodes")
        return 1
    
    print(f"Found {len(episode_dirs)} episodes in {args.episodes_dir}")
    print()
    
    # Sample episodes
    import random
    random.seed(42)
    sample = random.sample(episode_dirs, min(args.sample_size, len(episode_dirs)))
    
    # Analyze sample
    results = []
    for ep_dir in sample:
        result = analyze_episode(ep_dir)
        results.append(result)
        if args.verbose:
            print(f"Episode: {result['episode_id']}")
            print(f"Format: {'✓ CORRECT' if result['format_ok'] else 'INCORRECT'}")
            print(f"EE state: {'✓' if result['has_ee_state'] else '✗'} (dim={result['ee_state_dim']})")
            print(f"EE delta: {'✓' if result['has_ee_delta'] else '✗'} (dim={result['ee_delta_dim']})")
            print(f"Legacy qdot: {'present' if result['has_legacy'] else 'absent'} (dim={result['joint_action_dim']})")
            print(f"Length: {result['trajectory_length']} timesteps")
            print(f"Images: {result['num_front_images']} front, {result['num_wrist_images']} wrist {'✓' if result['images_match_trajectory'] else 'MISMATCH'}")
            if result.get('gripper'):
                g = result['gripper']
                print(f"Gripper: {g['opening_min']:.4f} → {g['opening_max']:.4f} (range={g['gripper_range']:.4f})")
                print(f"Gripper motion: {'✓' if g['has_motion'] else '✗'} (max_delta={g['delta_max']:.4f})")
                print(f"Grasp detected: {'✓' if g['has_closing'] else '✗'}")
            print()
    
    # Show columns from first episode
    if args.show_columns and results:
        first = results[0]
        if 'columns' in first:
            print("=" * 60)
            print("COLUMNS IN FIRST EPISODE")
            print("=" * 60)
            state_cols = [c for c in first['columns'] if c.startswith('state_')]
            action_cols = [c for c in first['columns'] if c.startswith('act_')]
            other_cols = [c for c in first['columns'] if not c.startswith('state_') and not c.startswith('act_')]
            
            print("\nState columns:")
            for col in state_cols:
                marker = "✓" if "ee_state" in col else "⚠" if "qdot" not in col else ""
                print(f"{marker} {col}")
            
            print("\nAction columns:")
            for col in action_cols:
                marker = "✓" if "ee_delta" in col else "⚠" if "qdot" not in col else ""
                print(f"{marker} {col}")
            
            print("\nOther columns:")
            for col in other_cols:
                print(f"{col}")
            print()
    
    # Aggregate stats
    format_ok_count = sum(r['format_ok'] for r in results if 'error' not in r)
    has_ee_state_count = sum(r['has_ee_state'] for r in results if 'error' not in r)
    has_ee_delta_count = sum(r['has_ee_delta'] for r in results if 'error' not in r)
    has_legacy_count = sum(r['has_legacy'] for r in results if 'error' not in r)
    
    ee_state_dims = [r['ee_state_dim'] for r in results if 'error' not in r and r['has_ee_state']]
    ee_delta_dims = [r['ee_delta_dim'] for r in results if 'error' not in r and r['has_ee_delta']]
    
    print("=" * 60)
    print("SUMMARY: 7D END-EFFECTOR FORMAT VERIFICATION")
    print("=" * 60)
    print(f"Total episodes: {len(episode_dirs)}")
    print(f"Sampled: {len(results)}")
    print()
    
    # Format check
    print("FORMAT CHECK")
    print("-" * 60)
    print(f"Correct 7D format: {format_ok_count}/{len(results)} ({100*format_ok_count/len(results):.1f}%)")
    print(f"state_ee_state present: {has_ee_state_count}/{len(results)}")
    print(f"act_ee_delta present: {has_ee_delta_count}/{len(results)}")
    print(f"Legacy act_qdot present: {has_legacy_count}/{len(results)}")
    print()
    
    if ee_state_dims:
        print("DIMENSION CHECK")
        print("-" * 60)
        ee_state_consistent = len(set(ee_state_dims)) == 1 and ee_state_dims[0] == 7
        ee_delta_consistent = len(set(ee_delta_dims)) == 1 and ee_delta_dims[0] == 7
        
        print(f"EE state dimension: {set(ee_state_dims)} {'✓ CORRECT' if ee_state_consistent else 'INCORRECT'}")
        print(f"EE delta dimension: {set(ee_delta_dims)} {'✓ CORRECT' if ee_delta_consistent else 'INCORRECT'}")
        print()
        
        # Image check
        image_match_count = sum(r['images_match_trajectory'] for r in results if 'error' not in r)
        avg_front = np.mean([r['num_front_images'] for r in results if 'error' not in r])
        avg_wrist = np.mean([r['num_wrist_images'] for r in results if 'error' not in r])
        
        print("IMAGE CHECK")
        print("-" * 60)
        print(f"Episodes with matching image count: {image_match_count}/{len(results)} ({100*image_match_count/len(results):.1f}%)")
        print(f"Average images per episode: {avg_front:.1f} front, {avg_wrist:.1f} wrist")
        print()
        
        # Gripper analysis
        gripper_results = [r['gripper'] for r in results if r.get('gripper')]
        if gripper_results:
            print("GRIPPER ANALYSIS (7D FORMAT)")
            print("-" * 60)
            has_motion = sum(g['has_motion'] for g in gripper_results)
            has_closing = sum(g['has_closing'] for g in gripper_results)
            has_opening = sum(g['has_opening'] for g in gripper_results)
            
            print(f"Episodes with gripper motion: {has_motion}/{len(gripper_results)} ({100*has_motion/len(gripper_results):.1f}%)")
            print(f"Episodes with closing (< 0.01m): {has_closing}/{len(gripper_results)} ({100*has_closing/len(gripper_results):.1f}%)")
        print(f"Episodes with opening (> 0.04m): {has_opening}/{len(gripper_results)} ({100*has_opening/len(gripper_results):.1f}%)")
            
        avg_delta = np.mean([g['delta_mean'] for g in gripper_results])
        max_delta = np.max([g['delta_max'] for g in gripper_results])
        avg_range = np.mean([g['gripper_range'] for g in gripper_results])
        
        print(f"Average gripper delta: {avg_delta:.4f} m/s (max={max_delta:.4f})")
        print(f"Average gripper range: {avg_range:.4f} m")
        print()
            
        if has_motion < len(gripper_results) * 0.5:
            print("Less than 50% of episodes show gripper motion")
        if has_closing < len(gripper_results) * 0.3:
            print("Less than 30% of episodes show grasping behavior")
        if avg_range < 0.02:
            print("Gripper range is small (< 2cm), may not be moving enough")
        print()
    
    # Final verdict
    print("=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    
    all_checks_pass = (
        format_ok_count == len(results) and
        has_ee_state_count == len(results) and
        has_ee_delta_count == len(results) and
        len(set(ee_state_dims)) == 1 and ee_state_dims[0] == 7 and
        len(set(ee_delta_dims)) == 1 and ee_delta_dims[0] == 7 and
        image_match_count >= len(results) * 0.9  # Allow 10% image mismatch
    )
    
    if all_checks_pass:
        print("PASSED: Episodes are correctly formatted for 7D VJEPA2-AC training!")
        print()
        print("All episodes have:")
        print("state_ee_state (7D): [pos(3), rot_rpy(3), gripper(1)]")
        print("act_ee_delta (7D): [dpos(3), drot_rpy(3), dgripper(1)]")
        print("Matching image counts")
        print()
        if gripper_results and has_motion >= len(gripper_results) * 0.5:
            print("Gripper control is active and functional!")
        print()
        print("Ready for training:")
        print("python train.py --config configs/vjepa2ac-continued.yaml")
        return 0
    else:
        print("FAILED: Episodes are NOT in the correct 7D format")
        print()
        
        if format_ok_count < len(results):
            print(f"{len(results) - format_ok_count}/{len(results)} episodes missing required columns")
        
        if has_ee_state_count < len(results):
            print(f"state_ee_state missing in {len(results) - has_ee_state_count} episodes")
        
        if has_ee_delta_count < len(results):
            print(f"act_ee_delta missing in {len(results) - has_ee_delta_count} episodes")
        
        if ee_state_dims and (len(set(ee_state_dims)) != 1 or ee_state_dims[0] != 7):
            print(f"EE state dimension incorrect: {set(ee_state_dims)} (expected: {{7}})")
        
        if ee_delta_dims and (len(set(ee_delta_dims)) != 1 or ee_delta_dims[0] != 7):
            print(f"EE delta dimension incorrect: {set(ee_delta_dims)} (expected: {{7}})")
        
        print()
        print("ACTION REQUIRED:")
        print("1. Regenerate dataset with updated data generation scripts")
        print("2. Ensure drake_dataset_generator.py saves ee_state and ee_delta")
        print("3. Run: python data.py --episodes 50 --scene-set abc --out ./episodes")
        print("4. Re-run this verification script")
        print()
        print("See: DATASET_REGENERATION_REQUIRED.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
