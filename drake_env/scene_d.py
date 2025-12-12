import numpy as np
from pathlib import Path
from typing import Optional

from drake_env.universal_planner import plan_scene_trajectory
from drake_env.universal_episode_generator import generate_episode, build_trajs


def generate_scene_d_episode(
    letter: str = "C",
    seed: int = None,
    duration: float = 25.0,
    fps: float = 4.0,
    meshcat=None,
    save_options=None,
    randomize: float = 0.5,
    **kwargs
) -> dict:
    """
    Generate Scene D episode: letter manipulation on shelf.
    
    Args:
        letter: Letter to manipulate (e.g., "C")
        seed: Random seed
        duration: Episode duration (ignored - trajectory determines length)
        fps: Frames per second
        meshcat: Meshcat visualizer
        save_options: Save options (for backward compatibility)
        
    Returns:
        Episode data dict
    """
    rng = np.random.default_rng(seed)
    
    assets_dir = Path(__file__).parent.parent / "assets"
    
    # === Scene D Planning ===
    path_pick, path_place, path_reset, q_grasp, q_approach = plan_scene_trajectory(
        letter=letter,
        meshcat=meshcat,
        assets_dir=assets_dir,
        randomize=randomize,
        rng=rng,
    )
    
    # Build trajectories
    traj_q, traj_wsg = build_trajs(path_pick, q_grasp, q_approach, path_place, path_reset)
    
    # === Scene D Configuration ===
    # Prepare scenario YAML with letter
    scenario_template = assets_dir / "franka_shelves_scenario.yaml"
    with open(scenario_template, 'r') as f:
        yaml_content = f.read()
    
    letter_file = f"file://{assets_dir}/{letter}_model/{letter}.sdf"
    yaml_content = yaml_content.replace("LETTER_FILE_PLACEHOLDER", letter_file)
    yaml_content = yaml_content.replace("LETTER_BODY_NAME", f"{letter}_body_link")
    
    # Camera positions: LEFT exocentric view matching Meta's VJEPA2 pretrained model
    # Meta trained on DROID dataset using left exocentric camera (~60° left of robot)
    # Y = +1.2 (positive) = LEFT of robot base (matches Meta's DROID training)
    workspace_center = np.array([0.70, 0.0, 0.42])
    front_camera_pos = np.array([0.25, 1.2, 1.12])  # LEFT exocentric (Y = +1.2)
    wrist_camera_pos = np.array([0.18, 0.75, 0.65])  # Wrist also on left side
    wrist_camera_target = np.array([0.70, 0.0, 0.35])
    
    # === Use Universal Generator ===
    episode_data = generate_episode(
        scenario_yaml=yaml_content,
        traj_q=traj_q,
        traj_wsg=traj_wsg,
        fps=fps,
        meshcat=meshcat,
        language_prompt=f"Pick up the letter {letter} off the table and place it on the shelf",
        subgoals=['grasp', 'near', 'final'],
        workspace_center=workspace_center,
        front_camera_pos=front_camera_pos,
        wrist_camera_pos=wrist_camera_pos,
        wrist_camera_target=wrist_camera_target,
        randomize=randomize,
        rng=rng,
    )
    
    # Return episode data dict
    return episode_data


__all__ = ["generate_scene_d_episode"]
