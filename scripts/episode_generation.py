from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from scripts.drake_dataset_generator import (
    DrakeWorld,
    EpisodeConfig,
    WorldParams,
    save_episode,
)


@dataclass
class SceneConfig:
    label: str
    description: str
    letter: str
    scene_type: str  # "letter_shelf" or "letter_on_stand"


@dataclass
class SaveOptions:
    output_root: Path
    export_html: bool = False
    html_dir: Optional[Path] = None
    num_frames: int = 30

    def episode_dir(self, scene: SceneConfig, seed: int) -> Path:
        return self.output_root / scene.label / f"{scene.label}_{seed:08d}"

    def html_path(self, scene: SceneConfig, seed: int, *, failed: bool = False) -> Optional[Path]:
        if not self.export_html or self.html_dir is None:
            return None
        suffix = "_failed" if failed else ""
        return self.html_dir / f"{scene.label}_{seed:08d}{suffix}.html"


class EpisodeGenerator:
    """Wrapper owning a reusable :class:`DrakeWorld`."""

    def __init__(self, params: WorldParams, meshcat=None) -> None:
        self.params = params
        self.world = DrakeWorld(params, meshcat=meshcat)

    def generate(
        self,
        scene: SceneConfig,
        seed: int,
        strength: float,
        *,
        render_images: bool,
        try_cover_trajectory: bool,
    ) -> EpisodeConfig:
        self.world.params.scene_type = scene.scene_type
        self.world.params.letter_initial = scene.letter
        episode = self.world.generate_episode(
            seed=seed,
            strength=strength,
            render_images=render_images,
            try_cover_trajectory=try_cover_trajectory,
        )
        return episode

    def generate_and_save(
        self,
        scene: SceneConfig,
        seed: int,
        save_options: SaveOptions,
        *,
        render_images: bool,
        try_cover_trajectory: bool,
        strength: float,
    ) -> bool:
        ep = self.generate(
            scene,
            seed,
            strength=strength,
            render_images=render_images,
            try_cover_trajectory=try_cover_trajectory,
        )
        if not ep.success:
            html_failed = save_options.html_path(scene, seed, failed=True)
            if html_failed:
                self.world.export_scene_html(html_failed, ep.q_start)
            return False

        episode_dir = save_options.episode_dir(scene, seed)
        episode_dir.mkdir(parents=True, exist_ok=True)
        save_episode(episode_dir, self.world, ep, num_frames=save_options.num_frames)

        html_path = save_options.html_path(scene, seed)
        if html_path is not None:
            self.world.export_scene_html(html_path)

        return True


class SceneDReferenceGenerator:
    """
    Episode generator for Scene D using exact reference notebook approach.
    
    This replicates the reference notebook (5_plan_place_initials.ipynb) exactly:
    - Two-scenario planning (base + grasp)
    - RRT-Connect with proper iteration counts
    - Sequential IK solving with joint centering
    """
    
    def __init__(self, meshcat=None):
        self.meshcat = meshcat
        # Create minimal params object for compatibility
        self.params = WorldParams(
            add_gripper=True,
            add_floor=False,
            camera_res=(640, 480),
            scene_type="franka_reference",
            arm="franka",
        )
    
    def generate_and_save(
        self,
        scene: SceneConfig,
        seed: int,
        save_options: SaveOptions,
        *,
        render_images: bool,
        try_cover_trajectory: bool,
        strength: float,
    ) -> bool:
        """Generate episode using universal scene system."""
        from drake_env.logging import EpisodeLogger
        
        try:
            print(f"Seed {seed}: Generating episode for {scene.label}")
            
            # Import appropriate scene generator
            # Frame generation is fixed at 4 fps.
            if scene.label == "scene_a":
                from drake_env.scene_a import generate_scene_a_episode
                episode_data = generate_scene_a_episode(
                    seed=seed,
                    fps=4.0,
                    meshcat=self.meshcat,
                    randomize=strength,
                )
            elif scene.label == "scene_b":
                from drake_env.scene_b import generate_scene_b_episode
                episode_data = generate_scene_b_episode(
                    seed=seed,
                    fps=4.0,
                    meshcat=self.meshcat,
                    randomize=strength,
                )
            elif scene.label == "scene_c":
                from drake_env.scene_c import generate_scene_c_episode
                episode_data = generate_scene_c_episode(
                    seed=seed,
                    fps=4.0,
                    meshcat=self.meshcat,
                    randomize=strength,
                )
            elif scene.label == "scene_d":
                from drake_env.scene_d import generate_scene_d_episode
                episode_data = generate_scene_d_episode(
                    letter=getattr(scene, 'letter', 'C'),
                    seed=seed,
                    fps=4.0,
                    meshcat=self.meshcat,
                    randomize=strength,
                )
            else:
                raise ValueError(f"Unknown scene: {scene.name}")
            
            # Check task success from metadata (letter must reach shelf goal).
            meta_in = episode_data.get("metadata", {}) if isinstance(episode_data, dict) else {}
            success_flag = bool(meta_in.get("success", True))
            task_flag = bool(meta_in.get("task_success", True))
            if not (success_flag and task_flag):
                print(
                    f"  Seed {seed}: episode failed letter-goal check "
                    f"(success={success_flag}, task_success={task_flag}); skipping save"
                )
                # Optionally export failure HTML for debugging
                html_failed = save_options.html_path(scene, seed, failed=True)
                # if html_failed and self.meshcat:
                #     try:
                #         html_content = self.meshcat.StaticHtml()
                #         html_failed.write_text(html_content, encoding="utf-8")
                #         print(f"Saved MeshCat failure HTML to {html_failed}")
                #     except Exception as e:
                #         print(f"Failed to export failure HTML: {e}")
                return False

            # Save episode using EpisodeLogger for successful episodes only
            episode_dir = save_options.episode_dir(scene, seed)
            episode_dir.mkdir(parents=True, exist_ok=True)
            
            logger = EpisodeLogger(str(episode_dir.parent))
            logger.start(episode_dir.name)
            
            # Log all timesteps
            for i in range(len(episode_data["images_front"])):
                logger.log(
                    idx=i,
                    img_front=episode_data["images_front"][i],
                    img_wrist=episode_data["images_wrist"][i],
                    state=episode_data["states"][i],
                    action=episode_data["actions"][i],
                    contact=episode_data.get("contacts", [{"force": 0.0}])[i],
                )
            
            # Save metadata
            meta = {
                "seed": seed,
                "language": episode_data["metadata"].get("language", ""),
                "subgoals": episode_data["metadata"].get("subgoals", []),
                "scene_name": scene.label,
            }
            if hasattr(scene, 'letter'):
                meta["letter"] = scene.letter
            logger.finish(meta)
            
            # Export HTML visualization
            html_path = save_options.html_path(scene, seed, failed=False)
            if html_path and self.meshcat:
                try:
                    html_content = self.meshcat.StaticHtml()
                    html_path.write_text(html_content, encoding="utf-8")
                    print(f"Saved MeshCat HTML to {html_path}")
                except Exception as e:
                    print(f"Failed to export HTML: {e}")
            
            print(f"Seed {seed}: Episode generated successfully")
            
            return True
            
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            return False


class DirectDrakeGenerator:
    """Episode generator using drake_env directly (no DrakeWorld wrapper).
    
    Used for scenes that require direct access to drake_env for specialized features
    like physics-based grasping, full randomization, or custom scene configurations.
    Automatically adapts to the scene_type specified in SceneConfig.
    
    The generic build_scene() function routes to appropriate scene implementations
    based on scene_type, so this generator works for any scene configuration.
    """
    
    def __init__(self, meshcat=None, scene_type="pick_place_block"):
        self.meshcat = meshcat
        # Create a minimal params object for metadata compatibility
        self.params = WorldParams(
            add_gripper=True,
            add_floor=False,
            camera_res=(640, 480),
            scene_type=scene_type,
            arm="franka",
        )
    
    def generate_and_save(
        self,
        scene: SceneConfig,
        seed: int,
        save_options: SaveOptions,
        *,
        render_images: bool,
        try_cover_trajectory: bool,
        strength: float,
    ) -> bool:
        """Generate and save an episode using the improved planner."""
        import numpy as np
        from drake_env.scenes import build_scene, sample_randomization
        from drake_env.logging import EpisodeLogger
        
        # Sample randomization parameters
        rng = np.random.default_rng(seed)
        rand = sample_randomization(strength, rng)
        
        # Variables for cleanup
        scene_handles = None
        
        try:
            print(f"Seed {seed}: Building scene...")
            
            # Build scene with scene_type from config
            scene_handles = build_scene(
                'pick_place', 
                rand, 
                image_size=(640, 480),
                meshcat=self.meshcat,
                scene_type=scene.scene_type  # Use scene_type from SceneConfig
            )
            
            # Import the physics-based planner from drake_env
            from drake_env.planners import compute_subgoals, plan_and_rollout
            
            # Compute subgoals based on block position
            print(f"Seed {seed}: Computing subgoals...")
            subgoals = compute_subgoals(scene_handles, 'pick_place')
            
            # Execute the trajectory with RRT-Connect planning
            # Use shorter duration to reduce memory usage (10.0 s = 40.0 frames at 4Hz)
            print(f"Seed {seed}: Planning trajectory with RRT-Connect...")
            episode_data = plan_and_rollout(
                scene_handles,
                subgoals,
                episode_length_sec=10.0,
                hz=4,
                use_rrt=True,
                rng=rng
            )
            
            # Save episode using EpisodeLogger
            episode_dir = save_options.episode_dir(scene, seed)
            episode_dir.mkdir(parents=True, exist_ok=True)
            
            logger = EpisodeLogger(str(episode_dir.parent))
            logger.start(episode_dir.name)
            
            # Compute ee_state and ee_delta for all timesteps
            plant = scene_handles.plant
            plant_context = plant.GetMyMutableContextFromRoot(scene_handles.context)
            
            try:
                ee_frame = plant.GetBodyByName("panda_hand").body_frame()
            except:
                ee_frame = plant.GetBodyByName("panda_link8").body_frame()
            
            prev_ee_state = None
            for i in range(len(episode_data["images_front"])):
                # Compute end-effector state for this timestep
                q = episode_data["states"][i]["q"]
                plant.SetPositions(plant_context, q)
                
                X_WE = plant.CalcRelativeTransform(plant_context, plant.world_frame(), ee_frame)
                ee_pos = X_WE.translation()
                ee_rot_rpy = X_WE.rotation().ToRollPitchYaw().vector()
                ee_gripper = q[7] if len(q) >= 9 else 0.0
                ee_state = np.concatenate([ee_pos, ee_rot_rpy, [ee_gripper]])
                
                # Compute action delta
                if prev_ee_state is not None:
                    ee_delta = ee_state - prev_ee_state
                else:
                    ee_delta = np.zeros(7)
                prev_ee_state = ee_state.copy()
                
                # Add ee_state and ee_delta to state and action dicts
                episode_data["states"][i]["ee_state"] = ee_state
                episode_data["actions"][i]["ee_delta"] = ee_delta
                
                # Log timestep
                logger.log(
                    idx=i,
                    img_front=episode_data["images_front"][i],
                    img_wrist=episode_data["images_wrist"][i],
                    state=episode_data["states"][i],
                    action=episode_data["actions"][i],
                    contact=episode_data["contacts"][i],
                )
            
            # Save metadata
            meta = {
                "seed": seed,
                "language": episode_data["language"],
                "subgoals": episode_data["subgoals"],
                "randomization": {
                    "friction": rand.friction,
                    "block_color": list(rand.block_color),
                    "block_pos_x": rand.block_pos_x,
                    "block_pos_y": rand.block_pos_y,
                    "block_yaw": rand.block_yaw,
                    "bin_pos_x": rand.bin_pos_x,
                    "bin_pos_y": rand.bin_pos_y,
                    "camera_radius": rand.camera_radius,
                    "camera_azimuth": rand.camera_azimuth,
                    "camera_elevation": rand.camera_elevation,
                },
            }
            logger.finish(meta)
            
            # Export HTML on success if requested
            html_path = save_options.html_path(scene, seed, failed=False)
            if html_path and scene_handles.meshcat:
                try:
                    html_content = scene_handles.meshcat.StaticHtml()
                    html_path.write_text(html_content, encoding="utf-8")
                    print(f"Saved MeshCat HTML to {html_path}")
                except Exception as e:
                    print(f"Failed to export HTML: {e}")
            
            print(f"Seed {seed}: Episode generated successfully")
            
            # Explicit cleanup to free memory
            del episode_data
            if scene_handles is not None:
                scene_handles.simulator.clear_monitor()
                del scene_handles
            import gc
            gc.collect()
            
            return True
            
        except KeyboardInterrupt:
            print(f"Interrupted by user")
            # Cleanup on interrupt
            if scene_handles is not None:
                try:
                    scene_handles.simulator.clear_monitor()
                    del scene_handles
                except:
                    pass
            import gc
            gc.collect()
            raise
            
        except Exception as e:
            print(f"Episode generation failed for seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            
            # Export HTML on failure if requested and scene was built
            html_failed = save_options.html_path(scene, seed, failed=True)
            if html_failed and scene_handles is not None and scene_handles.meshcat:
                try:
                    html_content = scene_handles.meshcat.StaticHtml()
                    html_failed.write_text(html_content, encoding="utf-8")
                    print(f"Saved MeshCat HTML to {html_failed}")
                except Exception as html_err:
                    print(f"Failed to export failure HTML: {html_err}")
            
            # Cleanup on failure
            if scene_handles is not None:
                try:
                    scene_handles.simulator.clear_monitor()
                    del scene_handles
                except:
                    pass
            import gc
            gc.collect()
            
            return False


# Backward compatibility alias
BlockPickPlaceGenerator = DirectDrakeGenerator


def save_dataset_metadata(
    output_root: Path,
    scenes: Iterable[SceneConfig],
    params: WorldParams,
) -> None:
    metadata = {
        "scenes": [scene.__dict__ for scene in scenes],
        "world_params": params.__dict__,
    }
    (output_root / "episode_generator.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

