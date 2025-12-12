from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.drake_dataset_generator import WorldParams
from scripts.episode_generation import EpisodeGenerator, SaveOptions, SceneConfig, save_dataset_metadata


SCENE_TABLE: Dict[str, SceneConfig] = {
    "a": SceneConfig("scene_a", "Shelf letter P manipulation", "P", "universal"),
    "b": SceneConfig("scene_b", "Shelf letter L manipulation", "L", "universal"),
    "c": SceneConfig("scene_c", "Shelf letter A manipulation", "A", "universal"),
    "d": SceneConfig("scene_d", "Shelf letter C manipulation", "C", "universal"),
    "e": SceneConfig("scene_e", "Table letter pick/place", "E", "letter_on_stand"),
    "f": SceneConfig("scene_f", "Table letter pick/place", "R", "letter_on_stand"),
}


def _selected_scenes(spec: str) -> List[SceneConfig]:
    cleaned = spec.replace(" ", "").lower()
    tokens = cleaned.split(",") if "," in cleaned else list(cleaned)
    requested = [token for token in tokens if token]
    unknown = [s for s in requested if s not in SCENE_TABLE]
    if unknown:
        raise ValueError(f"Unknown scene ids: {', '.join(unknown)}")
    return [SCENE_TABLE[s] for s in requested]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manipulation episodes")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per configuration")
    parser.add_argument("--scene-set", type=str, default="abcd", help="Scene labels to generate")
    parser.add_argument("--out", type=Path, default=Path("./episodes"), help="Output directory")
    parser.add_argument("--randomize", type=float, default=0.5, help="Randomization strength [0,1]")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--meshcat", action="store_true", help="Enable MeshCat visualization")
    parser.add_argument("--debug-html-dir", type=Path, default=Path("./episodes"), help="Directory for MeshCat HTML exports")
    args = parser.parse_args()

    selected = _selected_scenes(args.scene_set)
    if not selected:
        raise ValueError(f"No valid scenes selected from '{args.scene_set}'")

    # Group scenes by scene_type
    universal_scenes = [s for s in selected if s.scene_type == "universal"]
    stand_scenes = [s for s in selected if s.scene_type == "letter_on_stand"]

    meshcat = None
    # Start MeshCat whenever either interactive visualization is requested
    # or HTML export is enabled, so export_scene_html can always render.
    if args.meshcat or args.debug_html_dir:
        from pydrake.all import StartMeshcat

        meshcat = StartMeshcat()

    # Create generators for each scene type
    generators = {}
    
    # Universal scenes (A-D) - new unified system
    if universal_scenes:
        from scripts.episode_generation import SceneDReferenceGenerator
        generators["universal"] = SceneDReferenceGenerator(meshcat=meshcat)
    
    # Legacy DrakeWorld scenes (E, F) - old working system for out-of-distribution testing
    for scene in stand_scenes:
        key = f"{scene.scene_type}_{scene.letter}"
        if key not in generators:
            params = WorldParams(
                add_gripper=True,
                add_floor=False,
                camera_res=(640, 480),
                scene_type=scene.scene_type,
                letter_initial=scene.letter,
                arm="franka",
            )
            generators[key] = EpisodeGenerator(params, meshcat=meshcat)
    
    # Use first generator's params as template for metadata
    params_template = next(iter(generators.values())).params

    args.out.mkdir(parents=True, exist_ok=True)
    if args.debug_html_dir:
        args.debug_html_dir.mkdir(parents=True, exist_ok=True)

    # For legacy DrakeWorld scenes (E/F) we still need a fixed number of frames.
    # Use 4 fps with an effective clip length of ~6s → 24 frames.
    num_frames = 24

    save_opts = SaveOptions(
        output_root=args.out,
        export_html=bool(args.debug_html_dir),
        html_dir=args.debug_html_dir,
        num_frames=num_frames,
    )

    # Build initial summary from any existing episode directories to support
    # resuming generation into an existing --out directory.
    summary: Dict[str, List[str]] = {}
    for scene in selected:
        scene_root = args.out / scene.label
        ids: List[str] = []
        if scene_root.is_dir():
            for child in sorted(scene_root.iterdir()):
                if child.is_dir() and child.name.startswith(f"{scene.label}_"):
                    ids.append(child.name)
        summary[scene.label] = ids

    rng = np.random.default_rng(args.seed)

    for scene in selected:
        # Select generator based on scene type
        if scene.scene_type == "universal":
            # Scenes A-D: Universal system (same planner, different letters)
            generator = generators["universal"]
        else:
            # Scenes E-F: Legacy DrakeWorld scenes (need separate generators per letter)
            generator = generators[f"{scene.scene_type}_{scene.letter}"]

        # Ensure we save exactly args.episodes successful episodes per scene.
        # Start counting from any episodes that already exist on disk so that
        # rerunning with the same --out continues generation instead of
        # overwriting or duplicating work.
        existing_ids = summary.get(scene.label, [])
        num_success = len(existing_ids)
        attempts = 0
        # Allow a safety cap on attempts to avoid infinite loops in pathological cases
        max_attempts = max(args.episodes * 10, args.episodes + 5)

        # Track seeds already used for successful episodes for this scene,
        # inferred from existing episode IDs of the form '<label>_<seed>'.
        used_seeds = set()
        for eid in existing_ids:
            if isinstance(eid, str) and "_" in eid:
                seed_str = eid.rsplit("_", 1)[-1]
                try:
                    used_seeds.add(int(seed_str))
                except ValueError:
                    pass

        if num_success > 0:
            print(f"{scene.label} resuming from {num_success} existing episodes (target {args.episodes}).")

        while num_success < args.episodes and attempts < max_attempts:
            attempts += 1
            # Draw a new seed that has not yet been used for this scene.
            while True:
                seed_i = int(rng.integers(0, 2**31 - 1))
                if seed_i not in used_seeds:
                    break
            eid = f"{scene.label}_{seed_i:08d}"
            ok = generator.generate_and_save(
                scene,
                seed_i,
                save_opts,
                render_images=True,
                try_cover_trajectory=True,
                strength=args.randomize,
            )
            if ok:
                num_success += 1
                used_seeds.add(seed_i)
                summary[scene.label].append(eid)
                print(f"Saved {eid} (success {num_success}/{args.episodes}, attempts={attempts})")
            else:
                print(f"Failed {eid} (attempt {attempts})")

        if num_success < args.episodes:
            print(f"{scene.label} has only {num_success}/{args.episodes} successful episodes after {attempts} attempts.")

    meta = {
        "scenes": [scene.__dict__ for scene in selected],
        "randomize": args.randomize,
        "seed": args.seed,
        "fps": 4,
        "duration": None,
        "num_frames": num_frames,
        "episodes": summary,
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_dataset_metadata(args.out, selected, params_template)


if __name__ == "__main__":
    main()
