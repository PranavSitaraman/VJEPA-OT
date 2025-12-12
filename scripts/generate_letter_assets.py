from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from manipulation.letter_generation import create_sdf_asset_from_letter


@dataclass
class LetterSpec:
    letter: str
    height: float
    depth: float
    mass: float
    suffix: str  # e.g. "" or "_big"


DEFAULT_LETTERS: Sequence[str] = ("A", "C", "E", "L", "P", "R")
DEFAULT_FONT = "DejaVu Sans"

SIM_SPEC = LetterSpec(letter="?", height=0.12, depth=0.05, mass=0.01, suffix="")
PLAN_SPEC = LetterSpec(letter="?", height=0.15, depth=0.08, mass=0.01, suffix="_big")


def generate_for_letter(letter: str, specs: Iterable[LetterSpec], assets_dir: Path, font: str, friction: float, bbox_collision: bool) -> None:
    letter_upper = letter.upper()
    for spec_template in specs:
        spec = LetterSpec(
            letter=letter_upper,
            height=spec_template.height,
            depth=spec_template.depth,
            mass=spec_template.mass,
            suffix=spec_template.suffix,
        )
        output_dir = assets_dir / f"{spec.letter}{spec.suffix}_model"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Building {spec.letter}{spec.suffix} "
            f"(height={spec.height:.3f}m, depth={spec.depth:.3f}m)"
        )
        create_sdf_asset_from_letter(
            text=spec.letter,
            font_name=font,
            letter_height_meters=spec.height,
            extrusion_depth_meters=spec.depth,
            mass=spec.mass,
            output_dir=output_dir,
            mu_static=friction,
            use_bbox_collision_geometry=bbox_collision,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SDF assets for shelf letters")
    parser.add_argument(
        "--letters",
        type=str,
        default="".join(DEFAULT_LETTERS),
        help="Concatenated list of letters to generate (default: scenes A,C,E,L,P,R)",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="Root assets directory (matching notebook structure)",
    )
    parser.add_argument(
        "--font-name",
        type=str,
        default=DEFAULT_FONT,
        help="Font family to render letters with",
    )
    parser.add_argument(
        "--mu-static",
        type=float,
        default=1.0,
        help="Static friction coefficient to embed in SDF",
    )
    parser.add_argument(
        "--use-bbox-collision",
        action="store_true",
        help="Use bounding-box collision geometry for faster planning (default: convex decomposition)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    specs = [SIM_SPEC, PLAN_SPEC]
    for letter in args.letters:
        if not letter.strip():
            continue
        generate_for_letter(
            letter,
            specs,
            args.assets_dir,
            font=args.font_name,
            friction=args.mu_static,
            bbox_collision=args.use_bbox_collision,
        )


if __name__ == "__main__":
    main()
