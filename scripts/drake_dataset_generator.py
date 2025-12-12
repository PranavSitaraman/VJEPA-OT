from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# pydrake imports (only used when you execute this file in your own env)
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseKinematics,
    Meshcat,
    MeshcatVisualizer,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PiecewisePolynomial,
    PrismaticJoint,
    Quaternion,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    SceneGraph,
    Simulator,
    Solve,
    Sphere,
    Box,
    Cylinder,
    CoulombFriction,
    SpatialInertia,
    RotationalInertia,
    UnitInertia,
    VectorBase,
)
from pydrake.geometry import GeometryId, MakePhongIllustrationProperties, PerceptionProperties, Role, Rgba

from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    IrisInConfigurationSpace,
    IrisOptions,
    Point,
)

from pydrake.planning import GcsTrajectoryOptimization

# Use manipulation package for finding robot model resources
from manipulation.utils import FindResource
from manipulation.letter_generation import create_sdf_asset_from_letter
from pydrake.multibody.parsing import LoadModelDirectives, ProcessModelDirectives
from manipulation.station import LoadScenario, MakeHardwareStation

try:
    from pydrake.all import (
        RgbdSensor,
        CameraInfo,
        DepthRenderCamera,
        DepthRange,
        RenderCameraCore,
        ClippingRange,
        ColorRenderCamera,
        MakeRenderEngineVtk,
        RenderEngineVtkParams,
    )
    _HAS_RENDERING = True
except Exception as e:
    print(f"Rendering not available: {e}")
    _HAS_RENDERING = False


# -----------------------------
# Utility geometry / math
# -----------------------------

def file_uri(path: str) -> str:
    """Return a file:// URI for an absolute or relative path."""
    abs_path = path if os.path.isabs(path) else os.path.abspath(path)
    return f"file://{abs_path}"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def look_at(eye_W: np.ndarray,
            target_W: np.ndarray,
            up_W: np.ndarray = np.array([0.0, 0.0, 1.0])) -> RigidTransform:
    """Create a Drake RigidTransform that 'looks' from eye towards target.
    
    Drake camera convention: +Z forward, +X right, +Y down in image.
    """
    # Camera looks along +Z axis
    z = target_W - eye_W
    z = z / np.linalg.norm(z)
    
    # X-axis points right (perpendicular to up and forward)
    x = np.cross(up_W, z)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-6:
        # Degenerate up-vector; pick an arbitrary orthogonal
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
        x = x / np.linalg.norm(x)
    else:
        x = x / x_norm
    
    # Y-axis points down in image (negative of world up projection)
    y = np.cross(z, x)
    
    # Invert Y to match Drake's camera convention (Y-down)
    R = RotationMatrix(np.column_stack([x, -y, z]))
    return RigidTransform(R, eye_W)


def project_points_W_to_image(
    X_WC: RigidTransform,
    K: Tuple[float, float, float, float],  # fx, fy, cx, cy
    pts_W: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D world points to image coordinates using pinhole intrinsics K.
    Returns (uv, mask_in_front). Points behind camera are masked out.
    """
    assert pts_W.shape[1] == 3
    # Transform to camera frame
    X_CW = X_WC.inverse()
    pts_C = (X_CW.rotation().matrix() @ pts_W.T + X_CW.translation().reshape(3,1)).T
    z = pts_C[:, 2:3]
    mask = z[:, 0] > 1e-4
    fx, fy, cx, cy = K
    u = fx * (pts_C[:, 0] / z[:, 0]) + cx
    v = fy * (pts_C[:, 1] / z[:, 0]) + cy
    uv = np.stack([u, v], axis=1)
    return uv, mask


# -----------------------------
# Drake world construction
# -----------------------------

@dataclass
class DomainRandomization:
    light_intensity: float = 1.0
    clutter_count: int = 2
    friction: float = 0.9
    block_color: Tuple[float,float,float, float] = (1.0, 0.2, 0.2, 1.0)
    camera_jitter: float = 0.0
    letter_pose_noise: float = 1.0
    goal_jitter: float = 1.0
    mustard_presence: float = 1.0

@dataclass
class WorldParams:
    time_step: float = 0.002  # 2ms plant discrete time
    table_xy: Tuple[float, float] = (0.6, 0.8)   # Reachable workspace on table (X: forward, Y: side-to-side)
    table_h: float = 0.26                        # table height (with manipulation package table)
    table_origin_W: Tuple[float, float, float] = (0.5, 0.0, 0.0)  # Sample objects forward from robot base
    add_floor: bool = False  # Manipulation package table provides visual grounding
    arm: str = "franka"  # only "franka" is supported in this project
    add_gripper: bool = True
    letter_initial: str = "P"  # Letter to use as object
    assets_dir: str = "assets"  # Directory for generated assets
    scene_type: str = "letter_on_stand"  # "letter_on_stand" or "letter_shelf"
    # Safety margins
    min_object_to_table_edge: float = 0.10  # keep the object inside reachable area
    min_camera_distance: Tuple[float, float] = (1.0, 2.5)  # meters (min, max radius from ROI)
    camera_fov_y_deg: float = 60.0
    camera_res: Tuple[int, int] = (640, 480)
    # Planning options
    use_gcs_corridor: bool = True
    gcs_velocity_scale: float = 1.0
    gcs_rounding_seed: int = 42
    gcs_max_paths: int = 6
    gcs_timeout: float = 1.5  # Seconds for solver (if supported)
    iris_require_start_in_region: bool = False  # Require start configs to be in IRIS region (expensive)
    shelf_assets: str = "assets/"
    shelf_randomize_items: bool = True
    shelf_item_variants: Optional[List[str]] = None


@dataclass
class RichShelfConfig:
    directives_path: Path
    clutter_slots: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]]
    rotate_shelf_deg: float = 180.0
    translation: Tuple[float, float, float] = (0.9, 0.0, 0.3995)


@dataclass
class EpisodeConfig:
    seed: int
    object_pose_W: RigidTransform
    q_start: np.ndarray
    q_goal: np.ndarray
    X_WC: RigidTransform
    K: Tuple[float, float, float, float]  # fx, fy, cx, cy
    waypoints: List[np.ndarray]  # joint-space path (7 DoF arm positions)
    gripper_commands: List[float]  # gripper position at each waypoint (0.0=closed, 0.107=open)
    success: bool
    reason: str = ""
    goal_pose_W: Optional[RigidTransform] = None
    # Waypoints now include synchronized gripper commands
    # For full pick-and-place: approach (open) → grasp (close) → lift → place (open)


class DrakeWorld:
    def __init__(self, params: WorldParams, meshcat: Optional[Meshcat] = None):
        self.params = params
        self.meshcat = meshcat
        
        # Ensure letter asset exists (with graceful fallback if triangulation unavailable)
        os.makedirs(params.assets_dir, exist_ok=True)
        self._ensure_letter_asset(params.letter_initial, params.assets_dir)

        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=params.time_step)
        self.parser = Parser(self.plant)

        # Register package paths - use manipulation's approach
        package_map = self.parser.package_map()
        manipulation_path = FindResource("")
        package_map.Add("manipulation", manipulation_path)
        try:
            if not package_map.Contains("drake_models"):
                drake_models_root = Path(FindResource("models/manipulation_station/table_wide.sdf")).resolve().parent.parent
                package_map.Add("drake_models", str(drake_models_root))
        except Exception as exc:
            print(f"Failed to register drake_models package: {exc}")

        # Track model handles we might need per-scene
        self.arm_model = None
        self.gripper_model = None
        self.ee_frame = None
        self.arm_joint_names: List[str] = []
        self.table_model = None
        self.stand_model = None
        self.shelf_model = None
        self.mustard_model = None
        self.mustard_body = None
        self.object_model = None
        self.object_body = None
        self.shelf_object_models: List[ModelInstanceIndex] = []
        self.shelf_object_bodies: List[Any] = []
        self.object_attached: bool = False
        self._X_GO_attached: Optional[RigidTransform] = None
        self._gripper_collision_ids: Set[GeometryId] = set()
        self._shelf_collision_ids: Set[GeometryId] = set()

        # Add Franka Panda arm (7-DOF + hand) for non-shelf scenes.
        # For shelf scenes, the panda + hand robot is loaded via model directives
        # in _add_rich_shelf_environment().
        if self.params.scene_type != "letter_shelf":
            if params.arm != "franka":
                raise ValueError(f"Only 'franka' arm is supported, got '{params.arm}'")

            # Add 7-DOF Panda arm (no hand) and a separate hand model, mirroring shelf_scene.yaml
            arm_url = "package://drake_models/franka_description/urdf/panda_arm.urdf"
            hand_url = "package://drake_models/franka_description/urdf/hand.urdf"
            self.arm_model = self.parser.AddModelsFromUrl(arm_url)[0]
            self.gripper_model = self.parser.AddModelsFromUrl(hand_url)[0]

            # Weld Panda base to world. This matches the reference shelf setup.
            base_frame = self.plant.GetFrameByName("panda_link0", self.arm_model)
            X_WBase = RigidTransform([0.0, 0.0, 0.0])
            self.plant.WeldFrames(self.plant.world_frame(), base_frame, X_WBase)

            # Weld hand to the end-effector link
            # The Franka hand frame needs to be offset from panda_link8 to match the physical connection
            # panda_link8 is the flange, and the hand mounts to it with a specific offset
            arm_ee_frame = self.plant.GetFrameByName("panda_link8", self.arm_model)
            hand_frame = self.plant.GetFrameByName("panda_hand", self.gripper_model)
            # The hand is offset in Z (forward) by approximately 0.1034m from the flange
            # This matches the Franka documentation for the flange-to-hand offset
            X_FlangeToHand = RigidTransform(p=[0, 0, 0.1034])
            self.plant.WeldFrames(arm_ee_frame, hand_frame, X_FlangeToHand)

            # End-effector frame is the hand
            self.ee_frame = hand_frame

            # Franka has 7 arm joints (panda_joint1..7). We treat the hand as welded.
            self.arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]

            # Default configuration: neutral pose for Franka.
            self.default_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # radians

        # Add base environment (table + stand, or rich shelf scene)
        self._add_table()

        # For shelf scenes, the robot and letter are created by directives; wire up handles here.
        if self.params.scene_type == "letter_shelf":
            if not self.plant.HasModelInstanceNamed("panda") or not self.plant.HasModelInstanceNamed("hand"):
                raise RuntimeError("Shelf scene expects 'panda' and 'hand' models from shelf_scene.yaml.")
            self.arm_model = self.plant.GetModelInstanceByName("panda")
            self.gripper_model = self.plant.GetModelInstanceByName("hand")
            self.ee_frame = self.plant.GetFrameByName("panda_hand", self.gripper_model)
            self.arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
            self.default_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # radians

            # Letter model instance is named 'letter' in the directives.
            if self.plant.HasModelInstanceNamed("letter"):
                self.object_model = self.plant.GetModelInstanceByName("letter")
                self.object_body = self.plant.GetBodyByName(
                    f"{self.params.letter_initial}_body_link", self.object_model
                )
            else:
                raise RuntimeError("Shelf scene expects a 'letter' model instance from shelf_scene.yaml.")
        else:
            # Add object (letter) as a free body for non-shelf scenes.
            letter_sdf_path = os.path.join(
                params.assets_dir,
                f"{params.letter_initial}_model",
                f"{params.letter_initial}.sdf",
            )
            # Use absolute path directly - parser handles it correctly
            letter_abs_path = os.path.abspath(letter_sdf_path)
            self.object_model = self.parser.AddModels(letter_abs_path)[0]
            self.object_body = self.plant.GetBodyByName(
                f"{params.letter_initial}_body_link", self.object_model
            )

        # Optionally add a ground plane
        if params.add_floor:
            # A thin box at z = -0.005
            ground_model = self.plant.AddModelInstance("ground_model")
            ground = self.plant.AddRigidBody(
                "ground", ground_model, SpatialInertia(1.0, [0, 0, 0], UnitInertia.SolidBox(10.0, 10.0, 0.01))
            )
            self.plant.WeldFrames(self.plant.world_frame(), ground.body_frame())
            self.plant.RegisterCollisionGeometry(
                ground, RigidTransform([0, 0, -0.005]), Box(10.0, 10.0, 0.01), "ground_collision", CoulombFriction(0.9, 0.5)
            )
            self.plant.RegisterVisualGeometry(ground, RigidTransform([0,0,-0.005]), Box(10.0, 10.0, 0.01), "ground_visual", np.array([0.8, 0.8, 0.8, 1.0]))

        self.plant.Finalize()

        if _HAS_RENDERING:
            self.scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(RenderEngineVtkParams()))

        if self.meshcat is not None:
            MeshcatVisualizer.AddToBuilder(self.builder, self.scene_graph, self.meshcat)

        self.diagram = self.builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.diagram.GetMutableSubsystemContext(self.plant, self.diagram_context)
        self.sg_context = self.diagram.GetMutableSubsystemContext(self.scene_graph, self.diagram_context)

        self._cache_geometry_sets()

        # Cache joint limits
        self.q_lower = []
        self.q_upper = []
        for name in self.arm_joint_names:
            joint = self.plant.GetJointByName(name)
            self.q_lower.append(joint.position_lower_limits())
            self.q_upper.append(joint.position_upper_limits())
        self.q_lower = np.array(self.q_lower).reshape(-1)
        self.q_upper = np.array(self.q_upper).reshape(-1)

        # Cache for IRIS regions per context fingerprint (prevents recomputation)
        self._iris_region_cache: Dict[str, HPolyhedron] = {}
        self._iris_region_disabled: Set[str] = set()
        
        # Separate plant for IRIS (excludes floating objects, builds on demand)
        self._iris_plant: Optional[MultibodyPlant] = None
        self._iris_plant_context: Optional[Any] = None
        self._iris_scene_graph: Optional[SceneGraph] = None
        self._iris_arm_model: Optional[ModelInstanceIndex] = None

    # ----- geometry helpers -----

    def _add_table(self):
        """Add table plus optional fixtures (stand, shelf, obstacles)."""
        parser = Parser(self.plant)

        # Table (only add explicitly in non-shelf scenes; shelf directives include their own table)
        if self.params.scene_type != "letter_shelf":
            self.table_model = parser.AddModels(FindResource("models/table.sdf"))[0]
            X_WT = RigidTransform(
                RotationMatrix.MakeZRotation(-np.pi/2),
                [0.0, 0.0, -0.05]
            )
            self.plant.WeldFrames(
                self.plant.world_frame(),
                self.plant.GetFrameByName("table_link", self.table_model),
                X_WT
            )

        if self.params.scene_type != "letter_shelf":
            # Classic stand for letter-on-stand tasks
            self.stand_model = parser.AddModels(FindResource("models/stand.sdf"))[0]
            X_WS = RigidTransform([0.5, 0.0, 0.0])
            self.plant.WeldFrames(
                self.plant.world_frame(),
                self.plant.GetFrameByName("stand_body", self.stand_model),
                X_WS
            )
        else:
            self._add_rich_shelf_environment()

    def _add_rich_shelf_environment(self):
        """Load shelf scene assets mirroring the notebook setup."""
        base_directives_path = (
            Path(self.params.shelf_assets) / "shelf_scene.yaml"
            if self.params.shelf_assets
            else Path(__file__).resolve().parent.parent / "shelf_scene.yaml"
        )
        
        # Read the base YAML and replace the letter model path dynamically
        import yaml
        with open(base_directives_path, 'r') as f:
            yaml_content = f.read()
        
        # Replace the hardcoded letter P with the actual letter
        letter = self.params.letter_initial
        assets_path = Path(self.params.shelf_assets) if self.params.shelf_assets else Path(__file__).resolve().parent.parent / "assets"
        letter_model_path = f"file:///{assets_path.resolve()}/{letter}_model/{letter}.sdf"
        
        # Replace the letter model file path in the YAML
        yaml_content = yaml_content.replace(
            "file:///n/holylabs/ydu_lab/Lab/pranavsitaraman/JEPA-OT/assets/P_model/P.sdf",
            letter_model_path
        )
        
        # Write to a temporary file and load it
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write(yaml_content)
            tmp_path = tmp.name
        
        try:
            directives = LoadModelDirectives(tmp_path)
            ProcessModelDirectives(directives, self.parser)
        finally:
            Path(tmp_path).unlink()  # Clean up temp file

        # Cache key models for later use
        self.table_model = self.plant.GetModelInstanceByName("table") if self.plant.HasModelInstanceNamed("table") else None
        self.shelf_model = self.plant.GetModelInstanceByName("shelves")
        self.stand_model = self.plant.GetModelInstanceByName("stand") if self.plant.HasModelInstanceNamed("stand") else None
        self.mustard_model = self.plant.GetModelInstanceByName("mustard") if self.plant.HasModelInstanceNamed("mustard") else None

        inspector = self.scene_graph.model_inspector()
        if self.shelf_model is not None:
            for body in self.plant.GetBodyIndices(self.shelf_model):
                frame_id = self.plant.GetBodyFrameIdOrThrow(body)
                for gid in inspector.GetGeometries(frame_id, Role.kProximity):
                    self._shelf_collision_ids.add(gid)

        # Optional clutter diversification
        if self.params.shelf_randomize_items and self.shelf_model is not None:
            self._randomize_shelf_items(inspector)

        self.mustard_body = None
        if self.mustard_model is not None:
            try:
                self.mustard_body = self.plant.GetBodyByName("base_link", self.mustard_model)
            except Exception:
                bodies = self.plant.GetBodyIndices(self.mustard_model)
                if bodies:
                    self.mustard_body = self.plant.get_body(bodies[0])

    def _randomize_shelf_items(self, inspector):
        # Drake 1.45's Python bindings do not expose a stable API for
        # reading/writing default poses of arbitrary bodies in a way that is
        # portable across minor versions. To keep the shelf scene robust inside
        # the provided Singularity image, we disable per-object jitter here.
        #
        # The rest of the environment (letter pose, mustard placement,
        # camera pose, start configuration, and motion planning) still
        # provides substantial randomization.
        return

    # ----- randomization helpers -----

    def _blend(self, base: float, sample: float, strength: float) -> float:
        return (1.0 - strength) * base + strength * sample

    def _rgba_to_array(self, color: Rgba) -> np.ndarray:
        return np.array([float(color.r()), float(color.g()), float(color.b()), float(color.a())])

    def _make_rotation_with_y_axis(self, dir_world: np.ndarray) -> RotationMatrix:
        direction = np.asarray(dir_world, dtype=float)
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([0.0, 1.0, 0.0])
        y_axis = direction / np.linalg.norm(direction)
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(y_axis, up)) > 0.98:
            up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(y_axis, up)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis /= x_norm
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        return RotationMatrix(np.column_stack([x_axis, y_axis, z_axis]))

    def _cache_geometry_sets(self) -> None:
        self._gripper_collision_ids.clear()
        self._shelf_collision_ids.clear()
        if self.gripper_model is not None:
            inspector = self.scene_graph.model_inspector()
            for body in self.plant.GetBodyIndices(self.gripper_model):
                frame_id = self.plant.GetBodyFrameIdOrThrow(body)
                geom_ids = inspector.GetGeometries(frame_id, Role.kProximity)
                self._gripper_collision_ids.update(geom_ids)
        if self.shelf_model is not None:
            inspector = self.scene_graph.model_inspector()
            for body in self.plant.GetBodyIndices(self.shelf_model):
                frame_id = self.plant.GetBodyFrameIdOrThrow(body)
                geom_ids = inspector.GetGeometries(frame_id, Role.kProximity)
                self._shelf_collision_ids.update(geom_ids)

    def _signed_distance_gripper_to_shelf(self, q: np.ndarray) -> float:
        if not self._gripper_collision_ids or not self._shelf_collision_ids:
            return 1.0
        saved_q = self.plant.GetPositions(self.plant_context, self.arm_model).copy()
        self.plant.SetPositions(self.plant_context, self.arm_model, q)
        query_output = self.scene_graph.get_query_output_port().Eval(self.sg_context)
        distances = query_output.ComputeSignedDistancePairwiseClosestPoints()
        min_dist = float("inf")
        for pair in distances:
            id_a = pair.id_A
            id_b = pair.id_B
            if (
                (id_a in self._gripper_collision_ids and id_b in self._shelf_collision_ids)
                or (id_b in self._gripper_collision_ids and id_a in self._shelf_collision_ids)
            ):
                min_dist = min(min_dist, pair.distance)
        if min_dist == float("inf"):
            min_dist = 1.0
        self.plant.SetPositions(self.plant_context, self.arm_model, saved_q)
        return min_dist

    def export_scene_html(self, file_path: Path | str, q_override: Optional[np.ndarray] = None) -> bool:
        """Write a MeshCat HTML snapshot of the current world configuration."""
        if self.meshcat is None:
            print("MeshCat not available; skipping HTML export")
            return False
        saved_q: Optional[np.ndarray] = None
        try:
            if q_override is not None:
                saved_q = self.plant.GetPositions(self.plant_context, self.arm_model).copy()
                self.plant.SetPositions(self.plant_context, self.arm_model, q_override)
            self.diagram.ForcedPublish(self.diagram_context)
            if hasattr(self.meshcat, "StaticHtml"):
                html = self.meshcat.StaticHtml()
            else:
                print("MeshCat instance lacks StaticHtml(); skipping HTML export")
                return False
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            print(f"Saved MeshCat HTML to {path}")
            return True
        except Exception as exc:
            print(f"Failed to export MeshCat HTML: {exc}")
            return False
        finally:
            if saved_q is not None:
                self.plant.SetPositions(self.plant_context, self.arm_model, saved_q)

    def _design_shelf_keyframes(
        self,
        X_WO: RigidTransform,
        letter_goal: Optional[RigidTransform],
        rng: np.random.Generator,
        strength: float,
    ) -> Dict[str, RigidTransform]:
        p_WO = X_WO.translation()

        # --- Pick: top-down approach to avoid the shelf during grasping ---
        yaw_span = self._blend(np.deg2rad(6.0), np.deg2rad(55.0), strength)
        yaw_center = self._blend(0.0, rng.uniform(-yaw_span, yaw_span), min(strength * 0.5 + 0.2, 1.0))
        yaw_pick = np.clip(yaw_center, -yaw_span, yaw_span)
        R_pick = RotationMatrix.MakeXRotation(-np.pi / 2.0) @ RotationMatrix.MakeZRotation(yaw_pick)
        yaw_candidates: List[float] = []
        for frac in [0.0, 0.45, -0.45, 0.85, -0.85, 1.0, -1.0]:
            cand = np.clip(yaw_pick + frac * yaw_span, -yaw_span, yaw_span)
            if all(abs(cand - existing) > 1e-3 for existing in yaw_candidates):
                yaw_candidates.append(cand)

        grasp_height = self._blend(0.012, 0.018, strength)
        pre_clearance = self._blend(0.14, 0.20, strength)
        lift_height = self._blend(0.32, 0.38, strength)

        p_grasp = p_WO + np.array([0.0, 0.0, grasp_height])
        p_pre = p_grasp + np.array([0.0, 0.0, pre_clearance])
        p_lift = p_grasp + np.array([0.0, 0.0, lift_height])

        # Clamp above table to guard against noisy samples below the surface
        min_pick_height = self.params.table_h + 0.08
        p_grasp[2] = max(p_grasp[2], min_pick_height)
        p_pre[2] = max(p_pre[2], p_grasp[2] + 0.10)
        p_lift[2] = max(p_lift[2], p_grasp[2] + lift_height)

        X_pregrasp = RigidTransform(R_pick, p_pre.tolist())
        X_grasp = RigidTransform(R_pick, p_grasp.tolist())
        X_lift = RigidTransform(R_pick, p_lift.tolist())

        # --- Place: horizontal insertion into shelf ---
        shelf_target = (
            letter_goal.translation() if letter_goal is not None else np.array([0.78, 0.0, 0.45])
        )

        side_bias = 0.10 * np.sign(shelf_target[1])
        base_dir_place = np.array([1.0, side_bias, 0.0])
        jitter_place = np.array([
            rng.uniform(-0.1, 0.1) * strength,
            rng.uniform(-0.05, 0.05) * strength,
            0.0,
        ])
        approach_place = base_dir_place + jitter_place
        approach_place[2] = 0.0
        if np.linalg.norm(approach_place) < 1e-6:
            approach_place = np.array([1.0, 0.0, 0.0])
        R_place = self._make_rotation_with_y_axis(approach_place)
        place_xhat = R_place.matrix()[:, 0]
        place_yhat = R_place.matrix()[:, 1]
        place_zhat = R_place.matrix()[:, 2]

        lateral_place = self._blend(0.02 * np.sign(shelf_target[1]), rng.uniform(-0.02, 0.02), strength)
        pre_place_depth = self._blend(0.26, 0.33, strength)
        pre_place_raise = self._blend(0.13, 0.18, strength)
        place_depth = self._blend(0.085, 0.11, strength)
        place_raise = self._blend(0.018, 0.030, strength)

        p_pre_place = (
            shelf_target
            - pre_place_depth * place_yhat
            + lateral_place * place_xhat
            + pre_place_raise * place_zhat
        )
        p_place = (
            shelf_target
            - place_depth * place_yhat
            + lateral_place * place_xhat
            + place_raise * place_zhat
        )

        # Ensure we do not command poses behind the shelf backing
        p_pre_place[0] = min(p_pre_place[0], shelf_target[0] - 0.01)
        p_place[0] = min(p_place[0], shelf_target[0] - 0.015)

        X_pre_place = RigidTransform(R_place, p_pre_place.tolist())
        X_place = RigidTransform(R_place, p_place.tolist())

        # Workspace planning using convex corridor to avoid shelf collisions
        corridor_points = self._plan_shelf_workspace_path(p_lift, p_pre_place)
        corridor_frames: List[RigidTransform] = []
        R_corridor = R_pick
        for pt in corridor_points[1:-1]:  # drop start/goal
            corridor_frames.append(RigidTransform(R_corridor, pt.tolist()))

        ready_offset_x = max(p_pre[0] - 0.18, 0.35)
        ready_height = max(p_pre[2] + 0.20, self.params.table_h + 0.55)
        ready_pose = np.array([ready_offset_x, np.clip(p_pre[1], -0.25, 0.25), ready_height])
        X_pregrasp_ready = RigidTransform(R_pick, ready_pose.tolist())

        return {
            "pregrasp": X_pregrasp,
            "grasp": X_grasp,
            "lift": X_lift,
            "pre_place": X_pre_place,
            "place": X_place,
            "pre_place_corridor": corridor_frames,
            "pregrasp_yaws": yaw_candidates,
            "pregrasp_ready": X_pregrasp_ready,
        }

    def _plan_shelf_workspace_path(self, p_start: np.ndarray, p_goal: np.ndarray) -> List[np.ndarray]:
        """Plan a collision-free Cartesian corridor from lift to pre-place.

        Uses deterministic convex boxes and a detour waypoint that keeps the hand
        safely above and in front of the shelf before inserting toward the goal.
        Returns a list of waypoints (start, optional mid, goal).
        """

        z_table = self.params.table_h

        # Workspace limits for motion (approximate reachable volume)
        lb = np.array([0.30, -0.40, z_table + 0.05])
        ub = np.array([0.92, 0.40, z_table + 0.85])

        # Define convex keep-out regions approximating shelf face and table
        shelf_front = HPolyhedron.MakeBox(
            np.array([0.76, -0.22, z_table + 0.28]),
            np.array([0.95, 0.22, z_table + 0.82]),
        )
        table_block = HPolyhedron.MakeBox(
            np.array([0.25, -0.55, z_table - 0.03]),
            np.array([0.92, 0.55, z_table + 0.06]),
        )

        # Mid waypoint: come forward of shelf, stay high, center laterally between start and goal
        mid = np.array([
            min(p_goal[0] - 0.08, 0.78),
            np.clip((p_start[1] + p_goal[1]) * 0.5, lb[1] + 0.05, ub[1] - 0.05),
            max(p_start[2], p_goal[2], z_table + 0.50),
        ])

        # Project mid out of obstacles if needed
        def project_out(point: np.ndarray) -> np.ndarray:
            adjusted = point.copy()
            for obstacle in (shelf_front, table_block):
                if obstacle.PointInSet(adjusted):
                    signed_dist = obstacle.Eval(adjusted)
                    # Eval returns Ax - b; push along obstacle normal
                    A = obstacle.A()
                    b = obstacle.b()
                    # Find most violated halfspace (max Ax - b)
                    vals = (A @ adjusted) - b
                    idx = int(np.argmax(vals))
                    normal = A[idx]
                    depth = vals[idx]
                    adjusted = adjusted - (depth + 1e-3) * normal / (np.linalg.norm(normal) + 1e-9)
            adjusted = np.maximum(adjusted, lb + 1e-3)
            adjusted = np.minimum(adjusted, ub - 1e-3)
            return adjusted

        mid = project_out(mid)

        # Ensure mid is meaningfully different from start/goal
        if np.linalg.norm(mid - p_start) < 1e-4 or np.linalg.norm(mid - p_goal) < 1e-4:
            arch_height = max(p_start[2], p_goal[2]) + 0.18
            mid = np.array([
                min((p_start[0] + p_goal[0]) * 0.5, p_goal[0] - 0.05),
                np.clip((p_start[1] + p_goal[1]) * 0.5, lb[1] + 0.05, ub[1] - 0.05),
                np.clip(arch_height, lb[2], ub[2] - 1e-3),
            ])

        return [p_start, mid, p_goal]

    # ----- IRIS + GCS helpers -----

    def _reset_iris_planning_plant(self) -> None:
        """Invalidate IRIS planning plant cache.
        
        Call this at the start of each episode to ensure IRIS regions are computed
        with the current object poses (letter, mustard) as obstacles.
        """
        self._iris_plant = None
        self._iris_plant_context = None
        self._iris_scene_graph = None
        self._iris_region_cache.clear()

    def _build_iris_planning_plant(self) -> None:
        """Build a separate plant for IRIS that excludes floating objects.
        
        The planning plant has:
        - Arm and gripper (same as main plant)
        - Table and shelf fixtures (same as main plant)
        - Letter and mustard as WELDED bodies at their current poses
        
        This allows IRIS to work over just the 7-DOF arm configuration space.
        """
        from pydrake.geometry import GeometryInstance
        
        # Create fresh builder
        builder = DiagramBuilder()
        iris_plant, iris_scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=self.params.time_step)
        parser = Parser(iris_plant)
        
        # Register manipulation package
        package_map = parser.package_map()
        manipulation_path = FindResource("")
        package_map.Add("manipulation", manipulation_path)
        
        # Add arm (same as main plant). Only Franka is supported.
        if self.params.arm != "franka":
            raise ValueError(f"Unsupported arm type for IRIS planning: {self.params.arm}")

        arm_url = "package://drake_models/franka_description/urdf/panda_arm.urdf"
        iris_arm_model = parser.AddModelsFromUrl(arm_url)[0]
        iris_gripper_model = None  # Hand is not needed in the IRIS planning plant

        # Remember which model instance is the arm in the IRIS plant
        self._iris_arm_model = iris_arm_model
        
        # Add table
        iris_table_model = parser.AddModels(FindResource("models/table.sdf"))[0]
        X_WT = RigidTransform(
            RotationMatrix.MakeZRotation(-np.pi/2),
            [0.0, 0.0, -0.05]
        )
        iris_plant.WeldFrames(
            iris_plant.world_frame(),
            iris_plant.GetFrameByName("table_link", iris_table_model),
            X_WT
        )
        
        # Add stand or shelf depending on scene type
        if self.params.scene_type != "letter_shelf":
            iris_stand_model = parser.AddModels(FindResource("models/stand.sdf"))[0]
            X_WS = RigidTransform([0.5, 0.0, 0.0])
            iris_plant.WeldFrames(
                iris_plant.world_frame(),
                iris_plant.GetFrameByName("stand_body", iris_stand_model),
                X_WS
            )
        else:
            # Add shelf structure (same as main plant)
            iris_shelf_model = iris_plant.AddModelInstance("shelf")
            mass = 5.0
            shelf_unit_inertia = UnitInertia.SolidBox(0.6, 0.25, 0.8)
            shelf_body = iris_plant.AddRigidBody(
                "shelf_body",
                iris_shelf_model,
                SpatialInertia(mass, np.zeros(3), shelf_unit_inertia),
            )
            shelf_pose = RigidTransform([0.8, 0.0, 0.4])
            iris_plant.WeldFrames(iris_plant.world_frame(), shelf_body.body_frame(), shelf_pose)
            
            # Add shelf platforms
            shelf_levels = [0.10, 0.35, 0.60]
            for i, z in enumerate(shelf_levels):
                platform_pose = RigidTransform([0.0, 0.0, z - 0.01])
                platform_shape = Box(0.6, 0.25, 0.02)
                platform_name = f"shelf_platform_{i}"
                iris_plant.RegisterCollisionGeometry(
                    shelf_body,
                    platform_pose,
                    platform_shape,
                    f"{platform_name}_collision",
                    CoulombFriction(0.9, 0.8),
                )
            
            # Add shelf walls
            wall_shapes = [
                (RigidTransform([0.0, 0.12, 0.4]), Box(0.02, 0.01, 0.8)),
                (RigidTransform([0.0, -0.12, 0.4]), Box(0.02, 0.01, 0.8)),
                (RigidTransform([-0.25, 0.0, 0.4]), Box(0.02, 0.25, 0.8)),
            ]
            for pose, geom in wall_shapes:
                iris_plant.RegisterCollisionGeometry(
                    shelf_body, pose, geom, f"wall_collision", CoulombFriction(0.9, 0.8)
                )
        
        # Add letter as WELDED obstacle at current pose
        X_WO = self.plant.GetFreeBodyPose(self.plant_context, self.object_body)
        iris_letter_model = iris_plant.AddModelInstance(f"{self.params.letter_initial}_obstacle")
        letter_body = iris_plant.AddRigidBody(
            f"{self.params.letter_initial}_obstacle_body",
            iris_letter_model,
            SpatialInertia(0.02, np.zeros(3), UnitInertia.SolidBox(0.12, 0.12, 0.05)),
        )
        iris_plant.WeldFrames(iris_plant.world_frame(), letter_body.body_frame(), X_WO)
        # Add letter collision geometry (box approximation)
        iris_plant.RegisterCollisionGeometry(
            letter_body,
            RigidTransform(),
            Box(0.12, 0.12, 0.05),
            "letter_obstacle_collision",
            CoulombFriction(1.0, 1.0),
        )
        
        # Add mustard as WELDED obstacle at current pose (if present)
        if self.mustard_body is not None:
            X_WM = self.plant.GetFreeBodyPose(self.plant_context, self.mustard_body)
            iris_mustard_model = iris_plant.AddModelInstance("mustard_obstacle")
            mustard_body = iris_plant.AddRigidBody(
                "mustard_obstacle_body",
                iris_mustard_model,
                SpatialInertia(0.2, np.zeros(3), UnitInertia.SolidCylinder(0.03, 0.14)),
            )
            iris_plant.WeldFrames(iris_plant.world_frame(), mustard_body.body_frame(), X_WM)
            iris_plant.RegisterCollisionGeometry(
                mustard_body,
                RigidTransform(),
                Cylinder(0.03, 0.14),
                "mustard_obstacle_collision",
                CoulombFriction(0.5, 0.4),
            )
        
        # Finalize and build
        iris_plant.Finalize()
        diagram = builder.Build()
        
        # Store plant and context
        self._iris_plant = iris_plant
        self._iris_scene_graph = iris_scene_graph
        iris_diagram_context = diagram.CreateDefaultContext()
        self._iris_plant_context = iris_plant.GetMyContextFromRoot(iris_diagram_context)
        
        # Set arm to match main plant's current configuration
        q_arm = self.plant.GetPositions(self.plant_context, self.arm_model)
        iris_plant.SetPositions(self._iris_plant_context, iris_arm_model, q_arm)

    def _get_or_build_iris_region(self, label: str, q_seed: np.ndarray) -> Optional[HPolyhedron]:
        """Return (and cache) an IRIS configuration-space region around q_seed.
        
        Uses a separate planning plant that treats floating objects (letter, mustard)
        as welded obstacles. This allows IRIS to work over just the 7-DOF arm
        configuration space, matching the approach from the GCS notebook.
        """

        if label in self._iris_region_disabled:
            return None

        key = (label, tuple(np.round(q_seed, 3)))
        if key in self._iris_region_cache:
            return self._iris_region_cache[key]

        try:
            # Build planning plant on first use (caches objects at their current poses)
            if self._iris_plant is None:
                self._build_iris_planning_plant()

            # Get the arm model instance from the IRIS plant
            if self._iris_arm_model is None:
                raise RuntimeError("IRIS planning plant arm model not initialized.")
            iris_arm_model = self._iris_arm_model
            
            # Save current arm configuration in planning plant
            saved_q_iris = self._iris_plant.GetPositions(self._iris_plant_context, iris_arm_model)
            
            # Set arm to seed configuration in planning plant
            self._iris_plant.SetPositions(self._iris_plant_context, iris_arm_model, q_seed)
            
            # Configure IRIS options (matching notebook settings)
            options = IrisOptions()
            options.random_seed = self.params.gcs_rounding_seed
            options.num_collision_infeasible_samples = 12
            options.require_sample_point_is_contained = True
            options.configuration_space_margin = 0.05
            options.iteration_limit = 12

            # Run IRIS on the planning plant (no floating objects, just 7-DOF arm)
            region = IrisInConfigurationSpace(self._iris_plant, self._iris_plant_context, options)
            
            # Restore planning plant state
            self._iris_plant.SetPositions(self._iris_plant_context, iris_arm_model, saved_q_iris)
            
            if region is None:
                print(f"IRIS returned None for {label}")
                self._iris_region_disabled.add(label)
                return None
            
            # Verify region dimensionality (should be 7 for arm)
            if region.ambient_dimension() != 7:
                print(f"IRIS region has wrong dimension {region.ambient_dimension()} (expected 7)")
                self._iris_region_disabled.add(label)
                return None
            
            self._iris_region_cache[key] = region
            print(f"IRIS region created for {label}: {region.A().shape[0]} halfspaces")
            return region
            
        except Exception as exc:
            print(f"IRIS failed for {label}: {exc}")
            self._iris_region_disabled.add(label)
            return None

    def _plan_shelf_gcs_path(
        self,
        q_start: np.ndarray,
        corridor_qs: List[np.ndarray],
        q_goal: np.ndarray,
    ) -> Optional[List[np.ndarray]]:
        """Solve a GCS path through cached IRIS regions; returns joint samples."""

        dim = len(q_start)
        gcs = GcsTrajectoryOptimization(dim)

        # Add source/target as fixed points
        source = gcs.AddRegions([Point(q_start)], order=0, name="source")
        target = gcs.AddRegions([Point(q_goal)], order=0, name="target")

        corridor_nodes = []
        for idx, q_mid in enumerate(corridor_qs):
            region = self._get_or_build_iris_region(f"shelf_mid_{idx}", q_mid)
            if region is None:
                return None
            node = gcs.AddRegions([region], order=1, name=f"mid_{idx}")
            corridor_nodes.append(node)

        # Connect graph (source -> mid_0 -> ... -> mid_n -> target)
        prev = source
        for node in corridor_nodes:
            gcs.AddEdges(prev, node)
            prev = node
        gcs.AddEdges(prev, target)

        # Allow useful shortcuts
        gcs.AddEdges(source, target)
        for node in corridor_nodes:
            gcs.AddEdges(source, node)
            gcs.AddEdges(node, target)
        for a, b in zip(corridor_nodes, corridor_nodes[1:]):
            gcs.AddEdges(a, b)

        gcs.AddTimeCost()
        # Use only the arm joint velocity limits (first dim entries).
        vel_lo = self.plant.GetVelocityLowerLimits()[:dim]
        vel_hi = self.plant.GetVelocityUpperLimits()[:dim]
        gcs.AddVelocityBounds(self.params.gcs_velocity_scale * vel_lo, self.params.gcs_velocity_scale * vel_hi)

        options = GraphOfConvexSetsOptions()
        options.preprocessing = True
        options.max_rounded_paths = self.params.gcs_max_paths
        options.rounding_seed = self.params.gcs_rounding_seed

        try:
            traj, result = gcs.SolvePath(source, target, options)
        except Exception as exc:
            print(f"GCS solve failed: {exc}")
            return None

        if not result.is_success():
            return None

        num_samples = max(6, 3 + 4 * len(corridor_qs))
        times = np.linspace(traj.start_time(), traj.end_time(), num_samples)
        samples = traj.vector_values(times)

        path: List[np.ndarray] = []
        for col in samples.T:
            q = np.asarray(col)
            if path and np.linalg.norm(q - path[-1]) < 1e-4:
                continue
            if not self._configuration_is_collision_free(q):
                return None
            path.append(q)

        if np.linalg.norm(path[0] - q_start) > 1e-3:
            path.insert(0, q_start.copy())
        if np.linalg.norm(path[-1] - q_goal) > 1e-3:
            path.append(q_goal.copy())

        return path

    def _ensure_letter_asset(self, letter: str, assets_dir: str) -> None:
        letter_model_dir = Path(assets_dir) / f"{letter}_model"
        sdf_path = letter_model_dir / f"{letter}.sdf"
        letter_model_dir.mkdir(parents=True, exist_ok=True)
        if sdf_path.exists():
            return
        try:
            create_sdf_asset_from_letter(
                text=letter,
                font_name="DejaVu Sans",
                letter_height_meters=0.12,
                extrusion_depth_meters=0.05,
                mass=0.02,
                output_dir=letter_model_dir,
                mu_static=1,
                use_bbox_collision_geometry=True,
            )
        except Exception as exc:
            print(f"Letter asset generation failed for '{letter}': {exc}. Using fallback block.")
            self._write_fallback_letter_sdf(sdf_path, letter)

    def _write_fallback_letter_sdf(self, sdf_path: Path, letter: str) -> None:
        mass = 0.02
        size_x, size_y, size_z = 0.04, 0.04, 0.02
        Ixx = (mass / 12.0) * (size_y ** 2 + size_z ** 2)
        Iyy = (mass / 12.0) * (size_x ** 2 + size_z ** 2)
        Izz = (mass / 12.0) * (size_x ** 2 + size_y ** 2)
        sdf_contents = f"""<?xml version='1.0'?>
<sdf version='1.10'>
  <model name='{letter}_block'>
    <link name='{letter}_body_link'>
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{Ixx:.8f}</ixx>
          <iyy>{Iyy:.8f}</iyy>
          <izz>{Izz:.8f}</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>
      <visual name='visual'>
        <geometry>
          <box>
            <size>{size_x} {size_y} {size_z}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.9 0.2 0.2 1.0</ambient>
          <diffuse>0.9 0.2 0.2 1.0</diffuse>
        </material>
      </visual>
      <collision name='collision'>
        <geometry>
          <box>
            <size>{size_x} {size_y} {size_z}</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1.0</mu>
              <mu2>1.0</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
    </link>
  </model>
</sdf>
"""
        sdf_path.write_text(sdf_contents)

    def _interpolate_path(self, q_from: np.ndarray, q_to: np.ndarray, steps: int) -> List[np.ndarray]:
        if steps <= 0:
            return [q_from.copy(), q_to.copy()]
        path: List[np.ndarray] = []
        for i in range(steps + 1):
            alpha = float(i) / float(steps)
            path.append(self._lerp_joint(q_from, q_to, alpha))
        return path

    def _path_is_collision_free(self, q_from: np.ndarray, q_to: np.ndarray, steps: int = 20) -> bool:
        for alpha in np.linspace(0.0, 1.0, steps + 1):
            q = self._lerp_joint(q_from, q_to, alpha)
            if not self._configuration_is_collision_free(q):
                return False
        return True

    def _make_rotation_with_y_axis(self, dir_world: np.ndarray) -> RotationMatrix:
        direction = np.asarray(dir_world, dtype=float)
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([0.0, 1.0, 0.0])
        y_axis = direction / np.linalg.norm(direction)
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(y_axis, up)) > 0.98:
            up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(y_axis, up)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis /= x_norm
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        return RotationMatrix(np.column_stack([x_axis, y_axis, z_axis]))

    def _sample_letter_pose_stand(self, rng: np.random.Generator, strength: float) -> RigidTransform:
        base_pos = np.array([0.5, 0.0, 0.285])
        spread = np.array([0.10, 0.10, 0.0])
        offset = (rng.uniform(-1.0, 1.0, size=3) * spread) * strength
        pos = base_pos + offset
        pos[0] = np.clip(pos[0], 0.4, 0.6)
        pos[1] = np.clip(pos[1], -0.12, 0.12)
        yaw = self._blend(0.0, rng.uniform(-np.pi, np.pi), strength)
        R = RotationMatrix.MakeZRotation(yaw)
        return RigidTransform(R, pos.tolist())

    def _sample_letter_pose_shelf(self, rng: np.random.Generator, strength: float) -> RigidTransform:
        table_z = 0.26 + 0.025
        base_pos = np.array([0.45, 0.0, table_z])
        spread = np.array([0.10, 0.18, 0.0])
        offset = (rng.uniform(-1.0, 1.0, size=3) * spread) * strength
        pos = base_pos + offset
        pos[0] = np.clip(pos[0], 0.30, 0.58)
        pos[1] = np.clip(pos[1], -0.20, 0.20)
        yaw = self._blend(0.0, rng.uniform(-np.pi, np.pi), strength)
        R = RotationMatrix.MakeZRotation(yaw)
        return RigidTransform(R, pos.tolist())

    def sample_letter_pose(self, rng: np.random.Generator, strength: float) -> RigidTransform:
        if self.params.scene_type == "letter_shelf":
            return self._sample_letter_pose_shelf(rng, strength)
        return self._sample_letter_pose_stand(rng, strength)

    def sample_shelf_goal(self, rng: np.random.Generator, strength: float) -> RigidTransform:
        base_yaw = np.pi  # facing outward
        yaw = self._blend(base_yaw, base_yaw + rng.uniform(-0.35, 0.35), strength)
        R = RotationMatrix.MakeZRotation(yaw) @ RotationMatrix.MakeXRotation(-np.pi / 2)
        shelf_y_levels = [-0.1, 0.1]
        shelf_z_levels = [0.37, 0.62]
        level_idx = 0 if strength < 0.5 else rng.integers(0, len(shelf_z_levels))
        base_pos = np.array([0.78, shelf_y_levels[level_idx % len(shelf_y_levels)], shelf_z_levels[level_idx]])
        jitter = np.array([0.04, 0.10, 0.04])
        pos = base_pos + strength * rng.uniform(-jitter, jitter)
        pos[1] = np.clip(pos[1], -0.14, 0.14)
        pos[2] = np.clip(pos[2], 0.30, 0.70)
        return RigidTransform(R, pos.tolist())

    def sample_mustard_pose(
        self,
        rng: np.random.Generator,
        strength: float,
        letter_pose: RigidTransform,
    ) -> Optional[RigidTransform]:
        if self.mustard_body is None:
            return None

        table_z = self.params.table_h + 0.07  # 7cm tall cylinder center
        letter_xy = letter_pose.translation()[:2]
        center = np.array([letter_xy[0] - 0.18, letter_xy[1] - 0.30, table_z])
        span = np.array([0.06, 0.08, 0.0])
        min_dist = 0.32

        best_pose = None
        best_dist = -np.inf
        for _ in range(40):
            offset = rng.uniform(-1.0, 1.0, size=3) * span
            candidate = center + offset
            candidate[0] = np.clip(candidate[0], 0.18, 0.55)
            candidate[1] = np.clip(candidate[1], -0.40, -0.15)
            candidate[2] = table_z
            dist = np.linalg.norm(candidate[:2] - letter_xy)
            if dist >= min_dist and dist > best_dist:
                yaw = self._blend(0.0, rng.uniform(-np.pi, np.pi), strength)
                best_pose = RigidTransform(RotationMatrix.MakeZRotation(yaw), candidate.tolist())
                best_dist = dist

        if best_pose is None:
            fallback_xy = letter_xy + np.array([-0.22, -0.36])
            fallback_xy[0] = np.clip(fallback_xy[0], 0.18, 0.55)
            fallback_xy[1] = np.clip(fallback_xy[1], -0.40, -0.15)
            yaw = rng.uniform(-np.pi, np.pi)
            best_pose = RigidTransform(
                RotationMatrix.MakeZRotation(yaw),
                [fallback_xy[0], fallback_xy[1], table_z],
            )
            dist = np.linalg.norm(fallback_xy - letter_xy)
        else:
            dist = best_dist

        if self.params.scene_type == "letter_shelf":
            print(
                f"  Mustard distance from letter: {dist:.4f} m "
                f"(letter_xy={letter_xy}, mustard_xy={(best_pose.translation()[:2])})"
            )

        return best_pose

    # ----- randomization -----

    def set_mustard_pose(self, X_WM: Optional[RigidTransform]):
        if self.mustard_body is None or X_WM is None:
            return
        self.plant.SetFreeBodyPose(self.plant_context, self.mustard_body, X_WM)

    def _attach_object_to_gripper(self, X_WG: RigidTransform) -> None:
        """Attach object to gripper by recording relative transform."""
        self.object_attached = True
        X_WO = self.plant.GetFreeBodyPose(self.plant_context, self.object_body)
        # Store the relative transform from gripper to object at grasp time
        self._X_GO_attached = X_WG.inverse() @ X_WO

    def _detach_object(self, X_WO_target: Optional[RigidTransform] = None) -> None:
        """Detach object from gripper."""
        self.object_attached = False
        self._X_GO_attached = None
        if X_WO_target is not None:
            self.set_object_pose(X_WO_target)

    def _update_attached_object_pose(self) -> None:
        """Update object pose to follow gripper during grasping.
        
        IMPORTANT: Only update when object_attached is True to prevent
        the object from tracking gripper rotation before/after grasp.
        """
        if not self.object_attached:
            return  # Don't update if not attached - prevents phantom tracking
        if self._X_GO_attached is None:
            return  # Safety check - shouldn't happen if object_attached is True
        
        # Get current gripper pose
        X_WG = self.plant.CalcRelativeTransform(
            self.plant_context, self.plant.world_frame(), self.ee_frame
        )
        # Compute and set new object pose
        X_WO = X_WG @ self._X_GO_attached
        self.set_object_pose(X_WO)

    def sample_object_pose_on_table(self, rng: np.random.Generator) -> RigidTransform:
        """Uniformly place object on the stand workspace (matches notebook setup).
        
        The stand is at [0.5, 0.0, 0.0] with top surface at z=0.25m.
        Objects are placed on the stand, not the table.
        """
        P = self.params
        # Stand is 30cm x 30cm, centered at [0.5, 0.0, 0.0]
        # Sample within stand bounds with margin
        margin = 0.05
        x = rng.uniform(0.5 - 0.15 + margin, 0.5 + 0.15 - margin)  # 0.4 to 0.6
        y = rng.uniform(-0.15 + margin, 0.15 - margin)  # -0.1 to 0.1
        # Stand top at z=0.25m, stand is 2cm thick, so top surface at z=0.26m
        # Letter is 5cm thick, center at z=0.26 + 0.025 = 0.285m
        z = 0.26 + 0.025  # Stand top + half letter thickness
        yaw = rng.uniform(-np.pi, np.pi)
        return RigidTransform(RotationMatrix.MakeZRotation(yaw), [x, y, z])

    def set_object_pose(self, X_WO: RigidTransform):
        self.plant.SetFreeBodyPose(self.plant_context, self.object_body, X_WO)
    
    def simulate_object_settling(self, duration: float = 1.0) -> RigidTransform:
        """Simulate object falling and settling on table with gravity."""
        from pydrake.all import Simulator
        
        # Create a temporary simulator
        simulator = Simulator(self.diagram)
        simulator_context = simulator.get_mutable_context()
        
        # Copy current state to simulator
        simulator_context.SetTime(0.0)
        simulator_context.SetContinuousState(self.diagram_context.get_continuous_state_vector().CopyToVector())
        
        # Simulate forward
        simulator.AdvanceTo(duration)
        
        # Copy final state back
        self.diagram_context.SetContinuousState(simulator_context.get_continuous_state_vector().CopyToVector())
        
        # Return settled object pose
        return self.plant.GetFreeBodyPose(self.plant_context, self.object_body)

    def random_start_configuration(self, rng: np.random.Generator, strength: float = 1.0, max_tries: int = 200) -> Optional[np.ndarray]:
        """Rejection-sample a collision-free joint configuration, blending with nominal."""

        if strength <= 0.0:
            q = self.default_q.copy()
            self.plant.SetPositions(self.plant_context, self.arm_model, q)
            if self.is_collision_free():
                return q
            strength = 1e-3  # fall back to tiny randomization if nominal collides

        for _ in range(max_tries):
            sample = rng.uniform(self.q_lower, self.q_upper)
            q = self._lerp_joint(self.default_q, sample, strength)
            self.plant.SetPositions(self.plant_context, self.arm_model, q)
            if self.is_collision_free():
                if self.params.scene_type == "letter_shelf":
                    dist = self._signed_distance_gripper_to_shelf(q)
                    if dist < 0.03:
                        print(f"Reject start config: shelf distance {dist:.3f} m < 0.03 m")
                        continue
                if self.params.use_gcs_corridor and self.params.iris_require_start_in_region:
                    region = self._get_or_build_iris_region("start", q)
                    if region is None or not region.PointInSet(q):
                        continue
                return q
        return None

    # ----- collision queries -----

    def is_collision_free(self) -> bool:
        """Returns True if plant at current context is collision-free (no penetration).
        
        Allows gripper-stand contact (needed for grasping objects on stand).
        Also allows object-stand/table contact (object sits on surface).
        """
        query_object = self.scene_graph.get_query_output_port().Eval(self.sg_context)
        pairs = query_object.ComputePointPairPenetration()
        
        # Use small tolerance to avoid floating point precision issues
        depth_threshold = 1e-3  # 1mm tolerance
        
        for p in pairs:
            if p.depth > depth_threshold:
                # Get bodies involved in collision
                inspector = query_object.inspector()
                frame_A = inspector.GetFrameId(p.id_A)
                frame_B = inspector.GetFrameId(p.id_B)
                body_A = self.plant.GetBodyFromFrameId(frame_A)
                body_B = self.plant.GetBodyFromFrameId(frame_B)
                model_A = body_A.model_instance()
                model_B = body_B.model_instance()
                
                # Allow gripper-stand collisions (needed for grasping on stand)
                if self.stand_model is not None and (
                    (model_A == self.gripper_model and model_B == self.stand_model)
                    or (model_B == self.gripper_model and model_A == self.stand_model)
                ):
                    continue

                # Allow gripper-table collisions (needed for low grasps)
                if (model_A == self.gripper_model and model_B == self.table_model) or \
                   (model_B == self.gripper_model and model_A == self.table_model):
                    continue

                # Allow gripper-mustard collisions (gripper may pass near mustard during approach)
                if self.mustard_model is not None and (
                    (model_A == self.gripper_model and model_B == self.mustard_model)
                    or (model_B == self.gripper_model and model_A == self.mustard_model)
                ):
                    continue

                # Allow mustard resting on table
                if self.mustard_model is not None and (
                    (model_A == self.mustard_model and model_B == self.table_model)
                    or (model_B == self.mustard_model and model_A == self.table_model)
                ):
                    continue

                # Allow object-stand collisions (object sits on stand)
                if self.stand_model is not None and (
                    (model_A == self.object_model and model_B == self.stand_model)
                    or (model_B == self.object_model and model_A == self.stand_model)
                ):
                    continue

                # Allow object-table collisions (object may touch table edge)
                if (model_A == self.object_model and model_B == self.table_model) or \
                   (model_B == self.object_model and model_A == self.table_model):
                    continue

                # Allow object-shelf collisions (object resting on shelf ledge)
                if self.shelf_model is not None and (
                    (model_A == self.object_model and model_B == self.shelf_model)
                    or (model_B == self.object_model and model_A == self.shelf_model)
                ):
                    continue

                # Allow mustard touching the letter (permitted contact on table)
                if self.mustard_model is not None and (
                    (model_A == self.mustard_model and model_B == self.object_model)
                    or (model_B == self.mustard_model and model_A == self.object_model)
                ):
                    continue
                
                # Any other collision is not allowed
                if self.params.scene_type == "letter_shelf":
                    name_A = self.plant.GetModelInstanceName(model_A)
                    name_B = self.plant.GetModelInstanceName(model_B)
                    print(
                        "    disallowed collision:",
                        f"{name_A} vs {name_B} depth={p.depth:.4f}"
                    )
                return False
        
        return True

    def _configuration_is_collision_free(self, q: np.ndarray) -> bool:
        self.plant.SetPositions(self.plant_context, self.arm_model, q)
        self._update_attached_object_pose()
        return self.is_collision_free()

    # ----- IK for end-effector goal near object -----
    
    def solve_ik_for_grasp(self, X_WO: RigidTransform, rng: np.random.Generator, strength: float = 1.0) -> Optional[np.ndarray]:
        """Solve IK for a 'top-down' grasp AT the letter object for actual grasping.
        
        Tries multiple random grasp orientations to find a reachable configuration.
        """
        # Try multiple random grasp orientations
        max_attempts = 20  # Increased from 10 to 20
        for attempt in range(max_attempts):
            # Define grasp pose: gripper pointing down, positioned to grasp the letter
            # Gripper pointing down: rotate -90 degrees about X axis, then random yaw
            yaw = self._blend(0.0, rng.uniform(-np.pi, np.pi), strength)
            R_grasp = RotationMatrix.MakeXRotation(-np.pi/2) @ RotationMatrix.MakeZRotation(yaw)
            # Position gripper to grasp the letter - gripper fingers should be around the letter
            # Increased from 0.05m to 0.12m to avoid gripper-stand collisions
            height = self._blend(0.10, 0.12, strength)
            grasp_in_O = RigidTransform(R_grasp, [0.0, 0.0, height])  # offset above object center
            X_WG_target = X_WO @ grasp_in_O  # goal gripper pose in world

            # Create IK problem
            ik = InverseKinematics(self.plant, self.plant_context)
            q_vars = ik.q()[:7]  # Only the 7 arm joints
            prog = ik.prog()

            # Tolerances
            pos_tol = 0.015  # 1.5cm position tolerance
            theta_bound = 0.01 * np.pi  # ~1.8 degree orientation tolerance

            # Orientation constraint
            ik.AddOrientationConstraint(
                frameAbar=self.ee_frame,
                R_AbarA=RotationMatrix(),
                frameBbar=self.plant.world_frame(),
                R_BbarB=X_WG_target.rotation(),
                theta_bound=theta_bound,
            )

            # Position constraint
            p_WG = X_WG_target.translation()
            ik.AddPositionConstraint(
                frameB=self.ee_frame,
                p_BQ=np.array([0.0, 0.0, 0.0]),
                frameA=self.plant.world_frame(),
                p_AQ_lower=p_WG - pos_tol,
                p_AQ_upper=p_WG + pos_tol,
            )

            # Joint-centering cost
            q_nominal = np.array([0.0, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0])
            prog.AddQuadraticErrorCost(np.eye(7), q_nominal, q_vars)
            prog.SetInitialGuess(q_vars, q_nominal)
            
            # Solve
            result = Solve(prog)
            if result.is_success():
                return result.GetSolution(q_vars)
        
        # All attempts failed
        return None

    def solve_ik_pose(
        self,
        X_WG_target: RigidTransform,
        q_nominal: Optional[np.ndarray] = None,
        pos_tol: float = 0.02,
        theta_bound: float = 0.02 * np.pi,
    ) -> Optional[np.ndarray]:
        """General IK solver for a desired gripper pose."""

        ik = InverseKinematics(self.plant, self.plant_context)
        q_vars = ik.q()[:7]
        prog = ik.prog()

        ik.AddOrientationConstraint(
            self.ee_frame,
            RotationMatrix(),
            self.plant.world_frame(),
            X_WG_target.rotation(),
            theta_bound,
        )

        p_WG = X_WG_target.translation()
        ik.AddPositionConstraint(
            self.ee_frame,
            np.zeros(3),
            self.plant.world_frame(),
            p_WG - pos_tol,
            p_WG + pos_tol,
        )

        if q_nominal is None:
            q_nominal = np.array([0.0, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0])

        prog.AddQuadraticErrorCost(np.eye(7), q_nominal, q_vars)
        prog.SetInitialGuess(q_vars, q_nominal)

        result = Solve(prog)
        if result.is_success():
            return result.GetSolution(q_vars)
        return None

    # ----- Simple joint-space RRT-Connect planner -----
    
    def plan_rrt_connect(self, q_start: np.ndarray, q_goal: np.ndarray, rng: np.random.Generator,
                         max_iters: int = 10000, step: float = 0.15, max_extend: int = 15) -> Optional[List[np.ndarray]]:
        """
        A minimal RRT-Connect in joint space using straight-line interpolation.
        Returns a list of waypoints [q0, ..., qN] if successful.
        step: max joint step during collision checking per local extension (L2 in joint space).
        """
        def steer(q_from, q_to, step_size):
            d = q_to - q_from
            dist = np.linalg.norm(d)
            if dist <= step_size:
                return q_to.copy(), True
            return q_from + d / dist * step_size, False

        class Tree:
            def __init__(self, root):
                self.nodes = [root.copy()]
                self.parents = [-1]

            def add(self, q, parent):
                self.nodes.append(q.copy())
                self.parents.append(parent)
                return len(self.nodes) - 1

            def nearest(self, q):
                # naive linear search
                dists = [np.linalg.norm(q - n) for n in self.nodes]
                return int(np.argmin(dists))

            def path_to_root(self, idx):
                path = []
                while idx != -1:
                    path.append(self.nodes[idx])
                    idx = self.parents[idx]
                path.reverse()
                return path

        def collision_free_along(q_from, q_to, local_step):
            # walk along the segment and check collisions
            d = q_to - q_from
            L = np.linalg.norm(d)
            if L < 1e-9:
                return True
            n = max(1, int(math.ceil(L / local_step)))
            for i in range(1, n+1):
                q = q_from + d * (i / n)
                if not self._configuration_is_collision_free(q):
                    return False
            return True

        T_a = Tree(q_start)
        T_b = Tree(q_goal)

        def try_connect(tree_a, tree_b):
            # extend tree_a towards a random sample or towards the nearest of tree_b
            for _ in range(max_extend):
                # sample toward random or towards other tree's nearest node
                if rng.random() < 0.5:
                    q_rand = rng.uniform(self.q_lower, self.q_upper)
                else:
                    q_rand = tree_b.nodes[rng.integers(len(tree_b.nodes))]
                idx = tree_a.nearest(q_rand)
                q_near = tree_a.nodes[idx]
                q_new, reached = steer(q_near, q_rand, step)
                if collision_free_along(q_near, q_new, step):
                    new_idx = tree_a.add(q_new, idx)
                    # attempt to connect to tree_b
                    idx_b = tree_b.nearest(q_new)
                    q_near_b = tree_b.nodes[idx_b]
                    if collision_free_along(q_new, q_near_b, step):
                        # success: assemble path
                        path_a = tree_a.path_to_root(new_idx)
                        path_b = tree_b.path_to_root(idx_b)
                        path = path_a + list(reversed(path_b))
                        return path
            return None

        for it in range(max_iters):
            if it % 250 == 0 and it > 0:
                print(f"RRT iteration {it}/{max_iters}, nodes: {len(T_a.nodes)}, {len(T_b.nodes)}")
            path = try_connect(T_a, T_b)
            if path is not None:
                return path
            # swap roles
            path = try_connect(T_b, T_a)
            if path is not None:
                return list(reversed(path))
        print(f"RRT failed after {max_iters} iterations")
        return None

    # ----- Camera sampling & visibility checks -----

    def sample_camera_pose(
        self,
        rng: np.random.Generator,
        roi_center_W: np.ndarray,
        radius_range: Tuple[float, float],
        strength: float = 1.0,
    ) -> Tuple[RigidTransform, Tuple[float, float, float, float]]:
        """Sample a camera pose with constraints for natural viewpoints.
        
        Camera is positioned to look down at the table from a reasonable angle,
        similar to a third-person view or overhead camera.
        """
        base_radius = np.mean(radius_range)
        rand_radius = rng.uniform(radius_range[0], radius_range[1])
        r = self._blend(base_radius, rand_radius, strength)

        az = self._blend(0.0, rng.uniform(-np.pi, np.pi), strength)
        el_base = 0.6 * np.pi
        el_rand = rng.uniform(0.52 * np.pi, 0.67 * np.pi)
        el = self._blend(el_base, el_rand, strength)
        
        # Position camera on sphere around ROI
        eye = roi_center_W + r * np.array([
            math.cos(az) * math.sin(el),
            math.sin(az) * math.sin(el),
            math.cos(el)
        ])
        
        # Ensure camera is above table (z > 0.8m)
        min_height = 0.9
        if eye[2] < min_height:
            eye[2] = min_height + self._blend(0.0, rng.uniform(0.0, 0.4), strength)
        
        # Create camera transform with Z-up constraint
        X_WC = look_at(eye, roi_center_W, up_W=np.array([0.0, 0.0, 1.0]))
        
        # intrinsics from FOV and image size
        width, height = self.params.camera_res
        fy = (height/2) / math.tan(0.5 * math.radians(self.params.camera_fov_y_deg))
        fx = fy  # square pixels
        cx, cy = width/2, height/2
        return X_WC, (fx, fy, cx, cy)

    def camera_covers_trajectory(self, X_WC: RigidTransform, K, waypoints: List[np.ndarray], thresh_frac: float = 0.85) -> bool:
        """Check if a given camera sees most of the EE trajectory (projected within image bounds)."""
        ee_pts = []
        for q in waypoints:
            self.plant.SetPositions(self.plant_context, self.arm_model, q)
            X_WE = self.plant.CalcRelativeTransform(self.plant_context, self.plant.world_frame(), self.ee_frame)
            ee_pts.append(X_WE.translation())
        pts_W = np.vstack(ee_pts)
        width, height = self.params.camera_res
        uv, mask_front = project_points_W_to_image(X_WC, K, pts_W, width, height)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height) & mask_front
        frac = np.mean(inside.astype(float))
        return frac >= thresh_frac

    # ----- Multi-stage trajectory generation -----
    
    def _lerp_joint(self, q_base: np.ndarray, q_target: np.ndarray, blend: float) -> np.ndarray:
        return (1.0 - blend) * q_base + blend * q_target

    def plan_pick_and_place_trajectory(
        self,
        q_start: np.ndarray,
        X_WO: RigidTransform,
        rng: np.random.Generator,
        strength: float,
        X_goal: Optional[RigidTransform] = None,
    ) -> Optional[Tuple[List[np.ndarray], List[float]]]:
        """Plan pick-place with optional placement goal."""

        max_attempts = 12
        for attempt in range(max_attempts):
            if attempt > 0 and attempt % 4 == 0:
                print(f"Planning attempt {attempt}/{max_attempts}...")

            if self.params.scene_type == "letter_shelf":
                keyframes = self._design_shelf_keyframes(X_WO, X_goal, rng, strength)
                X_WG_pregrasp = keyframes["pregrasp"]
                X_WG_grasp = keyframes["grasp"]
                X_WG_lift = keyframes["lift"]
                X_WG_pre_place = keyframes["pre_place"] if X_goal is not None else None
                X_WG_place = keyframes["place"] if X_goal is not None else None
            else:
                yaw_pre = self._blend(0.0, rng.uniform(-np.pi, np.pi), strength)
                offset_pre = self._blend(0.08, 0.12, strength)
                X_WG_pregrasp = X_WO @ RigidTransform(
                    RotationMatrix.MakeXRotation(-np.pi / 2) @ RotationMatrix.MakeZRotation(yaw_pre),
                    [0.0, 0.0, offset_pre],
                )
                X_WG_grasp = X_WO
                X_WG_lift = X_WO @ RigidTransform([0.0, 0.0, self._blend(0.25, 0.35, strength)])
                X_WG_pre_place = X_goal @ RigidTransform([0.0, 0.0, 0.12]) if X_goal is not None else None
                X_WG_place = X_goal if X_goal is not None else None

            q_pregrasp = self.solve_ik_pose(X_WG_pregrasp, q_nominal=self.default_q)
            if (
                q_pregrasp is None
                or not self._configuration_is_collision_free(q_pregrasp)
                or (
                    self.params.scene_type == "letter_shelf"
                    and self._signed_distance_gripper_to_shelf(q_pregrasp) < 0.02
                )
            ) and self.params.scene_type == "letter_shelf":
                yaw_list = keyframes.get("pregrasp_yaws", [])
                fallback_found = False
                for yaw in yaw_list:
                    R_retry = RotationMatrix.MakeXRotation(-np.pi / 2.0) @ RotationMatrix.MakeZRotation(yaw)
                    X_retry = RigidTransform(R_retry, X_WG_pregrasp.translation())
                    q_retry = self.solve_ik_pose(X_retry, q_nominal=self.default_q)
                    if (
                        q_retry is not None
                        and self._configuration_is_collision_free(q_retry)
                        and self._signed_distance_gripper_to_shelf(q_retry) >= 0.02
                    ):
                        q_pregrasp = q_retry
                        X_WG_pregrasp = X_retry
                        fallback_found = True
                        break
                if not fallback_found:
                    q_pregrasp = None
            if q_pregrasp is None:
                if attempt == 0:
                    print(f"Attempt {attempt}: pregrasp IK failed")
                continue
            if not self._configuration_is_collision_free(q_pregrasp):
                if attempt == 0:
                    print(f"Attempt {attempt}: pregrasp collides")
                continue
            if self.params.scene_type == "letter_shelf":
                dist = self._signed_distance_gripper_to_shelf(q_pregrasp)
                if dist < 0.02:
                    if attempt == 0:
                        print(f"Attempt {attempt}: pregrasp shelf distance {dist:.3f} m")
                    continue

            q_grasp = self.solve_ik_pose(X_WG_grasp, q_nominal=q_pregrasp, pos_tol=0.015, theta_bound=0.01 * np.pi)
            if q_grasp is None:
                if attempt == 0:
                    print(f"Attempt {attempt}: grasp IK failed")
                continue
            if not self._configuration_is_collision_free(q_grasp):
                if attempt == 0:
                    print(f"Attempt {attempt}: grasp collides")
                continue

            path_to_pregrasp = None
            if self.params.scene_type == "letter_shelf" and self.params.use_gcs_corridor:
                high_ready = keyframes.get("pregrasp_ready")
                if high_ready is not None:
                    q_ready = self.solve_ik_pose(high_ready, q_nominal=q_start, pos_tol=0.03)
                    if q_ready is not None and self._configuration_is_collision_free(q_ready):
                        ready_dist = self._signed_distance_gripper_to_shelf(q_ready)
                        if ready_dist >= 0.03:
                            gcs_path = self._plan_shelf_gcs_path(q_start, [q_ready], q_pregrasp)
                            if gcs_path is not None:
                                path_to_pregrasp = gcs_path

            if path_to_pregrasp is None:
                path_to_pregrasp = self.plan_rrt_connect(q_start, q_pregrasp, rng)
                if path_to_pregrasp is None:
                    if not self._path_is_collision_free(q_start, q_pregrasp, steps=24):
                        continue
                    path_to_pregrasp = self._interpolate_path(q_start, q_pregrasp, 24)

            waypoints: List[np.ndarray] = []
            gripper_cmds: List[float] = []
            for q in path_to_pregrasp:
                waypoints.append(q)
                gripper_cmds.append(0.107)

            waypoints.append(q_grasp)
            gripper_cmds.append(0.107)

            # Attach object and close gripper
            self._attach_object_to_gripper(X_WG_grasp)
            waypoints.append(q_grasp)
            gripper_cmds.append(0.0)

            q_lift = self.solve_ik_pose(X_WG_lift, q_nominal=q_grasp, pos_tol=0.02)
            if q_lift is None or not self._configuration_is_collision_free(q_lift):
                self._detach_object()
                continue
            waypoints.append(q_lift)
            gripper_cmds.append(0.0)

            if self.params.scene_type == "letter_shelf" and X_goal is not None:
                corridor_frames = keyframes.get("pre_place_corridor", [])
                q_pre_place = self.solve_ik_pose(X_WG_pre_place, q_nominal=q_lift, pos_tol=0.03)
                if q_pre_place is None or not self._configuration_is_collision_free(q_pre_place):
                    self._detach_object()
                    continue

                path_to_place: Optional[List[np.ndarray]] = None

                if self.params.use_gcs_corridor and corridor_frames:
                    corridor_qs: List[np.ndarray] = []
                    q_seed = q_lift
                    corridor_valid = True
                    for idx, frame in enumerate(corridor_frames):
                        q_mid = self.solve_ik_pose(frame, q_nominal=q_seed, pos_tol=0.03)
                        if q_mid is None or not self._configuration_is_collision_free(q_mid):
                            corridor_valid = False
                            break
                        corridor_qs.append(q_mid)
                        q_seed = q_mid

                    if corridor_valid:
                        gcs_path = self._plan_shelf_gcs_path(q_lift, corridor_qs, q_pre_place)
                        if gcs_path is not None:
                            path_to_place = gcs_path
                            print("Using GCS corridor for shelf approach")

                if path_to_place is None:
                    path_to_place = self.plan_rrt_connect(q_lift, q_pre_place, rng, step=0.12)
                    if path_to_place is None:
                        if not self._path_is_collision_free(q_lift, q_pre_place, steps=24):
                            self._detach_object()
                            continue
                        path_to_place = self._interpolate_path(q_lift, q_pre_place, 24)

                for q in path_to_place[1:]:
                    waypoints.append(q)
                    gripper_cmds.append(0.0)

                q_place = self.solve_ik_pose(X_WG_place, q_nominal=path_to_place[-1], pos_tol=0.02)
                if q_place is None or not self._configuration_is_collision_free(q_place):
                    self._detach_object()
                    continue
                waypoints.append(q_place)
                gripper_cmds.append(0.0)

                # Detach object at shelf goal and open gripper
                self._detach_object(X_goal)
                waypoints.append(q_place)
                gripper_cmds.append(0.107)
            else:
                lift_offset = self._blend(0.25, 0.35, strength)
                X_upper = X_WO @ RigidTransform([0.0, 0.0, lift_offset])
                q_upper = self.solve_ik_pose(X_upper, q_nominal=q_lift)
                if q_upper is None or not self._configuration_is_collision_free(q_upper):
                    self._detach_object()
                    continue
                waypoints.append(q_upper)
                gripper_cmds.append(0.0)
                self._detach_object()
                waypoints.append(q_upper)
                gripper_cmds.append(0.107)

            return waypoints, gripper_cmds

        self._detach_object()
        return None

    # ----- Episode generation -----

    def generate_episode(self, seed: int, strength: float = 1.0, render_images: bool = False, try_cover_trajectory: bool = True) -> EpisodeConfig:
        rng = np.random.default_rng(seed)

        print(f"Seed {seed}: Sampling object pose...")
        X_WO = self.sample_letter_pose(rng, strength)
        self.set_object_pose(X_WO)

        mustard_pose = self.sample_mustard_pose(rng, strength, X_WO)
        self.set_mustard_pose(mustard_pose)

        # Reset IRIS planning plant cache so it's built with current object poses
        if self.params.use_gcs_corridor:
            self._reset_iris_planning_plant()

        q_start = self.random_start_configuration(rng)
        if q_start is None:
            print(f"Seed {seed}: Failed to find collision-free start configuration")
            return EpisodeConfig(seed, X_WO, np.zeros(7), np.zeros(7), RigidTransform(), (0,0,0,0), [], [], False, "no_start_config")
        self.plant.SetPositions(self.plant_context, self.arm_model, q_start)
        print(f"Seed {seed}: Start config collision-free: {self.is_collision_free()}")

        placement_goal = None
        if self.params.scene_type == "letter_shelf":
            placement_goal = self.sample_shelf_goal(rng, strength)

        plan = self.plan_pick_and_place_trajectory(q_start, X_WO, rng, strength, placement_goal)
        if plan is None:
            return EpisodeConfig(seed, X_WO, q_start, np.zeros_like(q_start), RigidTransform(), (0,0,0,0), [], [], False, "planning_failed", placement_goal)
        waypoints, gripper_commands = plan

        q_goal = waypoints[-1]

        print(f"Seed {seed}: Running camera sampling...")
        self.plant.SetPositions(self.plant_context, self.arm_model, q_start)
        X_WE_start = self.plant.CalcRelativeTransform(self.plant_context, self.plant.world_frame(), self.ee_frame)
        self.plant.SetPositions(self.plant_context, self.arm_model, q_goal)
        X_WE_goal = self.plant.CalcRelativeTransform(self.plant_context, self.plant.world_frame(), self.ee_frame)
        roi = 0.5 * (X_WE_start.translation() + X_WE_goal.translation())

        max_camera_tries = 200
        X_WC = None
        K = None
        for i in range(max_camera_tries):
            if i % 50 == 0:
                print(f"Seed {seed}: Camera attempt {i}/{max_camera_tries}...")
            X_candidate, K_candidate = self.sample_camera_pose(rng, roi, self.params.min_camera_distance, strength)
            
            # Check if camera is too close to robot base (avoid collision).
            # Franka base is welded at the world origin.
            camera_pos = X_candidate.translation()
            robot_base = np.array([0.0, 0.0, 0.0])
            if np.linalg.norm(camera_pos[:2] - robot_base[:2]) < 0.8:  # Too close in XY plane
                continue
            
            if try_cover_trajectory and not self.camera_covers_trajectory(X_candidate, K_candidate, waypoints):
                continue
            X_WC, K = X_candidate, K_candidate
            break
        if X_WC is None:
            # fall back to just seeing start+goal (looser requirement)
            print(f"Seed {seed}: Camera failed to cover trajectory, trying looser requirement...")
            for i in range(max_camera_tries):
                if i % 50 == 0:
                    print(f"Seed {seed}: Fallback camera attempt {i}/{max_camera_tries}...")
                X_candidate, K_candidate = self.sample_camera_pose(rng, roi, self.params.min_camera_distance, strength)
                
                # Check if camera is too close to robot base
                camera_pos = X_candidate.translation()
                robot_base = np.array([0.0, 0.0, 0.0])
                if np.linalg.norm(camera_pos[:2] - robot_base[:2]) < 0.8:
                    continue
                
                width, height = self.params.camera_res
                pts = np.vstack([X_WE_start.translation(), X_WE_goal.translation()])
                uv, mask = project_points_W_to_image(X_candidate, K_candidate, pts, width, height)
                inside = (uv[:,0] >= 0) & (uv[:,0] < width) & (uv[:,1] >= 0) & (uv[:,1] < height) & mask
                if inside.all():
                    X_WC, K = X_candidate, K_candidate
                    break
        if X_WC is None:
            # Final fallback: just use any camera pose that looks at the ROI
            print(f"Seed {seed}: Using fallback camera (no trajectory coverage guarantee)")
            X_WC, K = self.sample_camera_pose(rng, roi, self.params.min_camera_distance, strength)

        print(f"Seed {seed}: Camera placement successful")
        return EpisodeConfig(seed, X_WO, q_start, q_goal, X_WC, K, waypoints, gripper_commands, True, goal_pose_W=placement_goal)


# -----------------------------
# Saving / dataset I/O
# -----------------------------

def save_episode(out_dir: str, world: DrakeWorld, ep: EpisodeConfig, save_images: bool = False, num_frames: int = 30):
    """Save episode data in standard imitation learning format.
    
    Structure:
        episode_dir/
        ├── images/
        │   ├── frame_000.png
        │   ├── ...
        │   └── frame_029.png
        ├── actions.npy       # (num_frames, 8): [7 arm joints + 1 gripper]
        ├── metadata.json     # Episode info
        └── observations.npy  # (num_frames, 7): robot joint positions
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Interpolate waypoints to get exactly num_frames
    waypoints_array = np.stack(ep.waypoints, axis=0)  # (N, 7)
    gripper_array = np.array(ep.gripper_commands)     # (N,)
    
    # Linear interpolation to get num_frames samples
    t_waypoints = np.linspace(0, 1, len(ep.waypoints))
    t_frames = np.linspace(0, 1, num_frames)
    
    frames_q = np.zeros((num_frames, 7))
    frames_gripper = np.zeros(num_frames)
    
    for i in range(7):
        frames_q[:, i] = np.interp(t_frames, t_waypoints, waypoints_array[:, i])
    frames_gripper = np.interp(t_frames, t_waypoints, gripper_array)
    
    # Combine into actions: (num_frames, 8) = [7 arm + 1 gripper]
    actions = np.concatenate([frames_q, frames_gripper.reshape(-1, 1)], axis=1)
    
    # Save metadata
    meta = {
        "seed": ep.seed,
        "success": ep.success,
        "reason": ep.reason,
        "arm": world.params.arm,
        "num_frames": num_frames,
        "camera": {
            "width": world.params.camera_res[0],
            "height": world.params.camera_res[1],
            "fov_y_deg": world.params.camera_fov_y_deg,
            "fx_fy_cx_cy": list(map(float, ep.K)),
            "X_WC": {
                "R": world._mat_to_list(ep.X_WC.rotation().matrix()),
                "p": ep.X_WC.translation().tolist()
            }
        },
        "object_pose_W": {
            "R": world._mat_to_list(ep.object_pose_W.rotation().matrix()),
            "p": ep.object_pose_W.translation().tolist()
        },
        "gripper_states": {
            "open": 0.107,
            "closed": 0.0
        }
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    # Save actions (teacher labels)
    np.save(os.path.join(out_dir, "actions.npy"), actions)
    
    # Save observations (robot state)
    np.save(os.path.join(out_dir, "observations.npy"), frames_q)
    
    # Save episode data in parquet format for training
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pydrake.all import MultibodyPlant, RigidTransform
    
    # Compute end-effector states and deltas for VJEPA2-AC (7D representation)
    # Need to forward-kinematics the joint positions to get ee poses
    temp_plant = MultibodyPlant(time_step=0.001)
    from pydrake.all import Parser
    parser = Parser(temp_plant)
    # Add Franka model to compute FK
    import os as _os
    franka_urdf = _os.path.join(_os.path.dirname(__file__), "..", "assets", "franka_panda", "panda_arm_hand.urdf")
    if not _os.path.exists(franka_urdf):
        # Fallback to drake's installed models if available
        try:
            parser.AddModelsFromUrl("package://drake_models/franka_description/urdf/panda_arm_hand.urdf")
        except:
            print("Cannot load Franka model for ee_delta computation, using qdot only")
            temp_plant = None
    else:
        parser.AddModels(franka_urdf)
    
    if temp_plant is not None:
        temp_plant.Finalize()
        temp_context = temp_plant.CreateDefaultContext()
        try:
            ee_frame = temp_plant.GetBodyByName("panda_hand").body_frame()
        except:
            try:
                ee_frame = temp_plant.GetBodyByName("panda_link8").body_frame()
            except:
                temp_plant = None  # Can't find ee frame
    
    rows = []
    prev_ee_state = None
    for i in range(num_frames):
        row = {
            "t": i,
            "state_q": frames_q[i].tolist(),
            "act_qdot": frames_q[i].tolist(),  # Legacy: joint velocities
            "act_gripper": float(frames_gripper[i]),
        }
        
        # Compute end-effector state and delta (7D for VJEPA2-AC)
        if temp_plant is not None:
            q_full = np.concatenate([frames_q[i], [frames_gripper[i], frames_gripper[i]]])
            temp_plant.SetPositions(temp_context, q_full)
            X_WE = temp_plant.CalcRelativeTransform(temp_context, temp_plant.world_frame(), ee_frame)
            ee_pos = X_WE.translation()
            ee_rot_rpy = X_WE.rotation().ToRollPitchYaw().vector()
            ee_gripper = frames_gripper[i]
            ee_state = np.concatenate([ee_pos, ee_rot_rpy, [ee_gripper]])
            
            if prev_ee_state is not None:
                ee_delta = ee_state - prev_ee_state
            else:
                ee_delta = np.zeros(7)
            prev_ee_state = ee_state.copy()
            
            row["state_ee_state"] = ee_state.tolist()
            row["act_ee_delta"] = ee_delta.tolist()
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, os.path.join(out_dir, "episode.parquet"))
    
    # Render and save images at each frame
    if save_images and _HAS_RENDERING:
        print(f"Rendering {num_frames} frames...")
        
        # Build a temporary diagram with RgbdSensor for rendering
        from pydrake.all import Simulator
        import matplotlib.pyplot as plt
        
        builder_render = DiagramBuilder()
        plant_render, scene_graph_render = AddMultibodyPlantSceneGraph(builder_render, time_step=0.001)
        
        # Add robot (same as world)
        parser_render = Parser(plant_render)
        package_map_render = parser_render.package_map()
        manipulation_path = FindResource("")
        package_map_render.Add("manipulation", manipulation_path)
        
        if world.params.arm == "iiwa":
            from manipulation.scenarios import AddIiwa, AddWsg
            arm_model_render = AddIiwa(plant_render, collision_model="no_collision")
            gripper_model_render = AddWsg(plant_render, arm_model_render, welded=True)
        elif world.params.arm == "franka":
            franka_urdf_url = "package://drake_models/franka_description/urdf/panda_arm_hand.urdf"
            arm_model_render = parser_render.AddModelsFromUrl(franka_urdf_url)[0]
            gripper_model_render = arm_model_render
        else:
            raise ValueError(f"Unsupported arm type for rendering: {world.params.arm}")
        
        # Add table from manipulation package (same as planning scene)
        table_model_render = parser_render.AddModels(FindResource("models/table.sdf"))[0]
        X_WT = RigidTransform(
            RotationMatrix.MakeZRotation(-np.pi/2),
            [0.0, 0.0, -0.05]
        )
        plant_render.WeldFrames(
            plant_render.world_frame(),
            plant_render.GetFrameByName("table_link", table_model_render),
            X_WT
        )
        
        # Add stand from manipulation package (same as planning scene)
        stand_model_render = parser_render.AddModels(FindResource("models/stand.sdf"))[0]
        X_WS = RigidTransform([0.5, 0.0, 0.0])
        plant_render.WeldFrames(
            plant_render.world_frame(),
            plant_render.GetFrameByName("stand_body", stand_model_render),
            X_WS
        )
        
        # Add object
        letter_sdf_path = os.path.join(world.params.assets_dir, f"{world.params.letter_initial}_model", f"{world.params.letter_initial}.sdf")
        letter_abs_path = os.path.abspath(letter_sdf_path)
        if os.path.exists(letter_abs_path):
            object_model_render = parser_render.AddModels(letter_abs_path)[0]
            object_body_render = plant_render.GetBodyByName(f"{world.params.letter_initial}_body_link", object_model_render)
            # Set default pose - this will be overridden when we set positions in context
            plant_render.SetDefaultFreeBodyPose(object_body_render, ep.object_pose_W)
        else:
            # Fallback to default letter model
            object_model_render = parser_render.AddModels(FindResource("models/letter_A.sdf"))[0]
            object_body_render = plant_render.GetBodyByName("letter_A_body_link", object_model_render)
            plant_render.SetDefaultFreeBodyPose(object_body_render, ep.object_pose_W)
        
        plant_render.Finalize()
        
        # Add renderer to scene graph
        scene_graph_render.AddRenderer("renderer", MakeRenderEngineVtk(RenderEngineVtkParams()))
        
        # Add RgbdSensor at camera pose
        width, height = world.params.camera_res
        fx, fy, cx, cy = ep.K
        
        # Create camera intrinsics
        camera_core = RenderCameraCore(
            renderer_name="renderer",
            intrinsics=CameraInfo(width=width, height=height, focal_x=fx, focal_y=fy, center_x=cx, center_y=cy),
            clipping=ClippingRange(near=0.01, far=10.0),
            X_BS=RigidTransform()
        )
        
        color_camera = ColorRenderCamera(core=camera_core, show_window=False)
        depth_camera = DepthRenderCamera(core=camera_core, depth_range=DepthRange(0.01, 10.0))
        
        # Add sensor
        sensor = builder_render.AddSystem(
            RgbdSensor(
                parent_id=plant_render.GetBodyFrameIdOrThrow(plant_render.world_body().index()),
                X_PB=ep.X_WC,
                color_camera=color_camera,
                depth_camera=depth_camera
            )
        )
        
        builder_render.Connect(
            scene_graph_render.get_query_output_port(),
            sensor.query_object_input_port()
        )
        
        # Build and create context
        diagram_render = builder_render.Build()
        context_render = diagram_render.CreateDefaultContext()
        plant_context_render = plant_render.GetMyContextFromRoot(context_render)
        
        # Set object pose in rendering context (must match planning scene)
        plant_render.SetFreeBodyPose(plant_context_render, object_body_render, ep.object_pose_W)
        
        # Render each frame
        for i in range(num_frames):
            # Set robot pose
            plant_render.SetPositions(plant_context_render, arm_model_render, frames_q[i])
            
            # Get RGB image
            sensor_context = sensor.GetMyContextFromRoot(context_render)
            color_image = sensor.color_image_output_port().Eval(sensor_context)
            
            # Convert to numpy array and save
            from PIL import Image
            img_data = color_image.data
            img_rgb = np.reshape(img_data, (height, width, 4))[:, :, :3]  # Drop alpha channel
            
            # Save image using PIL (more compatible than cv2 on Apple Silicon)
            # Save as front_*.png and wrist_*.png (both from same camera for now)
            front_path = os.path.join(out_dir, f"front_{i:06d}.png")
            wrist_path = os.path.join(out_dir, f"wrist_{i:06d}.png")
            img_pil = Image.fromarray(img_rgb.astype(np.uint8))
            img_pil.save(front_path)
            img_pil.save(wrist_path)  # TODO: add separate wrist camera
        
        print(f"Saved {num_frames} front and wrist images to {out_dir}")
    elif save_images and not _HAS_RENDERING:
        print(f"Rendering not available. Install rendering dependencies.")
    
    # Also save legacy format for backward compatibility
    np.savez(os.path.join(out_dir, "arrays.npz"),
             q_start=ep.q_start,
             q_goal=ep.q_goal,
             waypoints=waypoints_array,
             gripper_commands=gripper_array,
             K=np.array(ep.K),
             X_WC_R=ep.X_WC.rotation().matrix(),
             X_WC_p=ep.X_WC.translation(),
             X_WO_R=ep.object_pose_W.rotation().matrix(),
             X_WO_p=ep.object_pose_W.translation())


# Small helper to convert rotation matrix to nested list (to keep JSON human-readable)
def _mat_to_list(M: np.ndarray) -> List[List[float]]:
    return [[float(M[i, j]) for j in range(3)] for i in range(3)]

# Monkey-patch onto class (to avoid cluttering the constructor above)
setattr(DrakeWorld, "_mat_to_list", staticmethod(_mat_to_list))


# -----------------------------
# CLI
# -----------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Drake embodied data generator (randomized resets + RRT teacher).")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out", type=str, default="data_runs/run_000")
    parser.add_argument("--arm", type=str, default="franka", choices=["iiwa", "franka"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_images", action="store_true", default=True, help="If set and rendering is available, saves RGB images.")
    parser.add_argument("--no_render", action="store_true", help="Disable image rendering.")
    parser.add_argument("--cover_traj", action="store_true", help="Require the camera to see most of the EE path.")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to interpolate and render per episode.")
    args = parser.parse_args()

    P = WorldParams(arm=args.arm)
    world = DrakeWorld(P)

    root = args.out
    os.makedirs(root, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    successes = 0
    render_images = args.render_images and not args.no_render
    
    for i in range(args.episodes):
        seed_i = int(rng.integers(0, 2**31-1))
        ep = world.generate_episode(seed=seed_i, render_images=False, try_cover_trajectory=args.cover_traj)
        print(f"[{i+1}/{args.episodes}] success={ep.success} reason={ep.reason}")
        if ep.success:
            run_dir = os.path.join(root, f"episode_{i:06d}")
            save_episode(run_dir, world, ep, save_images=render_images, num_frames=args.num_frames)
            successes += 1

    print(f"Finished. {successes}/{args.episodes} successful episodes written to {root}.")


if __name__ == "__main__":
    main()
