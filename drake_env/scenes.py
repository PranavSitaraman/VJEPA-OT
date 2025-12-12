from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

from pydrake.geometry import (
    SceneGraph,
    Rgba,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    GeometryInstance,
    PerceptionProperties,
    MakePhongIllustrationProperties,
)
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.systems.sensors import RgbdSensor, CameraConfig
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.perception import DepthImageToPointCloud
from pydrake.geometry import HalfSpace
from pydrake.multibody.tree import ModelInstanceIndex, SpatialInertia, UnitInertia
from pydrake.all import AddDefaultVisualization

# Table surface is at z=0.77m in world frame (standard manipulation station height)
# Ground plane (HalfSpace) is at z=0, so table is 0.77m above ground
TABLE_SURFACE_Z = 0.77
BLOCK_SIZE = 0.055  # 5.5cm cube for easier grasping
BLOCK_HALF_HEIGHT = BLOCK_SIZE / 2.0
BIN_SIZE = np.array([0.2, 0.15, 0.05])  # width, depth, height
BIN_POSITION = np.array([0.55, 0.15, TABLE_SURFACE_Z])  # x, y, z in world frame

# Weld constraint offset (block center relative to gripper in gripper frame)
# For top-down grasp with gripper pointing down, block is below gripper
# For 5.5cm block, gripper needs to be higher to allow fingers to wrap around from above
WELD_OFFSET_Z = BLOCK_SIZE + 0.04  # Gripper positioned 9.5cm above block center for proper grasp clearance

@dataclass
class DomainRandomization:
    light_intensity: float = 1.0
    clutter_count: int = 2
    friction: float = 3.0  # High friction for physics-based grasping
    block_color: Tuple[float,float,float, float] = (1.0, 0.2, 0.2, 1.0)
    camera_jitter: float = 0.0
    # Block randomization parameters
    block_pos_x: float = 0.40
    block_pos_y: float = 0.0
    block_yaw: float = 0.0
    # Goal/bin randomization parameters
    bin_pos_x: float = 0.55
    bin_pos_y: float = 0.15
    # Camera position parameters (FIXED to match Meta's VJEPA2 pretrained model)
    # Meta trained on DROID dataset using LEFT exocentric camera view (~60° left of robot)
    # We use +1.93 rad (+110.6°) to match the left exocentric view from Meta's training
    camera_radius: float = 1.46  # 1.46m distance
    camera_azimuth: float = 1.93  # +110.6° - LEFT of robot base (matches Meta's DROID training)
    camera_elevation: float = 1.07  # 61.4° - looking down at workspace

@dataclass
class SceneHandles:
    simulator: Simulator
    plant: MultibodyPlant
    scene_graph: SceneGraph
    diagram: Any
    context: Any
    ports: Dict[str, Any]
    model_indices: Dict[str, ModelInstanceIndex]
    meshcat: Optional[Any]

def _add_table(plant: MultibodyPlant, parser: Parser):
    """Add table to the scene, welded in place."""
    from pydrake.geometry import Box, ProximityProperties, AddRigidHydroelasticProperties
    from pydrake.multibody.plant import CoulombFriction
    
    try:
        table_instance = parser.AddModelsFromUrl(
            "package://drake_models/manipulation_station/table_wide.sdf"
        )[0]
        table_frame = plant.GetFrameByName("table_body", table_instance)
        # Weld table so its top surface is at TABLE_SURFACE_Z
        # The table_body frame has table top at z=0, so we translate it up by TABLE_SURFACE_Z
        X_WT = RigidTransform([0.0, 0.0, TABLE_SURFACE_Z])
        plant.WeldFrames(plant.world_frame(), table_frame, X_WT)
        
        # Add explicit collision geometry for table top surface with hydroelastic properties
        # The table SDF has collision at z=-0.05 (spans z=-0.1 to z=0 in table frame)
        # We add a rigid collision box just BELOW the surface to catch penetrations
        # Table top surface is at z=0 in table_body frame
        table_top_box = Box(1.2, 0.8, 0.15)  # 15cm thick
        table_top_pose = RigidTransform([0.0, 0.0, -0.075])  # Center at z=-0.075, spans z=-0.15 to z=0
        table_body = plant.GetBodyByName("table_body", table_instance)
        
        # Create proximity properties with rigid hydroelastic support
        table_props = ProximityProperties()
        table_props.AddProperty("material", "coulomb_friction", CoulombFriction(0.7, 0.6))
        # Add RIGID hydroelastic properties - table is infinitely stiff, zero compliance
        # This ensures the table surface provides infinite resistance to penetration
        AddRigidHydroelasticProperties(resolution_hint=0.01, properties=table_props)
        
        plant.RegisterCollisionGeometry(
            table_body,
            table_top_pose,
            table_top_box,
            "table_top_collision",
            table_props
        )
    except Exception as e:
        print(f"Could not load table: {e}")
        pass

def _add_franka(plant: MultibodyPlant, parser: Parser) -> ModelInstanceIndex:
    """Add Franka Panda arm with gripper and return model instance."""
    franka = parser.AddModelsFromUrl(
        "package://drake_models/franka_description/urdf/panda_arm_hand.urdf"
    )[0]
    # Weld the base to the world at table height
    base_frame = plant.GetFrameByName("panda_link0", franka)
    X_WBase = RigidTransform([0.0, 0.0, 0.77])  # Position on table
    plant.WeldFrames(plant.world_frame(), base_frame, X_WBase)
    return franka


def _make_rgbd_sensor(
    name: str,
    builder: DiagramBuilder,
    scene_graph: SceneGraph,
    pose: RigidTransform,
    image_size: Tuple[int, int],
    fps: float,
) -> RgbdSensor:
    cfg = CameraConfig()
    cfg.name = name
    cfg.width = int(image_size[0])
    cfg.height = int(image_size[1])
    cfg.rgb = True
    cfg.depth = True
    cfg.label = False
    cfg.show_rgb = False
    cfg.renderer_name = "renderer"
    cfg.fps = float(max(fps, 1e-6))
    cfg.focal.x = cfg.focal.y = 60.0
    cfg.clipping_near = 0.05
    cfg.clipping_far = 5.0
    cfg.z_near = 0.05
    cfg.z_far = 5.0
    color_camera, depth_camera = cfg.MakeCameras()
    sensor = RgbdSensor(
        parent_id=scene_graph.world_frame_id(),
        X_PB=pose,
        color_camera=color_camera,
        depth_camera=depth_camera,
    )
    sensor.set_name(name)
    builder.AddSystem(sensor)
    builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    return sensor

def build_scene(
    task_name: str,
    rand: DomainRandomization,
    image_size=(256,256),
    camera_fps: float = 4.0,
    meshcat: Optional[Any] = None,
    scene_type: str = "pick_place_block",
) -> SceneHandles:
    """Build a scene for manipulation tasks.
    
    Supported scene types:
    - "pick_place_block": Scene C (physics-based block pick-and-place)
    - "franka_reference": Scene D (reference notebook with Franka)
    Other scenes (a, b, e, f) use DrakeWorld with YAML scenarios.
    
    Args:
        task_name: Task identifier (e.g., "pick_place")
        rand: Domain randomization parameters (not used for franka_reference)
        image_size: Camera resolution (width, height)
        camera_fps: Camera frame rate
        meshcat: Optional Meshcat instance for visualization
        scene_type: "pick_place_block" or "franka_reference"
    
    Returns:
        SceneHandles with simulator, plant, and other components
    """
    if scene_type == "pick_place_block":
        return build_scene_c(task_name, rand, image_size, camera_fps, meshcat)
    elif scene_type == "franka_reference":
        from drake_env.scenes_franka_reference import build_scene_franka_reference
        # Map task_name to shelf-letter scenes AD; default to Scene D (letter C)
        name = task_name.lower()
        if name == "scene_a":
            letter = "P"
        elif name == "scene_b":
            letter = "L"
        elif name == "scene_c":
            letter = "A"
        else:
            # scene_d or any other label falls back to reference letter C
            letter = "C"
        return build_scene_franka_reference(
            task_name,
            letter=letter,
            image_size=image_size,
            camera_fps=camera_fps,
            meshcat=meshcat,
        )
    
    raise ValueError(f"Unknown scene_type: {scene_type}. Supported: 'pick_place_block', 'franka_reference'")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.002)
    parser = Parser(plant)

    franka_model = _add_franka(plant, parser)
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), HalfSpace(), "ground", CoulombFriction(0.9, 0.8))
    _add_table(plant, parser)

    from pydrake.geometry import Box
    block = Box(0.05, 0.05, 0.05)
    bin_box = Box(0.2, 0.15, 0.05)

    block_model = plant.AddModelInstance("block")
    block_body = plant.AddRigidBody(
        "block",
        block_model,
        SpatialInertia.MakeFromCentralInertia(
            0.5,
            np.zeros(3),
            UnitInertia.SolidBox(0.05, 0.05, 0.05),
        ),
    )
    X_WB = RigidTransform([0.5, 0, 0.77])
    plant.WeldFrames(plant.world_frame(), block_body.body_frame(), X_WB)
    block_rgba = Rgba(*rand.block_color)
    block_color = np.asarray(rand.block_color, dtype=np.float64).reshape(4, 1)
    block_instance = GeometryInstance(RigidTransform(), block.Clone(), "block_vis")
    block_instance.set_illustration_properties(MakePhongIllustrationProperties(block_color))
    block_id = plant.RegisterVisualGeometry(block_body, block_instance)
    block_props = PerceptionProperties()
    block_props.AddProperty("phong", "diffuse", Rgba(*rand.block_color))
    scene_graph.AssignRole(plant.get_source_id(), block_id, block_props)
    plant.RegisterCollisionGeometry(block_body, RigidTransform(), block, "block_col", CoulombFriction(rand.friction, rand.friction * 0.8))

    bin_model = plant.AddModelInstance("bin")
    bin_body = plant.AddRigidBody(
        "bin",
        bin_model,
        SpatialInertia.MakeFromCentralInertia(
            1.0,
            np.zeros(3),
            UnitInertia.SolidBox(0.2, 0.15, 0.05),
        ),
    )
    X_WC = RigidTransform([0.7, 0.2, 0.77])
    plant.WeldFrames(plant.world_frame(), bin_body.body_frame(), X_WC)
    bin_rgba = Rgba(0.2, 0.8, 0.2, 1.0)
    bin_color = np.array([0.2, 0.8, 0.2, 1.0], dtype=np.float64).reshape(4, 1)
    bin_instance = GeometryInstance(RigidTransform(), bin_box.Clone(), "bin_vis")
    bin_instance.set_illustration_properties(MakePhongIllustrationProperties(bin_color))
    bin_id = plant.RegisterVisualGeometry(bin_body, bin_instance)
    bin_props = PerceptionProperties()
    bin_props.AddProperty("phong", "diffuse", Rgba(0.2, 0.8, 0.2, 1.0))
    scene_graph.AssignRole(plant.get_source_id(), bin_id, bin_props)
    plant.RegisterCollisionGeometry(bin_body, RigidTransform(), bin_box, "bin_col", CoulombFriction(rand.friction, rand.friction * 0.8))

    try:
        # Configure VTK for software rendering to avoid slow GPU device enumeration
        vtk_params = RenderEngineVtkParams()
        # Force software rendering - prevents VTK from trying to enumerate GPU devices
        import os
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
        os.environ.setdefault("VTK_USE_OFFSCREEN_EGL", "0")
        scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(vtk_params))
    except RuntimeError:
        pass

    plant.Finalize()

    def _camera_pose(position: np.ndarray, target: np.ndarray) -> RigidTransform:
        forward = target - position
        norm_forward = np.linalg.norm(forward)
        if norm_forward < 1e-6:
            forward = np.array([1.0, 0.0, 0.0])
            norm_forward = 1.0
        forward /= norm_forward
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(up, forward)
        norm_side = np.linalg.norm(side)
        if norm_side < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            side = np.cross(up, forward)
            norm_side = np.linalg.norm(side)
        side /= norm_side
        true_up = np.cross(forward, side)
        R = RotationMatrix(np.column_stack((side, true_up, forward)))
        return RigidTransform(R, position)

    target = np.array([0.6, 0.05, 0.82])
    front_position = np.array([-0.35, 0.0, 1.05 + rand.camera_jitter])
    X_WC_front = _camera_pose(front_position, target)
    sensor_front = _make_rgbd_sensor("front", builder, scene_graph, X_WC_front, image_size, camera_fps)

    wrist_position = np.array([0.1, -0.35, 0.9 + rand.camera_jitter])
    X_WC_wrist = _camera_pose(wrist_position, target)
    sensor_wrist = _make_rgbd_sensor("wrist", builder, scene_graph, X_WC_wrist, image_size, camera_fps)

    if meshcat is not None:
        AddDefaultVisualization(builder, meshcat=meshcat)
    else:
        AddDefaultVisualization(builder)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Initialize robot configuration
    plant_context = plant.GetMyMutableContextFromRoot(context)
    q = plant.GetPositions(plant_context).copy()
    
    # Set arm to a reasonable home configuration (semi-extended, elbow up)
    # These values work well for manipulation tasks on a table
    if plant.num_positions() >= 7:
        q[0] = 0.0      # joint 1 (base rotation)
        q[1] = -0.785   # joint 2 (shoulder, -45deg)
        q[2] = 0.0      # joint 3 (shoulder rotation)
        q[3] = -2.356   # joint 4 (elbow, -135deg)
        q[4] = 0.0      # joint 5 (forearm rotation)
        q[5] = 1.571    # joint 6 (wrist, 90deg)
        q[6] = 0.785    # joint 7 (flange rotation, 45deg)
    
    # Initialize gripper to partially open position (0.04m = 4cm apart)
    if plant.num_positions() >= 9:  # 7 arm + 2 gripper
        q[7] = 0.02  # left finger (half of desired opening)
        q[8] = 0.02  # right finger (half of desired opening)
    
    plant.SetPositions(plant_context, q)

    ports = {
        "rgb_front": sensor_front.color_image_output_port(),
        "rgb_wrist": sensor_wrist.color_image_output_port(),
    }
    model_indices = {"franka": franka_model}

    return SceneHandles(simulator, plant, scene_graph, diagram, context, ports, model_indices, meshcat)

def build_scene_c(
    task_name: str,
    rand: DomainRandomization,
    image_size=(640, 480),
    camera_fps: float = 4.0,
    meshcat: Optional[Any] = None,
) -> SceneHandles:
    """Build physics-based pick-and-place scene for scene_c.
    
    This creates a scene with a FREE-BODY block that can be grasped
    and manipulated using physics-based grasping with weld constraints.
    """
    # Clear Meshcat to remove any previous scene visualization
    if meshcat is not None:
        meshcat.Delete()
        meshcat.DeleteAddedControls()
    
    # Fix LCM networking for Meshcat
    import os
    os.environ["LCM_DEFAULT_URL"] = "udpm://239.255.76.67:7667?ttl=0"
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.002)
    parser = Parser(plant)

    franka_model = _add_franka(plant, parser)
    
    # Add visible ground plane
    from pydrake.geometry import Box
    ground_box = Box(3.0, 3.0, 0.01)  # Large thin box for visible ground
    ground_color = np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float64).reshape(4, 1)
    ground_instance = GeometryInstance(
        RigidTransform([0, 0, -0.005]),  # Slightly below z=0
        ground_box.Clone(),
        "ground_vis"
    )
    ground_instance.set_illustration_properties(MakePhongIllustrationProperties(ground_color))
    ground_id = plant.RegisterVisualGeometry(plant.world_body(), ground_instance)
    ground_props = PerceptionProperties()
    ground_props.AddProperty("phong", "diffuse", Rgba(0.7, 0.7, 0.7, 1.0))
    scene_graph.AssignRole(plant.get_source_id(), ground_id, ground_props)
    
    # Add collision geometry
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), HalfSpace(), "ground", CoulombFriction(0.9, 0.8))
    _add_table(plant, parser)

    block = Box(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    bin_box = Box(BIN_SIZE[0], BIN_SIZE[1], BIN_SIZE[2])

    # Add block as FREE BODY with physics using parser (automatically creates floating joint)
    # Compute correct inertia for cube: I = (1/6) * mass * side^2
    block_mass = 0.05  # kg
    block_inertia = (1.0/6.0) * block_mass * BLOCK_SIZE**2
    
    # Create a simple SDF for a floating box
    import tempfile
    block_sdf = f'''<?xml version="1.0"?>
<sdf version="1.7">
  <model name="block">
    <link name="block_body">
      <inertial>
        <mass>{block_mass}</mass>
        <inertia>
          <ixx>{block_inertia:.10f}</ixx>
          <iyy>{block_inertia:.10f}</iyy>
          <izz>{block_inertia:.10f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>{BLOCK_SIZE} {BLOCK_SIZE} {BLOCK_SIZE}</size></box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>{rand.friction}</mu>
              <mu2>{rand.friction * 0.8}</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>{BLOCK_SIZE} {BLOCK_SIZE} {BLOCK_SIZE}</size></box>
        </geometry>
        <material>
          <diffuse>{rand.block_color[0]} {rand.block_color[1]} {rand.block_color[2]} {rand.block_color[3]}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
        f.write(block_sdf)
        block_sdf_path = f.name
    
    try:
        block_instance = parser.AddModels(block_sdf_path)[0]
        block_body = plant.GetBodyByName("block_body", block_instance)
    finally:
        import os
        os.unlink(block_sdf_path)
    
    # Block will be positioned via SetFreeBodyPose after finalization
    # Visual properties already set in SDF

    # Add bin (welded on table surface at randomized position)
    # Bin dimensions: 20cm x 15cm x 5cm, half-height = 0.025m
    bin_model = plant.AddModelInstance("bin")
    bin_body = plant.AddRigidBody(
        "bin",
        bin_model,
        SpatialInertia.MakeFromCentralInertia(
            1.0,
            np.zeros(3),
            UnitInertia.SolidBox(BIN_SIZE[0], BIN_SIZE[1], BIN_SIZE[2]),
        ),
    )
    # Use randomized bin position from domain randomization
    bin_position = np.array([rand.bin_pos_x, rand.bin_pos_y, TABLE_SURFACE_Z])
    X_Wbin = RigidTransform(bin_position)
    plant.WeldFrames(plant.world_frame(), bin_body.body_frame(), X_Wbin)
    bin_rgba = Rgba(0.2, 0.8, 0.2, 0.3)  # Semi-transparent (alpha=0.3)
    bin_color = np.array([0.2, 0.8, 0.2, 0.3], dtype=np.float64).reshape(4, 1)
    bin_instance = GeometryInstance(RigidTransform(), bin_box.Clone(), "bin_vis")
    bin_instance.set_illustration_properties(MakePhongIllustrationProperties(bin_color))
    bin_id = plant.RegisterVisualGeometry(bin_body, bin_instance)
    bin_props = PerceptionProperties()
    bin_props.AddProperty("phong", "diffuse", Rgba(0.2, 0.8, 0.2, 0.3))
    scene_graph.AssignRole(plant.get_source_id(), bin_id, bin_props)
    plant.RegisterCollisionGeometry(bin_body, RigidTransform(), bin_box, "bin_col", CoulombFriction(rand.friction, rand.friction * 0.8))

    # Configure VTK renderer with GPU support for better quality
    vtk_params = RenderEngineVtkParams()
    scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(vtk_params))

    # Configure contact solver for rigid body simulation
    # Use SAP (Semi-Analytical Primal) - robust discrete contact solver
    from pydrake.multibody.plant import ContactModel, DiscreteContactApproximation
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    
    # Use point contact model for rigid body dynamics  
    plant.set_contact_model(ContactModel.kPoint)
    
    # Use default SAP parameters - Drake's defaults are tuned for stability
    # Small penetrations (~1-3mm) are normal in discrete contact simulation
    # and represent the regularization needed for numerical stability
    
    plant.Finalize()
    
    # Set default positions AFTER finalizing
    # Total DOF: 9 (robot: 7 arm + 2 fingers) + 7 (floating block) = 16
    # Modified Franka home: gripper pointing DOWN (not standard home)
    # Joint 7 set to 0.0 instead of π/4 so gripper points down
    # Gripper FULLY OPEN (0.107m per finger = max opening) before grasping
    q_home_robot = np.array([
        0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.0,
        0.107, 0.107  # Fully open gripper (max opening ~10cm)
    ])
    # Block: QuaternionFloatingJoint uses [qw, qx, qy, qz, x, y, z] order
    # Initialize at RESTING position on table (not floating)
    # Table top at TABLE_SURFACE_Z, block center at TABLE_SURFACE_Z + BLOCK_HALF_HEIGHT
    block_z = TABLE_SURFACE_Z + BLOCK_HALF_HEIGHT
    block_quat = RotationMatrix.MakeZRotation(rand.block_yaw).ToQuaternion().wxyz()
    q_home_block = np.concatenate([
        block_quat,  # quaternion first: [qw, qx, qy, qz]
        [rand.block_pos_x, rand.block_pos_y, block_z]  # position: [x, y, z]
    ])
    q_home_all = np.concatenate([q_home_robot, q_home_block])
    plant.SetDefaultPositions(q_home_all)

    def _camera_pose(position: np.ndarray, target: np.ndarray) -> RigidTransform:
        forward = target - position
        norm_forward = np.linalg.norm(forward)
        if norm_forward < 1e-6:
            forward = np.array([1.0, 0.0, 0.0])
            norm_forward = 1.0
        forward /= norm_forward
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(up, forward)
        norm_side = np.linalg.norm(side)
        if norm_side < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            side = np.cross(up, forward)
            norm_side = np.linalg.norm(side)
        side /= norm_side
        true_up = np.cross(forward, side)
        # Drake camera convention: +X right, +Y DOWN, +Z forward
        # To get upright images, we need to flip the Y axis
        R = RotationMatrix(np.column_stack((side, -true_up, forward)))
        return RigidTransform(R, position)

    # IMPROVED CAMERA SYSTEM - Always above table, proper orientation
    # Target point: center of manipulation workspace
    workspace_center = np.array([0.45, 0.0, TABLE_SURFACE_Z + 0.1])  # 10cm above table
    
    # Front camera: randomized position ABOVE table, looking down at workspace
    # Spherical coordinates with constraints to keep camera above table
    elevation_min = 0.3 * np.pi  # 54° minimum (looking down)
    elevation_max = 0.45 * np.pi  # 81° maximum (not too steep)
    
    # Ensure camera elevation keeps it above table
    safe_elevation = max(rand.camera_elevation, elevation_min)
    safe_elevation = min(safe_elevation, elevation_max)
    
    front_position = np.array([
        workspace_center[0] + rand.camera_radius * np.cos(rand.camera_azimuth) * np.sin(safe_elevation),
        workspace_center[1] + rand.camera_radius * np.sin(rand.camera_azimuth) * np.sin(safe_elevation),
        workspace_center[2] + rand.camera_radius * np.cos(safe_elevation) + abs(rand.camera_jitter)  # abs() ensures above table
    ])
    
    # Ensure minimum height above table
    min_camera_height = TABLE_SURFACE_Z + 0.3  # 30cm above table minimum
    front_position[2] = max(front_position[2], min_camera_height)
    
    X_WC_front = _camera_pose(front_position, workspace_center)
    sensor_front = _make_rgbd_sensor("front", builder, scene_graph, X_WC_front, image_size, camera_fps)

    # Wrist camera: side view from robot's perspective, also above table
    wrist_position = np.array([
        0.2,  # Behind robot base
        -0.5 + rand.camera_jitter * 0.1,  # Side position with small jitter
        TABLE_SURFACE_Z + 0.4 + abs(rand.camera_jitter)  # 40cm above table minimum
    ])
    wrist_target = np.array([0.45, 0.0, TABLE_SURFACE_Z + 0.05])  # Look at table surface
    X_WC_wrist = _camera_pose(wrist_position, wrist_target)
    sensor_wrist = _make_rgbd_sensor("wrist", builder, scene_graph, X_WC_wrist, image_size, camera_fps)

    # Initialize Meshcat for interactive visualization (only if explicitly provided)
    # By default, meshcat=None to avoid creating multiple instances during batch generation
    if meshcat is not None:
        print(f"\n🌐 Meshcat visualization available at: {meshcat.web_url()}\n")
    
    # Use MeshcatVisualizer directly (no LCM required)
    if meshcat is not None:
        from pydrake.all import MeshcatVisualizer, MeshcatVisualizerParams
        # Reduce publish frequency to save memory during long simulations
        params = MeshcatVisualizerParams()
        params.publish_period = 0.1  # Update at 10Hz instead of every timestep
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Initialize context with default positions and velocities
    plant_context = plant.GetMyMutableContextFromRoot(context)
    q_default = plant.GetDefaultPositions()
    plant.SetPositions(plant_context, q_default)
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    # Gravity is enabled by default for all bodies
    
    # Force Meshcat to publish the initial state (only if visualizing)
    if meshcat is not None:
        diagram.ForcedPublish(context)  # Publish without advancing time

    ports = {
        "rgb_front": sensor_front.color_image_output_port(),
        "rgb_wrist": sensor_wrist.color_image_output_port(),
    }
    model_indices = {"franka": franka_model}

    return SceneHandles(simulator, plant, scene_graph, diagram, context, ports, model_indices, meshcat)

def sample_randomization(strength: float, rng: np.random.Generator) -> DomainRandomization:
    """Sample domain randomization parameters.
    
    Args:
        strength: Randomization strength (0.0 = minimal, 1.0 = maximal)
        rng: Random number generator
    
    Returns:
        DomainRandomization with sampled parameters
    """
    color = (float(rng.random()), float(rng.random()), float(rng.random()), 1.0)
    
    # FULL randomization for data generation
    # Robot base at [0, 0, 0.77], arm reach ~0.8m
    # Table surface at z=0.77m, safe workspace: x=[0.3, 0.65], y=[-0.25, 0.25]
    
    # Block position: Wide randomization across reachable workspace
    block_x = float(rng.uniform(0.30, 0.55))  # 30-55cm from robot base (25cm range)
    block_y = float(rng.uniform(-0.20, 0.20))  # ±20cm lateral (40cm range)
    block_yaw = float(rng.uniform(-np.pi, np.pi))  # Full 360° rotation
    
    # Bin/goal position: Randomize placement location
    # Keep bin reachable but separated from block spawn area
    bin_x = float(rng.uniform(0.50, 0.70))  # 50-70cm from robot (behind block spawn)
    bin_y = float(rng.uniform(-0.15, 0.25))  # Lateral variation
    
    # Camera viewpoint: FIXED position following Meta VJEPA2 paper recommendations
    # Meta found that training with varying camera positions WITHOUT camera conditioning
    # degrades performance. The model's inferred coordinate axis is sensitive to camera position.
    # 
    # Must match Meta's pretrained VJEPA2 model expectations!
    # Meta trained on DROID dataset using LEFT exocentric camera view (~60° left of robot)
    # We use +1.93 rad (+110.6°) to match the left exocentric view from Meta's training
    #   - Azimuth: +1.93 rad (+110.6°) - LEFT of robot base (matches Meta's DROID)
    #   - Elevation: 1.07 rad (61.4°) - looking down at workspace
    #   - Radius: 1.46m
    #
    # Small jitter (±5°) is acceptable for robustness without breaking coordinate inference.
    camera_radius = float(1.46 + 0.05 * strength * rng.standard_normal())  # ~1.46m ± small jitter
    camera_azimuth = float(1.93 + 0.05 * strength * rng.standard_normal())  # +110.6° ± 3° jitter (LEFT exocentric)
    camera_elevation = float(1.07 + 0.03 * strength * rng.standard_normal())  # 61.4° ± 2° jitter
    
    return DomainRandomization(
        light_intensity=float(0.5 + 1.5*strength*rng.random()),
        camera_jitter=float(0.01*strength*rng.standard_normal()),
        clutter_count=int(rng.integers(0, 3 + int(5*strength))),
        friction=float(np.clip(2.0 + 2.0*rng.random(), 2.0, 4.0)),  # High friction: 2.0-4.0
        block_color=color,
        block_pos_x=block_x,
        block_pos_y=block_y,
        block_yaw=block_yaw,
        bin_pos_x=bin_x,
        bin_pos_y=bin_y,
        camera_radius=camera_radius,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
    )
