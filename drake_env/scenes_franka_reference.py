from __future__ import annotations

import numpy as np
import tempfile
from pathlib import Path
from typing import Any, Optional

from pydrake.all import DiagramBuilder, RigidTransform, RotationMatrix, Simulator
from pydrake.systems.sensors import RgbdSensor, CameraConfig
from pydrake.geometry import RenderEngineVtkParams, MakeRenderEngineVtk
from manipulation.station import LoadScenario, MakeHardwareStation


def _camera_pose(position: np.ndarray, target: np.ndarray) -> RigidTransform:
    """Create camera pose looking from position to target.

    Uses the same convention as the universal episode generator: +X right,
    +Y down, +Z forward.
    """
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
    R = RotationMatrix(np.column_stack((side, -true_up, forward)))
    return RigidTransform(R, position)


def _make_rgbd_sensor(
    name: str,
    builder: DiagramBuilder,
    scene_graph,
    station,
    pose: RigidTransform,
    image_size: tuple[int, int],
    fps: float,
) -> RgbdSensor:
    """Create an RGB-D sensor wired to the station's query_object port."""
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
    builder.Connect(
        station.GetOutputPort("query_object"),
        sensor.query_object_input_port(),
    )
    return sensor


def build_scene_franka_reference(
    task_name: str,
    letter: str = "C",
    image_size: tuple[int, int] = (256, 256),
    camera_fps: float = 4.0,
    meshcat: Optional[Any] = None,
):
    """Build Franka shelf/letter scene compatible with planners & tests.

    This loads the shared franka_shelves_scenario.yaml, substitutes the
    requested letter, constructs a HardwareStation, and attaches front and
    wrist RGB-D cameras.
    """
    from drake_env.scenes import SceneHandles

    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    scenario_template = assets_dir / "franka_shelves_scenario.yaml"

    with open(scenario_template, "r") as f:
        yaml_content = f.read()

    letter_file = f"file://{assets_dir}/{letter}_model/{letter}.sdf"
    letter_body_name = f"{letter}_body_link"
    letter_pose = RigidTransform(
        RotationMatrix.MakeZRotation(np.pi),
        [0.62, -0.05, 0.26],
    )
    
    # Use local stand.sdf with rigid hydroelastic properties
    stand_file = f"file://{assets_dir}/stand.sdf"
    yaml_content = yaml_content.replace("package://manipulation/stand.sdf", stand_file)
    
    yaml_content = yaml_content.replace("LETTER_FILE_PLACEHOLDER", letter_file)
    yaml_content = yaml_content.replace("LETTER_BODY_NAME", letter_body_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = tmp.name

    try:
        scenario = LoadScenario(filename=tmp_path)
        builder = DiagramBuilder()
        station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))

        plant = station.GetSubsystemByName("plant")
        scene_graph = station.GetSubsystemByName("scene_graph")

        vtk_params = RenderEngineVtkParams()
        scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(vtk_params))

        # Camera positions: LEFT exocentric view matching Meta's VJEPA2 pretrained model
        # Meta trained on DROID dataset using left exocentric camera (~60° left of robot)
        # Y = +1.2 (positive) = LEFT of robot base (matches Meta's DROID training)
        workspace_center = np.array([0.70, 0.0, 0.42])
        front_camera_pos = np.array([0.25, 1.2, 1.12])  # LEFT exocentric (Y = +1.2)
        wrist_camera_pos = np.array([0.18, 0.75, 0.65])  # Wrist also on left side
        wrist_camera_target = np.array([0.70, 0.0, 0.35])

        X_WC_front = _camera_pose(front_camera_pos, workspace_center)
        sensor_front = _make_rgbd_sensor(
            "front",
            builder,
            scene_graph,
            station,
            X_WC_front,
            image_size,
            camera_fps,
        )

        X_WC_wrist = _camera_pose(wrist_camera_pos, wrist_camera_target)
        sensor_wrist = _make_rgbd_sensor(
            "wrist",
            builder,
            scene_graph,
            station,
            X_WC_wrist,
            image_size,
            camera_fps,
        )

        # Export station command ports so higher-level evaluation code can drive
        # the HardwareStation using the same panda.position / hand.position
        # interfaces as in the training-time universal episode generator.
        try:
            panda_cmd_port = station.GetInputPort("panda.position")
            builder.ExportInput(panda_cmd_port, "panda.position")
        except Exception:
            pass
        # Try Franka hand first, then fall back to WSG for backward compatibility
        try:
            hand_cmd_port = station.GetInputPort("hand.position")
            builder.ExportInput(hand_cmd_port, "hand.position")
        except Exception:
            try:
                wsg_cmd_port = station.GetInputPort("wsg.position")
                builder.ExportInput(wsg_cmd_port, "wsg.position")
            except Exception:
                pass

        diagram = builder.Build()
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()

        plant_context = plant.GetMyMutableContextFromRoot(context)
        try:
            q_default = plant.GetDefaultPositions()
            plant.SetPositions(plant_context, q_default)
        except Exception:
            pass
        try:
            plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
        except Exception:
            pass

        # Reset letter pose/velocity to match episode generation default
        # IMPORTANT: SetFreeBodySpatialVelocity signature is (body, V_PB, context)
        # Missing/incorrect context was causing phantom letter motion!
        try:
            letter_body = plant.GetBodyByName(letter_body_name)
            plant.SetFreeBodyPose(plant_context, letter_body, letter_pose)
            # Create zero spatial velocity (6D: angular(3) + linear(3))
            from pydrake.multibody.math import SpatialVelocity
            zero_velocity = SpatialVelocity(np.zeros(3), np.zeros(3))
            # Correct signature: SetFreeBodySpatialVelocity(body, velocity, context)
            plant.SetFreeBodySpatialVelocity(letter_body, zero_velocity, plant_context)
        except Exception:
            pass

        ports = {
            "rgb_front": sensor_front.color_image_output_port(),
            "rgb_wrist": sensor_wrist.color_image_output_port(),
        }

        try:
            franka_model = plant.GetModelInstanceByName("panda")
            model_indices = {"franka": franka_model}
        except Exception:
            model_indices = {}

        if meshcat is not None:
            diagram.ForcedPublish(context)

        return SceneHandles(
            simulator=simulator,
            plant=plant,
            scene_graph=scene_graph,
            diagram=diagram,
            context=context,
            ports=ports,
            model_indices=model_indices,
            meshcat=meshcat,
        )

    finally:
        Path(tmp_path).unlink(missing_ok=True)
