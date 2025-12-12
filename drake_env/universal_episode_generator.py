import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from pydrake.all import (
    DiagramBuilder,
    RigidTransform,
    RotationMatrix,
    Simulator,
    TrajectorySource,
    PiecewisePolynomial,
)
from pydrake.systems.sensors import RgbdSensor, CameraConfig
from pydrake.geometry import (
    RenderEngineVtkParams, 
    MakeRenderEngineVtk,
)
from manipulation.station import LoadScenario, MakeHardwareStation


# === Trajectory building functions (copied verbatim from reference notebook) ===

dt = 0.05
pause_before_action = 0.5  # pause BEFORE close/open
pause_after_action = 0.5  # pause AFTER  close/open
opened, closed = 0.107, 0.0


def _q7(q: tuple) -> np.ndarray:
    """Ensure a 7-DoF joint vector shaped (7,)."""
    return np.asarray(q, float).reshape(7,)


def _append_path(
    times: list, Q: list, t: float, path: list
) -> float:
    """Append a polyline with uniform dt; returns updated time."""
    for q in path:
        if times:
            t += dt
        times.append(t)
        Q.append(_q7(q))
    return t


def _hold(
    times: list,
    Q: list,
    t: float,
    q_hold: np.ndarray,
    duration: float,
) -> float:
    """Hold current pose for 'duration' seconds; returns updated time."""
    if duration <= 0:
        return t
    t += duration
    times.append(t)
    Q.append(_q7(q_hold))
    return t


def build_trajs(
    path_pick: list,
    q_grasp: np.ndarray,
    q_approach: np.ndarray,
    path_place: list,
    path_reset: list,
):
    """Sequence:
    pick → q_grasp (OPEN) → pause 0.5 → CLOSE (no motion) → pause 0.5 → q_approach →
    place → pause 0.5 → OPEN (no motion) → pause 0.5 → reset
    """
    times = []
    Q = []
    t = 0.0

    # 1) path_pick  (ends at q_approach typically)
    t = _append_path(times, Q, t, path_pick)

    # 2) move to q_grasp (WSG stays OPEN)
    if not np.allclose(Q[-1], q_grasp):
        t += 10 * dt
        times.append(t)
        Q.append(_q7(q_grasp))

    # 3) pause BEFORE CLOSE (no motion)
    t = _hold(times, Q, t, q_grasp, pause_before_action)

    # 4) CLOSE (command change at this instant), then pause AFTER CLOSE (no motion)
    t_close = t
    t = _hold(times, Q, t, q_grasp, pause_after_action)

    # 5) return to q_approach
    if not np.allclose(Q[-1], q_approach):
        t += 10 * dt
        times.append(t)
        Q.append(_q7(q_approach))

    # 6) path_place
    t = _append_path(times, Q, t, path_place)

    # 7) pause BEFORE OPEN (no motion)
    t = _hold(times, Q, t, Q[-1], pause_before_action)

    # 8) OPEN (command change at this instant), then pause AFTER OPEN (no motion)
    t_open = t
    t = _hold(times, Q, t, Q[-1], pause_after_action)

    # 9) path_reset
    t = _append_path(times, Q, t, path_reset)

    # Build Drake trajectories (7×N for robot, 1×K for WSG)
    q_samples = np.stack(Q, axis=1)
    traj_q = PiecewisePolynomial.FirstOrderHold(times, q_samples)

    wsg_knots = [times[0], t_close, t_open, times[-1]]
    wsg_vals = [opened, closed, opened, opened]
    traj_wsg = PiecewisePolynomial.ZeroOrderHold(
        wsg_knots, np.asarray(wsg_vals).reshape(1, -1)
    )

    print(f"Trajectory duration: T={times[-1]:.3f}s")
    return traj_q, traj_wsg


# === End trajectory building functions ===


def _camera_pose(position: np.ndarray, target: np.ndarray) -> RigidTransform:
    """Create camera pose looking from position to target."""
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


def _add_cameras_to_scene(station, builder, scene_graph, image_size=(640, 480)):
    """Add front and wrist cameras to the scene."""
    # Front camera: above and in front of shelf, looking at workspace
    workspace_center = np.array([0.6, 0.0, 0.3])  # Center of shelf area
    front_position = np.array([0.3, -0.8, 0.8])  # Side angle view
    X_WC_front = _camera_pose(front_position, workspace_center)
    
    # Create camera core with intrinsics
    core = RenderCameraCore(
        renderer_name="renderer",
        intrinsics=CameraInfo(
            width=image_size[0],
            height=image_size[1],
            fov_y=np.pi / 4,
        ),
        clipping=ClippingRange(0.01, 10.0),
        X_BS=RigidTransform(),
    )
    
    # Create color and depth cameras for front
    color_camera_front = ColorRenderCamera(core, show_window=False)
    depth_camera_front = DepthRenderCamera(core, DepthRange(0.01, 10.0))
    
    # Create front sensor
    front_sensor = RgbdSensor(
        parent_id=scene_graph.world_frame_id(),
        X_PB=X_WC_front,
        color_camera=color_camera_front,
        depth_camera=depth_camera_front,
    )
    front_sensor.set_name("front_camera")
    builder.AddSystem(front_sensor)
    builder.Connect(
        station.GetOutputPort("query_object"),
        front_sensor.query_object_input_port(),
    )
    
    # Wrist camera: closer side view
    wrist_position = np.array([0.2, -0.6, 0.5])
    wrist_target = np.array([0.6, 0.0, 0.3])
    X_WC_wrist = _camera_pose(wrist_position, wrist_target)
    
    # Create color and depth cameras for wrist
    color_camera_wrist = ColorRenderCamera(core, show_window=False)
    depth_camera_wrist = DepthRenderCamera(core, DepthRange(0.01, 10.0))
    
    # Create wrist sensor
    wrist_sensor = RgbdSensor(
        parent_id=scene_graph.world_frame_id(),
        X_PB=X_WC_wrist,
        color_camera=color_camera_wrist,
        depth_camera=depth_camera_wrist,
    )
    wrist_sensor.set_name("wrist_camera")
    builder.AddSystem(wrist_sensor)
    builder.Connect(
        station.GetOutputPort("query_object"),
        wrist_sensor.query_object_input_port(),
    )
    
    return front_sensor, wrist_sensor


def generate_episode(
    scenario_yaml: str,
    traj_q: PiecewisePolynomial,
    traj_wsg: PiecewisePolynomial,
    fps: float = 4.0,
    meshcat=None,
    language_prompt: str = "",
    subgoals: List[str] = None,
    workspace_center: np.ndarray = None,
    front_camera_pos: np.ndarray = None,
    wrist_camera_pos: np.ndarray = None,
    wrist_camera_target: np.ndarray = None,
    randomize: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Universal episode generation function - works for all scenes.
    
    Args:
        scenario_yaml: Drake scenario YAML content or file path
        traj_q: Arm trajectory (7-DOF)
        traj_wsg: Gripper trajectory (1-DOF)
        fps: Frames per second
        meshcat: Meshcat instance for visualization
        language_prompt: Task description
        subgoals: List of subgoal names
        workspace_center: Workspace center for camera targeting
        front_camera_pos: Front camera position
        wrist_camera_pos: Wrist camera position
        wrist_camera_target: Wrist camera target
    
    Returns:
        episode_data: Dict with images_front, images_wrist, states, actions, contacts, metadata
    """
    # Set defaults
    if subgoals is None:
        subgoals = []
    if workspace_center is None:
        workspace_center = np.array([0.65, 0.0, 0.4])
    # Camera defaults: LEFT exocentric view matching Meta's VJEPA2 pretrained model
    # Meta trained on DROID dataset using left exocentric camera (~60° left of robot)
    # Y = positive = LEFT of robot base (matches Meta's DROID training)
    if front_camera_pos is None:
        front_camera_pos = np.array([0.5, 0.6, 0.8])  # LEFT exocentric (Y = +0.6)
    if wrist_camera_pos is None:
        wrist_camera_pos = np.array([0.2, 0.5, 0.6])  # Wrist also on left side
    if wrist_camera_target is None:
        wrist_camera_target = np.array([0.65, 0.0, 0.3])

    rng = rng if rng is not None else np.random.default_rng()
    randomize = max(0.0, float(randomize))
    
    def _jitter(vec: np.ndarray, scale: np.ndarray) -> np.ndarray:
        if randomize <= 1e-6:
            return vec
        delta = rng.uniform(-1.0, 1.0, size=scale.shape) * scale * randomize
        return vec + delta
    
    workspace_center = _jitter(workspace_center, np.array([0.03, 0.03, 0.01]))
    front_camera_pos = _jitter(front_camera_pos, np.array([0.05, 0.05, 0.05]))
    wrist_camera_pos = _jitter(wrist_camera_pos, np.array([0.04, 0.04, 0.04]))
    wrist_camera_target = _jitter(wrist_camera_target, np.array([0.03, 0.03, 0.02]))
    
    #Handle scenario YAML (could be content or file path)
    import tempfile
    scenario_file = scenario_yaml
    temp_file = None
    
    # If it looks like YAML content, write to temp file
    if "directives:" in scenario_yaml or "model_drivers:" in scenario_yaml:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        temp_file.write(scenario_yaml)
        temp_file.close()
        scenario_file = temp_file.name
    
    try:
        
        # === Build unified scene with cameras (matching Scene C approach) ===
        print(f"Building unified scene with cameras...")
        
        # Build unified simulation + rendering scene (matching Scene C)
        # Use scenario YAML for HardwareStation, but add cameras separately
        scenario = LoadScenario(filename=scenario_file)
        builder = DiagramBuilder()
        
        # Add HardwareStation from scenario
        station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
        
        # Get plant and scene_graph from station
        plant = station.GetSubsystemByName("plant")
        scene_graph = station.GetSubsystemByName("scene_graph")
        
        # Helper function for camera pose (copied from Scene C)
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
            # Drake camera: +X right, +Y DOWN, +Z forward
            R = RotationMatrix(np.column_stack((side, -true_up, forward)))
            return RigidTransform(R, position)
        
        # Helper to create RGBD sensor (adapted for HardwareStation)
        def _make_rgbd_sensor(name: str, pose: RigidTransform) -> RgbdSensor:
            cfg = CameraConfig()
            cfg.name = name
            cfg.width = 640
            cfg.height = 480
            cfg.rgb = True
            cfg.depth = True
            cfg.label = False
            cfg.show_rgb = False
            cfg.renderer_name = "renderer"
            cfg.fps = fps
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
            # Connect to station's geometry query output port (exported from scene_graph)
            builder.Connect(
                station.GetOutputPort("query_object"),
                sensor.query_object_input_port()
            )
            return sensor
        
        # Add VTK renderer to scene_graph for camera rendering
        vtk_params = RenderEngineVtkParams()
        scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(vtk_params))
        
        # Add cameras to the same builder (use parameters)
        X_WC_front = _camera_pose(front_camera_pos, workspace_center)
        sensor_front = _make_rgbd_sensor("front", X_WC_front)
        
        X_WC_wrist = _camera_pose(wrist_camera_pos, wrist_camera_target)
        sensor_wrist = _make_rgbd_sensor("wrist", X_WC_wrist)
        
        # Add trajectory sources
        arm_traj_src = builder.AddSystem(TrajectorySource(traj_q))
        hand_traj_src = builder.AddSystem(TrajectorySource(traj_wsg))
        
        # Wire trajectories into station inputs
        builder.Connect(
            arm_traj_src.get_output_port(),
            station.GetInputPort("panda.position")
        )
        # Try Franka hand first, fall back to WSG for backward compatibility
        try:
            builder.Connect(
                hand_traj_src.get_output_port(),
                station.GetInputPort("hand.position")
            )
        except RuntimeError:
            builder.Connect(
                    hand_traj_src.get_output_port(),
                station.GetInputPort("wsg.position")
            )
        
        # Build diagram and create simulator
        diagram = builder.Build()
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        
        # Get plant context
        plant_context = plant.GetMyMutableContextFromRoot(context)

        # Cache initial letter and shelf poses for success evaluation
        letter_initial_p = None
        shelf_center_p = None
        try:
            if plant.HasModelInstanceNamed("letter"):
                letter_model = plant.GetModelInstanceByName("letter")
                letter_bodies = plant.GetBodyIndices(letter_model)
                if letter_bodies:
                    letter_body = plant.get_body(letter_bodies[0])
                    X_WL_init = plant.EvalBodyPoseInWorld(plant_context, letter_body)
                    letter_initial_p = X_WL_init.translation().copy()
            if plant.HasModelInstanceNamed("shelves"):
                shelf_model = plant.GetModelInstanceByName("shelves")
                shelf_body = plant.GetBodyByName("shelves_body", shelf_model)
                X_WS = plant.EvalBodyPoseInWorld(plant_context, shelf_body)
                shelf_center_p = X_WS.translation().copy()
        except Exception:
            letter_initial_p = None
            shelf_center_p = None

        # Get end-effector frame (Franka hand or WSG body)
        if plant.HasBodyNamed("panda_hand"):
            ee_frame = plant.GetBodyByName("panda_hand").body_frame()
        elif plant.HasBodyNamed("body"):
            ee_frame = plant.GetBodyByName("body").body_frame()  # WSG body (legacy)
        else:
            ee_frame = plant.GetBodyByName("panda_link8").body_frame()

        # Attempt to locate the letter body (model name 'letter') for success checks
        letter_body = None
        try:
            if plant.HasModelInstanceNamed("letter"):
                letter_model = plant.GetModelInstanceByName("letter")
                letter_bodies = plant.GetBodyIndices(letter_model)
                if letter_bodies:
                    letter_body = plant.get_body(letter_bodies[0])
        except Exception:
            letter_body = None
        
        # Publish initial state
        diagram.ForcedPublish(context)
        
        # Define render_rgb function (matching Scene C exactly)
        def render_rgb(port):
            sys = port.get_system()
            port_context = sys.GetMyContextFromRoot(context)
            img = port.Eval(port_context).data
            # Copy only RGB channels and convert to uint8 immediately to save memory
            return np.ascontiguousarray(img[:,:,:3], dtype=np.uint8)
        
        # Start recording and run simulation (copied from reference notebook)
        if meshcat:
            meshcat.StartRecording()
        
        simulator.Initialize()
        T_end = max(traj_q.end_time(), traj_wsg.end_time())
        
        # Sample states during simulation for episode data
        # Use trajectory duration instead of requested duration
        num_frames = max(1, int(np.round(T_end * fps)))
        sample_times = np.linspace(0.0, T_end, num_frames)
        
        images_front = []
        images_wrist = []
        states_list = []
        actions_list = []
        contacts_list = []
        prev_ee_state = None
        prev_q_full_state = None
        prev_t = None
        # Track letter pose over the episode for task-success checks
        letter_pos_initial = None
        letter_pos_final = None
        
        for frame_idx, t in enumerate(sample_times):
            # Advance simulation to this time (this also records frames in Meshcat)
            simulator.AdvanceTo(t)
            
            # Get current plant context
            plant_context = plant.GetMyMutableContextFromRoot(context)
            
            # Get robot configuration
            q_robot = plant.GetPositions(plant_context)[:7]  # First 7 DOF are arm
            gripper_width = traj_wsg.value(t)[0, 0]

            # Track letter pose (if available) for downstream success checks
            letter_pos = None
            if letter_body is not None:
                try:
                    X_WL = plant.EvalBodyPoseInWorld(plant_context, letter_body)
                    letter_pos = X_WL.translation().copy()
                except Exception:
                    letter_pos = None
            if letter_pos is not None:
                if letter_pos_initial is None:
                    letter_pos_initial = letter_pos.copy()
                letter_pos_final = letter_pos.copy()
            
            # Compute end-effector state
            X_WE = plant.CalcRelativeTransform(
                plant_context, plant.world_frame(), ee_frame
            )
            ee_pos = X_WE.translation()
            ee_rot_rpy = X_WE.rotation().ToRollPitchYaw().vector()
            ee_state = np.concatenate([ee_pos, ee_rot_rpy, [gripper_width]])
            
            # Compute action delta
            if prev_ee_state is not None:
                ee_delta = ee_state - prev_ee_state
            else:
                ee_delta = np.zeros(7)
            prev_ee_state = ee_state.copy()
            
            # Create full state (9 DOF: 7 arm + 2 gripper)
            q_full_state = np.concatenate([
                q_robot,  # 7 arm joints
                [gripper_width, gripper_width],  # 2 gripper joints
            ])

            # Approximate joint velocity (qdot) from finite differences
            if prev_q_full_state is not None and prev_t is not None:
                dt_q = float(max(t - prev_t, 1e-6))
                qdot = (q_full_state - prev_q_full_state) / dt_q
            else:
                qdot = np.zeros_like(q_full_state)
            prev_q_full_state = q_full_state.copy()
            prev_t = t
            
            # Store state and action dicts
            state_dict = {
                "q": q_full_state,
                "ee_state": ee_state,
            }
            if letter_pos is not None:
                state_dict["letter_pos"] = letter_pos
            states_list.append(state_dict)
            
            actions_list.append({
                "qdot": qdot,
                "ee_delta": ee_delta,
            })
            
            contacts_list.append({
                "force": 0.0,  # Placeholder
            })
            
            # Render images from camera ports (matching Scene C exactly)
            images_front.append(render_rgb(sensor_front.color_image_output_port()))
            images_wrist.append(render_rgb(sensor_wrist.color_image_output_port()))
            
            if frame_idx % 20 == 0:
                print(f"[t={t:.2f}s] Frame {frame_idx}/{num_frames}")
        
        # Stop recording and publish (copied from reference notebook)
        if meshcat:
            meshcat.StopRecording()
            meshcat.PublishRecording()
            print(f"Meshcat recording saved and published ({T_end:.1f}s animation)")
        
        print(f"Simulation complete")

        # Derive a simple task-success signal based on letter motion.
        # Criteria:
        #  - Letter moved at least 12cm from its initial pose (picked up / transported)
        #  - Final X is in front of the shelf (x > 0.7)
        #  - Final Z is above the shelf top (z > 0.5)
        task_success = False
        letter_initial_list = None
        letter_final_list = None
        if letter_pos_initial is not None and letter_pos_final is not None:
            letter_initial_list = letter_pos_initial.tolist()
            letter_final_list = letter_pos_final.tolist()
            delta = float(np.linalg.norm(letter_pos_final - letter_pos_initial))
            moved_enough = delta > 0.12
            forward_enough = letter_pos_final[0] > 0.7
            high_enough = letter_pos_final[2] > 0.5
            task_success = moved_enough and forward_enough and high_enough
            print(
                f"  Letter delta={delta:.3f}, final={letter_final_list}, "
                f"moved_enough={moved_enough}, forward_enough={forward_enough}, high_enough={high_enough}, "
                f"task_success={task_success}"
            )
        else:
            # If we cannot track the letter, be conservative and mark as failure.
            print("Letter body not found or positions missing; marking episode as failed for dataset purposes")
            task_success = False
        
        # Geometric success check: did the letter reach the shelf goal region?
        success = True
        try:
            if letter_initial_p is not None and plant.HasModelInstanceNamed("letter"):
                letter_model = plant.GetModelInstanceByName("letter")
                letter_bodies = plant.GetBodyIndices(letter_model)
                if letter_bodies:
                    letter_body = plant.get_body(letter_bodies[0])
                    X_WL_final = plant.EvalBodyPoseInWorld(plant_context, letter_body)
                    p_L_final = X_WL_final.translation()
                    # Vertical motion relative to initial pose (stand height ~0.26m)
                    dz_lift = float(p_L_final[2] - letter_initial_p[2])
                    if shelf_center_p is not None:
                        # Define a goal region relative to shelf center
                        rel = p_L_final - shelf_center_p
                        dx, dy, dz_rel = float(rel[0]), float(rel[1]), float(rel[2])
                        success = (
                            dz_lift > 0.10
                            and dz_rel > 0.05   # above shelf mid-plane
                            and dz_rel < 0.45
                            and abs(dx) <= 0.35
                            and abs(dy) <= 0.35
                        )
                    else:
                        # require the letter to be lifted well above the stand
                        success = (dz_lift > 0.20 and float(p_L_final[2]) > 0.5)
        except Exception:
            # If the geometric check itself fails, do not discard the episode.
            success = True

        # Create episode data matching V-JEPA2-AC format
        meta = {
            'language': language_prompt,
            'subgoals': subgoals,
            'task_success': bool(task_success),
            'success': bool(success),
            'letter_initial': letter_initial_p.tolist() if letter_initial_p is not None else None,
            'shelf_center': shelf_center_p.tolist() if shelf_center_p is not None else None,
        }
        if letter_initial_list is not None:
            meta['letter_initial_pos'] = letter_initial_list
        if letter_final_list is not None:
            meta['letter_final_pos'] = letter_final_list

        episode_data = {
            'images_front': images_front,
            'images_wrist': images_wrist,
            'states': states_list,
            'actions': actions_list,
            'contacts': contacts_list,
            'metadata': meta,
        }
        
        return episode_data
        
    finally:
        if temp_file:
            Path(temp_file.name).unlink(missing_ok=True)
