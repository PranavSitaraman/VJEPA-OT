import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

from pydrake.all import (
    DiagramBuilder,
    InverseKinematics,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Solve,
)
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
from pydrake.math import RollPitchYaw
from manipulation.station import LoadScenario, MakeHardwareStation


class ManipulationStationSim:
    """
    Wrapper for manipulation station (from reference notebook).
    Handles scenario loading, collision checking, and configuration.
    """
    
    def __init__(
        self,
        scenario_file: Optional[str] = None,
        q_robot: Optional[tuple] = None,
        gripper_setpoint: float = 0.1,
        meshcat=None,
    ) -> None:
        self.meshcat = meshcat
        self.scenario = None
        self.station = None
        self.plant = None
        self.scene_graph = None
        self.query_output_port = None
        self.diagram = None

        # contexts
        self.context_diagram = None
        self.context_station = None
        self.context_scene_graph = None
        self.context_plant = None

        # mark initial configuration
        self.q0 = None
        self.okay_collisions = None
        self.gripper_setpoint = gripper_setpoint
        
        # Robot name (Franka vs IIWA)
        self.robot_name = "panda"
        self.gripper_name = "wsg"  # Schunk WSG gripper (Drake-supported driver)

        if scenario_file is not None:
            self.choose_sim(scenario_file, q_robot, gripper_setpoint)

    def choose_sim(
        self,
        scenario_file: str,
        q_robot: Optional[tuple] = None,
        gripper_setpoint: float = 0.1,
    ) -> None:
        """Load a scenario and initialize the simulation."""
        
        self.clear_meshcat()

        self.scenario = LoadScenario(filename=scenario_file)
        builder = DiagramBuilder()
        self.station = builder.AddSystem(
            MakeHardwareStation(self.scenario, meshcat=self.meshcat)
        )

        self.plant = self.station.GetSubsystemByName("plant")
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")

        # scene graph query output port
        self.query_output_port = self.scene_graph.GetOutputPort("query")

        self.diagram = builder.Build()

        # contexts
        self.context_diagram = self.diagram.CreateDefaultContext()
        self.context_station = self.diagram.GetSubsystemContext(
            self.station, self.context_diagram
        )
        self.station.GetInputPort(f"{self.robot_name}.position").FixValue(
            self.context_station, np.zeros(7)
        )
        self.station.GetInputPort(f"{self.gripper_name}.position").FixValue(
            self.context_station, [0.1]
        )
        self.context_scene_graph = self.station.GetSubsystemContext(
            self.scene_graph, self.context_station
        )
        self.context_plant = self.station.GetMutableSubsystemContext(
            self.plant, self.context_station
        )

        # mark initial configuration
        self.gripper_setpoint = gripper_setpoint
        if q_robot is None:
            self.q0 = self.plant.GetPositions(
                self.context_plant, self.plant.GetModelInstanceByName(self.robot_name)
            )
        else:
            self.q0 = q_robot
            self.SetStationConfiguration(q_robot, gripper_setpoint)

        self.DrawStation(self.q0, 0.1)
        query_object = self.query_output_port.Eval(self.context_scene_graph)
        self.okay_collisions = len(query_object.ComputePointPairPenetration())

    def clear_meshcat(self) -> None:
        """Clear meshcat visualization."""
        if self.meshcat is not None:
            self.meshcat.Delete()
            self.meshcat.DeleteAddedControls()

    def SetStationConfiguration(self, q_robot: tuple, gripper_setpoint: float) -> None:
        """
        Set robot and gripper configuration.
        
        Args:
            q_robot: (7,) tuple, joint angles in radians
            gripper_setpoint: float, gripper opening distance in meters
        """
        self.plant.SetPositions(
            self.context_plant,
            self.plant.GetModelInstanceByName(self.robot_name),
            q_robot,
        )
        self.plant.SetPositions(
            self.context_plant,
            self.plant.GetModelInstanceByName(self.gripper_name),
            [-gripper_setpoint / 2, gripper_setpoint / 2],
        )

    def DrawStation(self, q_robot: tuple, gripper_setpoint: float = 0.1) -> None:
        """Draw current configuration in meshcat."""
        self.SetStationConfiguration(q_robot, gripper_setpoint)
        self.diagram.ForcedPublish(self.context_diagram)

    def ExistsCollision(self, q_robot: tuple, gripper_setpoint: float) -> bool:
        """
        Check for unwanted collision at given configuration.
        
        Args:
            q_robot: robot joint configuration
            gripper_setpoint: gripper width
            
        Returns:
            True if unwanted collision exists, False otherwise
        """
        self.SetStationConfiguration(q_robot, gripper_setpoint)
        query_object = self.query_output_port.Eval(self.context_scene_graph)
        collision_pairs = query_object.ComputePointPairPenetration()

        return len(collision_pairs) > self.okay_collisions


# RRT utilities from reference notebook
@dataclass
class Range:
    """Range for a configuration space dimension."""
    low: float
    high: float


class TreeNode:
    """Tree node for RRT."""
    def __init__(self, value: tuple, parent=None):
        self.value = value
        self.parent = parent


class ConfigurationSpace:
    """Configuration space with distance metric and path generation."""
    
    def __init__(self, ranges: List[Range], distance_fn, max_steps: List[float]):
        self.ranges = ranges
        self.distance = distance_fn
        self.max_steps = np.array(max_steps)
        self.dim = len(ranges)
    
    def sample(self) -> tuple:
        """Sample random configuration."""
        q = []
        for r in self.ranges:
            q.append(np.random.uniform(r.low, r.high))
        return tuple(q)
    
    def path(self, start: tuple, end: tuple) -> List[tuple]:
        """Generate path from start to end."""
        start_arr = np.array(start)
        end_arr = np.array(end)
        delta = end_arr - start_arr
        
        # Number of steps based on max step sizes
        n_steps = np.max(np.abs(delta / self.max_steps))
        n_steps = max(1, int(np.ceil(n_steps)))
        
        path = []
        for i in range(n_steps + 1):
            alpha = i / n_steps
            q = start_arr + alpha * delta
            path.append(tuple(q))
        
        return path


class RRT:
    """RRT tree structure."""
    
    def __init__(self, root: TreeNode, cspace: ConfigurationSpace):
        self.root = root
        self.cspace = cspace
        self.nodes = [root]
    
    def nearest(self, q: tuple) -> TreeNode:
        """Find nearest node to q."""
        best_node = self.nodes[0]
        best_dist = self.cspace.distance(q, best_node.value)
        
        for node in self.nodes[1:]:
            dist = self.cspace.distance(q, node.value)
            if dist < best_dist:
                best_dist = dist
                best_node = node
        
        return best_node
    
    def add_configuration(self, parent: TreeNode, q: tuple) -> TreeNode:
        """Add new configuration as child of parent."""
        new_node = TreeNode(q, parent)
        self.nodes.append(new_node)
        return new_node


class RRT_Connect_tools:
    """RRT-Connect tools from reference notebook."""
    
    def __init__(
        self,
        sim: ManipulationStationSim,
        start: tuple,
        goal: tuple,
    ) -> None:
        self.sim = sim
        self.start = start
        self.goal = goal

        nq = 7
        joint_limits = np.zeros((nq, 2))
        for i in range(nq):
            # Franka joints are named panda_joint1, panda_joint2, etc. (no underscore)
            joint = sim.plant.GetJointByName(f"{sim.robot_name}_joint{i+1}")
            joint_limits[i, 0] = joint.position_lower_limits()[0]
            joint_limits[i, 1] = joint.position_upper_limits()[0]

        range_list = []
        for joint_limit in joint_limits:
            range_list.append(Range(joint_limit[0], joint_limit[1]))

        def l2_distance(q1: tuple, q2: tuple = None):
            if q2 is None:
                # Distance from origin
                sum = 0
                for q_i in q1:
                    sum += q_i**2
                return np.sqrt(sum)
            else:
                # Distance between two configurations
                sum = 0
                for q1_i, q2_i in zip(q1, q2):
                    sum += (q1_i - q2_i)**2
                return np.sqrt(sum)

        # Use smaller step size (1.0 degree vs 1.5) for finer collision checking
        # This helps avoid missing collisions when letter is grasped
        max_steps = nq * [np.pi / 180 * 1.0]  # 1.0 degrees (more conservative than reference)
        self.cspace = ConfigurationSpace(range_list, l2_distance, max_steps)
        self.rrt_tree_start = RRT(TreeNode(start), self.cspace)
        self.rrt_tree_goal = RRT(TreeNode(goal), self.cspace)

    def sample_node_in_configuration_space(self) -> tuple:
        """Sample a random valid configuration from the c-space."""
        q_sample = self.cspace.sample()
        return q_sample

    def calc_intermediate_qs_wo_collision(
        self, start: tuple, end: tuple
    ) -> List[tuple]:
        """
        Check if path from start to end collides with obstacles.

        Returns:
            list of tuples along the path that are not in collision
        """
        path = self.cspace.path(start, end)
        safe_path = []
        for configuration in path:
            # Use gripper_setpoint of 0.1m (10cm open) as in reference notebook
            # This is the gripper width during motion planning, not a collision margin
            if self.sim.ExistsCollision(np.array(configuration), 0.1):
                return safe_path
            safe_path.append(configuration)
        return safe_path

    def backup_path_from_node(self, node: TreeNode) -> List[tuple]:
        """Reconstruct path from tree root to the given node."""
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path

    def extend_once(self, tree: RRT, q_target: tuple) -> Optional[TreeNode]:
        """Extend tree by one step toward q_target."""
        q_near_node = tree.nearest(q_target)
        edge = self.calc_intermediate_qs_wo_collision(q_near_node.value, q_target)
        if len(edge) <= 1:
            return None
        q_step = edge[1]
        new_node = tree.add_configuration(q_near_node, q_step)
        return new_node

    def connect_greedy(
        self, tree: RRT, q_target: tuple, eps: float = 1e-2
    ) -> Tuple[Optional[TreeNode], bool]:
        """
        Greedily add collision-free segments toward q_target.

        Returns:
            (last_node, complete): last_node reached; complete=True if within eps
        """
        near_node = tree.nearest(q_target)
        q_near_node = near_node.value
        path = self.calc_intermediate_qs_wo_collision(q_near_node, q_target)
        if len(path) > 1:
            last_node = near_node
            for j in range(1, len(path)):
                last_node = tree.add_configuration(last_node, path[j])

            return last_node, (self.cspace.distance(last_node.value, q_target) < eps)

        return (None, False)

    @staticmethod
    def concat_paths(path_a: List[tuple], path_b: List[tuple]) -> List[tuple]:
        """Concatenate two paths, de-duplicating shared joint."""
        if path_a and path_b and path_a[-1] == path_b[0]:
            return path_a + path_b[1:]
        return path_a + path_b


def check_equal(a: tuple, b: tuple, atol: float = 1e-9) -> bool:
    """Robust waypoint equality for tuples/ndarrays."""
    return np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=0.0)


def splice_with_shortcut(
    path: list, i: int, j: int, edge: list
) -> list:
    """
    Replace the inclusive subpath path[i:j+1] with 'edge' (which connects
    path[i] -> path[j]). Returns a new path list.
    Assumes edge[0] == path[i] and edge[-1] == path[j].
    """
    prefix = path[:i]
    suffix = path[j + 1 :]
    return prefix + edge + suffix


def shortcut_path(
    sim: ManipulationStationSim,
    path: list,
    passes: int = 400,
    min_separation: int = 2,
) -> list:
    """
    Randomized shortcutting / smoothing.

    Each pass:
      1) Randomly select indices i < j with j >= i + min_separation.
      2) Try a direct connection using calc_intermediate_qs_wo_collision.
      3) If the connection fully reaches path[j], splice it in.
    
    Returns:
        Smoothed path (list[tuple]).
    """
    if not path or len(path) < 3:
        return path

    tools = RRT_Connect_tools(sim, path[0], path[-1])
    rng = np.random.default_rng()

    current = path

    for _ in range(passes):
        n = len(current)
        # need at least 3 waypoints and room for separation
        if n < 3 or n <= min_separation:
            break

        # choose i, j with j >= i + min_separation
        i = int(rng.integers(0, n - min_separation))
        j = int(rng.integers(i + min_separation, n))

        q_a = current[i]
        q_b = current[j]

        # attempt a direct, collision-free edge from q_a→q_b
        edge = tools.calc_intermediate_qs_wo_collision(q_a, q_b)
        
        # success only if edge exists and fully reaches q_b
        if len(edge) > 1 and check_equal(edge[0], q_a) and check_equal(edge[-1], q_b):
            current = splice_with_shortcut(current, i, j, edge)
    
    # final dedup / tidy (drop immediate duplicates)
    cleaned = []
    prev = None
    for q in current:
        if prev is None or not check_equal(q, prev):
            cleaned.append(q)
        prev = q

    return cleaned


def rrt_connect_planning(
    sim: ManipulationStationSim,
    q_start: tuple,
    q_goal: tuple,
    max_iterations: int = 2000,
    eps: float = 1e-2,
) -> Tuple[Optional[List[tuple]], int]:
    """
    RRT-Connect motion planning (from reference notebook).
    
    Returns:
        (path, num_iterations): path as list of configs, or None if failed
    """
    tools = RRT_Connect_tools(sim, start=q_start, goal=q_goal)
    T_start = tools.rrt_tree_start
    T_goal = tools.rrt_tree_goal

    for it in range(max_iterations):
        # 1) sample in C-space
        q_rand = tools.sample_node_in_configuration_space()

        # 2) alternate active/other trees
        active_is_start = (it % 2 == 0)
        T_active = T_start if active_is_start else T_goal
        T_other = T_goal if active_is_start else T_start

        # 3) single-step extend of active tree toward q_rand
        node_a = tools.extend_once(T_active, q_rand)
        if node_a is None:
            continue
        q_new = node_a.value

        # 4) greedy connect of the other tree toward q_new
        node_b, complete = tools.connect_greedy(T_other, q_new, eps)

        # 5) if connected, splice paths and return
        if complete and node_b is not None:
            path_a = tools.backup_path_from_node(node_a)
            path_b = tools.backup_path_from_node(node_b)

            # ensure start -> ... -> goal ordering
            if not active_is_start:
                path_a, path_b = path_b, path_a

            full_path = tools.concat_paths(path_a, list(reversed(path_b)))
            return full_path, it + 1

    return None, max_iterations


def solve_ik_for_pose(
    plant,
    X_WG_target: RigidTransform,
    q_nominal: tuple = tuple(np.array([0.0, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0])),
    theta_bound: float = 0.01 * np.pi,
    pos_tol: float = 0.015,
) -> tuple:
    """
    Solve IK for a single end-effector pose (from reference notebook).
    
    Args:
        plant: MultibodyPlant
        X_WG_target: Target gripper pose
        q_nominal: Nominal configuration for joint centering
        theta_bound: Orientation tolerance (radians)
        pos_tol: Position tolerance (meters)
    
    Returns:
        tuple: Joint configuration (7 DOF)
    """
    world_frame = plant.world_frame()
    # Try Franka hand frame first, fall back to WSG body for backward compatibility
    try:
        gripper_frame = plant.GetFrameByName("panda_hand")  # Franka hand
    except RuntimeError:
        gripper_frame = plant.GetFrameByName("body")  # WSG body (legacy)

    ik = InverseKinematics(plant)
    q_vars = ik.q()[:7]
    prog = ik.prog()

    # Orientation: align gripper frame with target within theta_bound
    ik.AddOrientationConstraint(
        frameAbar=gripper_frame,
        R_AbarA=RotationMatrix(),  # identity in gripper frame
        frameBbar=world_frame,
        R_BbarB=X_WG_target.rotation(),
        theta_bound=theta_bound,
    )

    # Position: place gripper origin within a box of size 2*pos_tol around target
    p_WG = X_WG_target.translation()
    ik.AddPositionConstraint(
        frameB=gripper_frame,
        p_BQ=[0.0, 0.0, 0.0],  # gripper origin
        frameA=world_frame,
        p_AQ_lower=p_WG - pos_tol,
        p_AQ_upper=p_WG + pos_tol,
    )

    # Joint-centering cost around q_nominal
    prog.AddQuadraticErrorCost(np.eye(7), np.array(q_nominal), q_vars)

    # Initial guess
    prog.SetInitialGuess(q_vars, q_nominal)

    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError("IK did not succeed")

    return tuple(result.GetSolution(q_vars))


def _make_R_with_y_along_dir_and_z_up(dir_world: np.ndarray) -> RotationMatrix:
    """
    Create rotation whose +Y axis points along dir_world, and +Z is world-up.
    From reference notebook.
    """
    zW = np.array([0.0, 0.0, 1.0])  # keep gripper "upright"
    yW = dir_world / (np.linalg.norm(dir_world) + 1e-12)
    # if yW almost parallel to zW, nudge slightly
    if abs(np.dot(yW, zW)) > 0.99:
        zW = np.array([0.0, 1.0, 0.0])
    xW = np.cross(yW, zW)
    xW /= np.linalg.norm(xW) + 1e-12
    zW = np.cross(xW, yW)
    zW /= np.linalg.norm(zW) + 1e-12
    R = np.column_stack([xW, yW, zW])
    return RotationMatrix(R)


def _ensure_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _jitter_vec(
    rng: np.random.Generator,
    scale: np.ndarray,
    strength: float,
) -> np.ndarray:
    if strength <= 1e-6:
        return np.zeros_like(scale)
    return strength * (rng.uniform(-1.0, 1.0, size=scale.shape) * scale)


def design_grasp_pose(
    X_WO: RigidTransform,
    *,
    randomize: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> RigidTransform:
    """Design grasp pose from reference notebook."""
    rng = _ensure_rng(rng)
    p_WO = X_WO.translation()
    
    # Approach from +Y side (shelf/outer side)
    APPROACH_DIR_WORLD = np.array([0.0, 1.0, 0.0])
    
    # Gripper rotation: +Y points along approach direction, +Z up
    R_WG = _make_R_with_y_along_dir_and_z_up(APPROACH_DIR_WORLD)
    
    # Place hand just outside letter along approach axis, with small lift
    yhat = R_WG.matrix()[:, 1]
    p_WG = p_WO - 0.07 * yhat + np.array([-0.01, 0.0, 0.015])
    p_WG += _jitter_vec(rng, np.array([0.02, 0.015, 0.015]), randomize)
    
    return RigidTransform(R_WG, p_WG)


def design_pregrasp_pose(
    X_WG: RigidTransform,
    *,
    randomize: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> RigidTransform:
    """Design pre-grasp pose from reference notebook."""
    rng = _ensure_rng(rng)
    # Keep same rotation as grasp; only translate back along +Y and up
    R_WG = X_WG.rotation()
    yhat = R_WG.matrix()[:, 1]
    retreat = 0.18 + _jitter_vec(rng, np.array([0.03]), randomize)[0]
    lift = np.array([0.0, 0.0, 0.03]) + _jitter_vec(rng, np.array([0.0, 0.0, 0.01]), randomize)
    p_WG = X_WG.translation() - retreat * yhat + lift
    p_WG += _jitter_vec(rng, np.array([0.015, 0.01, 0.01]), randomize)
    return RigidTransform(R_WG, p_WG)


def design_goal_poses(
    *,
    randomize: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> RigidTransform:
    """Design goal pose adapted for Franka with more clearance."""
    rng = _ensure_rng(rng)
    # Different orientation than grasp
    base_rpy = np.array([0.0, 0.0, 3 * np.pi / 2])
    rpy_jitter = _jitter_vec(rng, np.array([0.05, 0.05, 0.2]), randomize)
    R_WG_goal = RollPitchYaw(*(base_rpy + rpy_jitter)).ToRotationMatrix()
    # Position on top shelf - moved forward (reduced X) from 0.74 to 0.70 for more clearance
    # This gives 20cm clearance from shelf at x=0.9 instead of 16cm
    p_goal = np.array([0.70, 0.0, 0.58])
    p_goal += _jitter_vec(rng, np.array([0.04, 0.05, 0.03]), randomize)
    return RigidTransform(R_WG_goal, p_goal)


def get_initial_pose(plant, body_name: str, plant_context) -> RigidTransform:
    """
    Get initial pose of body, accounting for COM offset.
    From reference notebook.
    """
    body = plant.GetBodyByName(body_name)
    X_WS = plant.EvalBodyPoseInWorld(plant_context, body)
    X_SO = RigidTransform(body.default_spatial_inertia().get_com())
    return X_WS @ X_SO


def plan_scene_trajectory(
    letter: str = "C",
    meshcat=None,
    assets_dir: Optional[Path] = None,
    *,
    randomize: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[tuple], List[tuple], List[tuple], np.ndarray, np.ndarray]:
    """
    Complete trajectory planning for Scene D (reference notebook approach).
    
    This exactly replicates the reference notebook's workflow:
    1. Create two scenarios (base and grasp)
    2. Compute keyframe poses
    3. Solve IK for all keyframes
    4. Plan paths using RRT-Connect with appropriate scenarios
    
    Args:
        letter: Letter to manipulate (e.g., "C")
        meshcat: Meshcat instance for visualization
        assets_dir: Path to assets directory
    
    Returns:
        (path_pick, path_place, path_reset): Three trajectory segments
    """
    import tempfile
    
    rng = _ensure_rng(rng)
    randomize = max(0.0, float(randomize))
    planning_seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(planning_seed)
    
    if assets_dir is None:
        assets_dir = Path(__file__).resolve().parent.parent / "assets"
    
    # === Step 1: Create scenario files ===
    print(f"Creating scenario files for letter {letter}...")
    
    # Read base scenario template
    scenario_base_template = assets_dir / "franka_shelves_scenario.yaml"
    with open(scenario_base_template, 'r') as f:
        yaml_base = f.read()
    
    # Read grasp scenario template
    scenario_grasp_template = assets_dir / "franka_shelves_scenario_grasp.yaml"
    with open(scenario_grasp_template, 'r') as f:
        yaml_grasp = f.read()
    
    # Replace placeholders
    letter_path = assets_dir / f"{letter}_model" / f"{letter}.sdf"
    if not letter_path.exists():
        raise FileNotFoundError(f"Missing letter asset: {letter_path}")
    letter_file = f"file://{letter_path}"

    letter_big_path = assets_dir / f"{letter}_big_model" / f"{letter}.sdf"
    if letter_big_path.exists():
        letter_grasp_file = f"file://{letter_big_path}"
    else:
        letter_grasp_file = letter_file

    letter_body_name = f"{letter}_body_link"
    
    yaml_base = yaml_base.replace("LETTER_FILE_PLACEHOLDER", letter_file)
    yaml_base = yaml_base.replace("LETTER_BODY_NAME", letter_body_name)
    letter_translation_base = np.array([0.62, -0.05, 0.26])
    letter_translation = letter_translation_base + _jitter_vec(
        rng, np.array([0.05, 0.06, 0.03]), randomize
    )
    yaml_base = yaml_base.replace(
        "translation: [0.62, -0.05, 0.26]",
        f"translation: [{letter_translation[0]:.4f}, {letter_translation[1]:.4f}, {letter_translation[2]:.4f}]",
    )
    letter_yaw = 180.0 + randomize * rng.uniform(-12.0, 12.0)
    yaml_base = yaml_base.replace(
        "rotation: !Rpy { deg: [0, 0, 180] }",
        f"rotation: !Rpy {{ deg: [0, 0, {letter_yaw:.2f}] }}",
    )
    # Slightly move shelf clutter to vary obstacles
    clutter_specs = [
        ("translation: [0.05, -0.18, 0.245]", np.array([0.02, 0.02, 0.015])),
        ("translation: [0.1, 0.15, -0.035]", np.array([0.02, 0.02, 0.02])),
        ("translation: [0.1, -0.17, -0.072]", np.array([0.02, 0.02, 0.015])),
    ]
    for original, scale in clutter_specs:
        jitter = _jitter_vec(rng, scale, randomize)
        if not np.any(jitter):
            continue
        base_vals = np.fromstring(original.split("[")[1].split("]")[0], sep=",")
        new_vals = base_vals + jitter
        yaml_base = yaml_base.replace(
            original,
            f"translation: [{new_vals[0]:.4f}, {new_vals[1]:.4f}, {new_vals[2]:.4f}]",
        )
    
    # For grasp scenario, use letter weld transform adapted for Franka
    # Reference notebook uses [0.07, 0.2, -0.03] for IIWA, but we adjust Y to be 
    # more conservative and avoid side collisions with the wider Franka workspace
    yaml_grasp = yaml_grasp.replace("LETTER_FILE_PLACEHOLDER", letter_grasp_file)
    yaml_grasp = yaml_grasp.replace("LETTER_BODY_NAME", letter_body_name)
    weld_translation = np.array([0.07, 0.15, -0.03]) + _jitter_vec(
        rng, np.array([0.01, 0.015, 0.01]), randomize
    )
    yaml_grasp = yaml_grasp.replace(
        "LETTER_WELD_TRANSLATION",
        f"[{weld_translation[0]:.4f}, {weld_translation[1]:.4f}, {weld_translation[2]:.4f}]",
    )
    weld_yaw = 180.0 + randomize * rng.uniform(-8.0, 8.0)
    yaml_grasp = yaml_grasp.replace("LETTER_WELD_ROTATION", f"[0, 0, {weld_yaw:.2f}]")
    
    # Write to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='_base.yaml', delete=False) as tmp:
        tmp.write(yaml_base)
        scenario_base_file = tmp.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_grasp.yaml', delete=False) as tmp:
        tmp.write(yaml_grasp)
        scenario_grasp_file = tmp.name
    
    try:
        # === Step 2: Initialize simulation with base scenario ===
        print(f"Loading base scenario...")
        sim = ManipulationStationSim(meshcat=meshcat)
        
        # Home configuration
        q_home_nominal = np.array([0.0, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0])
        q_home = q_home_nominal + _jitter_vec(
            rng, np.array([0.05, 0.1, 0.05, 0.1, 0.05, 0.08, 0.05]), randomize
        )
        sim.choose_sim(scenario_base_file, q_robot=tuple(q_home), gripper_setpoint=0.1)
        
        # === Step 3: Compute keyframe poses ===
        print(f"Computing keyframes...")
        
        # Get initial poses
        X_WGinitial = sim.plant.EvalBodyPoseInWorld(
            sim.context_plant, sim.plant.GetBodyByName("body")
        )
        X_WOinitial = get_initial_pose(
            sim.plant, letter_body_name, sim.context_plant
        )
        
        # Design keyframes
        X_WGgrasp = design_grasp_pose(X_WOinitial, randomize=randomize, rng=rng)
        X_WGapproach = design_pregrasp_pose(X_WGgrasp, randomize=randomize, rng=rng)
        X_WGgoal = design_goal_poses(randomize=randomize, rng=rng)
        
        # === Step 4: Solve IK for all keyframes ===
        print(f"Solving IK...")
        
        q_initial = solve_ik_for_pose(sim.plant, X_WGinitial, q_nominal=tuple(q_home))
        q_approach = solve_ik_for_pose(sim.plant, X_WGapproach, q_nominal=q_initial)
        q_grasp = solve_ik_for_pose(sim.plant, X_WGgrasp, q_nominal=q_approach)
        q_goal = solve_ik_for_pose(sim.plant, X_WGgoal, q_nominal=q_grasp)
        
        print(f"IK solutions found for all keyframes")
        
        # === Step 5: Plan path segments ===
        print(f"Planning trajectory segments...")
        
        # Segment 1: Initial -> Approach (with FREE letter, base scenario)
        print(f"Planning pick path (initial -> approach)...")
        sim.choose_sim(scenario_base_file, q_robot=q_initial)
        path_pick, num_iter = rrt_connect_planning(
            sim, q_initial, q_approach, max_iterations=3000
        )
        if path_pick is None:
            raise RuntimeError("Pick path planning failed")
        print(f"Pick path found ({num_iter} iterations, {len(path_pick)} waypoints)")
        
        # Segment 2: Approach -> Goal (with WELDED letter, grasp scenario)
        print(f"Planning place path (approach -> goal)...")
        sim.choose_sim(scenario_grasp_file, q_robot=q_approach)
        path_place, num_iter = rrt_connect_planning(
            sim, q_approach, q_goal, max_iterations=8000
        )
        if path_place is None:
            raise RuntimeError("Place path planning failed")
        print(f"Place path found ({num_iter} iterations, {len(path_place)} waypoints)")
        
        # Segment 3: Goal -> Initial (return to home, base scenario)
        print(f"Planning reset path (goal -> initial)...")
        sim.choose_sim(scenario_base_file, q_robot=q_goal)
        path_reset, num_iter = rrt_connect_planning(
            sim, q_goal, q_initial, max_iterations=2000
        )
        if path_reset is None:
            print(f"Reset path planning failed, using interpolation")
            path_reset = [q_goal, q_initial]
        else:
            print(f"Reset path found ({num_iter} iterations, {len(path_reset)} waypoints)")
        
        print(f"All trajectory segments planned successfully!")
        
        # === Step 6: Apply shortcutting to smooth paths ===
        print(f"Applying shortcutting to smooth paths...")
        
        sim.choose_sim(scenario_base_file, q_robot=q_initial)
        short_path_pick = shortcut_path(sim, path_pick, passes=200, min_separation=2)
        print(f"Pick path: {len(path_pick)} → {len(short_path_pick)} waypoints")
        
        sim.choose_sim(scenario_grasp_file, q_robot=q_approach)
        short_path_place = shortcut_path(sim, path_place, passes=200, min_separation=2)
        print(f"Place path: {len(path_place)} → {len(short_path_place)} waypoints")
        
        sim.choose_sim(scenario_base_file, q_robot=q_goal)
        short_path_reset = shortcut_path(sim, path_reset, passes=200, min_separation=2)
        print(f"Reset path: {len(path_reset)} → {len(short_path_reset)} waypoints")
        
        return short_path_pick, short_path_place, short_path_reset, q_grasp, q_approach
        
    finally:
        # Clean up temp files
        Path(scenario_base_file).unlink(missing_ok=True)
        Path(scenario_grasp_file).unlink(missing_ok=True)
