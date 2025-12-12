from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import math

from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.systems.analysis import Simulator
from pydrake.all import JacobianWrtVariable, InverseKinematics, Solve
from pydrake.multibody.tree import WeldJoint

@dataclass
class Subgoal:
    name: str
    pose_world: np.ndarray  # 4x4 SE3 pose

def _ee_frame(plant):
    """Get end-effector frame for Franka Panda arm with gripper."""
    # Try different gripper frames in order of preference
    # 1. panda_hand: Scene C (arm+hand combined model)
    # 2. body: Scene D (separate wsg gripper)  
    # 3. panda_link8: Fallback for arm-only
    try:
        return plant.GetBodyByName("panda_hand").body_frame()
    except:
        pass
    try:
        return plant.GetBodyByName("body").body_frame()
    except:
        pass
    return plant.GetBodyByName("panda_link8").body_frame()

def _get_manipuland_body_name(plant):
    """Detect the manipuland body name.
    
    Tries in order:
    1. block_body (scene C)
    2. manipuland_body (old reference scenes)
    3. Letter bodies like C_body_link, P_body_link, etc. (scene D and DrakeWorld scenes)
    """
    try:
        plant.GetBodyByName("block_body")
        return "block_body"
    except:
        pass
    
    try:
        plant.GetBodyByName("manipuland_body")
        return "manipuland_body"
    except:
        pass
    
    # Try letter body names (C_body_link, P_body_link, etc.)
    for letter in ["C", "P", "L", "E", "R", "A"]:
        try:
            body_name = f"{letter}_body_link"
            plant.GetBodyByName(body_name)
            return body_name
        except:
            pass
    
    raise RuntimeError("No manipuland found (expected 'block_body', 'manipuland_body', or letter body like 'C_body_link')")

def _get_gripper_body_name(plant):
    """Detect the gripper body name.
    
    Tries in order:
    1. panda_hand (scene C with arm+hand model)
    2. body (scene D with separate wsg gripper)
    """
    try:
        plant.GetBodyByName("panda_hand")
        return "panda_hand"
    except:
        pass
    
    try:
        plant.GetBodyByName("body")
        return "body"
    except:
        pass
    
    raise RuntimeError("No gripper found (expected 'panda_hand' or 'body')")

def _get_arm_joints(plant):
    """Get list of Franka Panda arm joint names."""
    return [f"panda_joint{i}" for i in range(1, 8)]

def _get_joint_limits(plant):
    """Get joint position limits for all plant positions."""
    from pydrake.multibody.tree import JointIndex
    nq = plant.num_positions()
    q_lower = np.full(nq, -np.pi)  # Default to reasonable joint limits
    q_upper = np.full(nq, np.pi)
    
    for i in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(i))
        start = joint.position_start()
        n_pos = joint.num_positions()
        for j in range(n_pos):
            lower = joint.position_lower_limits()[j]
            upper = joint.position_upper_limits()[j]
            # Clamp infinite bounds to reasonable values
            if np.isfinite(lower):
                q_lower[start + j] = lower
            if np.isfinite(upper):
                q_upper[start + j] = upper
    
    return q_lower, q_upper

def compute_subgoals(scene, task_name: str) -> List[Subgoal]:
    """Compute subgoals for pick-and-place task.
    
    Detects scene type and uses appropriate keyframe generation:
    - Scene C (block + bin): Vertical grasp from above, place in bin
    - Scene D (shelf): Horizontal approach from side, place on shelf
    """
    plant = scene.plant
    plant_context = plant.GetMyMutableContextFromRoot(scene.context)
    
    # Get manipuland's actual position in the scene
    manipuland_body_name = _get_manipuland_body_name(plant)
    block_body = plant.GetBodyByName(manipuland_body_name)
    X_WB = plant.CalcRelativeTransform(plant_context, plant.world_frame(), block_body.body_frame())
    
    # Detect scene type by checking for bin (scene C) vs shelves (scene D)
    has_bin = False
    has_shelves = False
    try:
        plant.GetBodyByName("bin")
        has_bin = True
    except:
        pass
    try:
        plant.GetBodyByName("shelves_body")
        has_shelves = True
    except:
        pass
    
    if has_shelves:
        # Scene D: Reference shelf pick-and-place (horizontal approach from side)
        return _compute_shelf_subgoals(plant, plant_context, X_WB)
    elif has_bin:
        # Scene C: Block pick-and-place (vertical grasp from above)
        return _compute_bin_subgoals(plant, plant_context, X_WB)
    else:
        raise RuntimeError("Unknown scene type: no bin or shelves found")

def _make_R_with_y_along_dir_and_z_up(dir_world: np.ndarray) -> RotationMatrix:
    """Create rotation with +Y along dir_world and +Z up (from reference notebook)."""  
    zW = np.array([0.0, 0.0, 1.0])  # keep gripper upright
    yW = dir_world / (np.linalg.norm(dir_world) + 1e-12)
    if abs(np.dot(yW, zW)) > 0.99:
        zW = np.array([0.0, 1.0, 0.0])
    xW = np.cross(yW, zW); xW /= (np.linalg.norm(xW) + 1e-12)
    zW = np.cross(xW, yW); zW /= (np.linalg.norm(zW) + 1e-12)
    R = np.column_stack([xW, yW, zW])
    return RotationMatrix(R)

def _compute_shelf_subgoals(plant, plant_context, X_WB) -> List[Subgoal]:
    """Compute subgoals for scene D (shelf pick-and-place, reference notebook approach).""" 
    p_WO = X_WB.translation()
    
    # Approach from +Y side (shelf/outer side) - exactly from reference notebook
    APPROACH_DIR_WORLD = np.array([0.0, 1.0, 0.0])
    
    # Grasp pose: gripper +Y points along approach, +Z up (exactly from reference)
    R_WG = _make_R_with_y_along_dir_and_z_up(APPROACH_DIR_WORLD)
    yhat = R_WG.matrix()[:, 1]
    p_WG_grasp = p_WO - 0.07 * yhat + np.array([-0.01, 0.0, 0.015])
    X_WG_grasp = RigidTransform(R_WG, p_WG_grasp)
    
    # Pre-grasp: retreat along +Y and lift (exactly from reference)
    p_WG_pregrasp = p_WG_grasp - 0.18 * yhat + np.array([0.0, 0.0, 0.03])
    X_WG_pregrasp = RigidTransform(R_WG, p_WG_pregrasp)
    
    # Goal: place on shelf (exactly from reference notebook)
    # Position: [0.74, 0, 0.58] - top shelf, right of cheez-it box
    # Rotation: RollPitchYaw(0, 0, 3*pi/2) - different from grasp orientation
    from pydrake.math import RollPitchYaw
    R_WG_goal = RollPitchYaw(0.0, 0.0, 3 * np.pi / 2).ToRotationMatrix()
    p_WG_goal = np.array([0.74, 0.0, 0.58])
    X_WG_goal = RigidTransform(R_WG_goal, p_WG_goal)
    
    # Reference notebook uses only 4 keyframes: initial, approach, grasp, goal
    # No separate lift subgoal - motion planner handles the path
    return [
        Subgoal("pre_grasp", X_WG_pregrasp.GetAsMatrix4()),
        Subgoal("grasp", X_WG_grasp.GetAsMatrix4()),
        Subgoal("place", X_WG_goal.GetAsMatrix4()),
    ]

def _compute_bin_subgoals(plant, plant_context, X_WB) -> List[Subgoal]:
    """Compute subgoals for scene C (block + bin pick-and-place)."""
    from drake_env.scenes import BIN_SIZE, BLOCK_HALF_HEIGHT, TABLE_SURFACE_Z, WELD_OFFSET_Z
    
    p_WB = X_WB.translation()
    
    # Grasp from above
    R_WG_grasp = RotationMatrix(np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]))
    p_WG_grasp = p_WB + np.array([0.0, 0.0, 0.12])
    X_WG_grasp = RigidTransform(R_WG_grasp, p_WG_grasp)
    
    # Pre-grasp
    p_WG_pregrasp = p_WG_grasp + np.array([0.0, 0.0, 0.15])
    X_WG_pregrasp = RigidTransform(R_WG_grasp, p_WG_pregrasp)
    
    # Lift
    X_WG_lift = RigidTransform(R_WG_grasp, [p_WB[0], p_WB[1], 1.0])
    
    # Get bin position
    bin_body = plant.GetBodyByName("bin")
    X_Wbin = plant.EvalBodyPoseInWorld(plant_context, bin_body)
    bin_pos = X_Wbin.translation()
    bin_top_z = bin_pos[2] + BIN_SIZE[2]
    
    # Pre-place above bin
    X_WG_preplace = RigidTransform(R_WG_grasp, [bin_pos[0], bin_pos[1], 1.05])
    
    # Place into bin
    target_block_bottom = bin_top_z + 0.01
    gripper_place_z = target_block_bottom + BLOCK_HALF_HEIGHT + WELD_OFFSET_Z
    X_WG_place = RigidTransform(R_WG_grasp, [bin_pos[0], bin_pos[1], gripper_place_z])
    
    return [
        Subgoal("pre_grasp", X_WG_pregrasp.GetAsMatrix4()),
        Subgoal("grasp", X_WG_grasp.GetAsMatrix4()),
        Subgoal("lift", X_WG_lift.GetAsMatrix4()),
        Subgoal("pre_place", X_WG_preplace.GetAsMatrix4()),
        Subgoal("place", X_WG_place.GetAsMatrix4()),
    ]


def _min_signed_distance(scene, root_context) -> float:
    """Compute minimum signed distance between collision geometries.
    
    Falls back to penetration depth for geometry pairs that don't
    support signed distance queries (e.g., Box-Halfspace).
    """
    sg_context = scene.scene_graph.GetMyContextFromRoot(root_context)
    query = scene.scene_graph.get_query_output_port().Eval(sg_context)
    
    # Try signed distance first, but catch unsupported geometry pairs
    try:
        signed = query.ComputeSignedDistancePairwiseClosestPoints()
        if len(signed) > 0:
            return float(min([p.distance for p in signed]))
    except RuntimeError:
        # Signed distance not supported for some geometry pairs (e.g., Box-Halfspace)
        # Fall back to penetration depth
        pass
    
    # Use penetration depth as fallback
    pairs = query.ComputePointPairPenetration()
    if len(pairs) == 0:
        return 0.2  # No contacts, assume safe distance
    
    # Negative depth means penetration (collision)
    min_depth = min([-p.depth for p in pairs])
    return float(min_depth)

def _is_collision_free(scene, root_context, exclude_block: bool = False) -> bool:
    """Check if current configuration is collision-free.
    
    Args:
        scene: SceneHandles object
        root_context: Root context
        exclude_block: If True, ignore collisions with the block (for grasping)
    
    Returns:
        True if collision-free
    """
    sg_context = scene.scene_graph.GetMyContextFromRoot(root_context)
    query = scene.scene_graph.get_query_output_port().Eval(sg_context)
    pairs = query.ComputePointPairPenetration()
    
    if not exclude_block:
        # Check all collisions
        for p in pairs:
            if p.depth > 0.0:
                return False
        return True
    
    # Exclude collisions involving the block
    inspector = scene.scene_graph.model_inspector()
    plant = scene.plant
    
    # Get manipuland geometry IDs (works for both block_body and manipuland_body)
    try:
        manipuland_body_name = _get_manipuland_body_name(plant)
        block_body = plant.GetBodyByName(manipuland_body_name)
        block_frame_id = plant.GetBodyFrameIdOrThrow(block_body.index())
        block_geom_ids = set(inspector.GetGeometries(block_frame_id))
    except:
        # If no manipuland, just check all collisions
        block_geom_ids = set()
    
    # Check collisions, excluding those involving block
    for p in pairs:
        if p.depth > 0.0:
            # Skip if either geometry belongs to block
            if p.id_A not in block_geom_ids and p.id_B not in block_geom_ids:
                return False
    
    return True

def solve_ik_for_pose(scene, target_pose: RigidTransform, q_nominal: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Solve inverse kinematics for a target end-effector pose.
    
    Args:
        scene: SceneHandles object
        target_pose: Desired end-effector pose in world frame
        q_nominal: Nominal joint configuration for cost (default: home position)
    
    Returns:
        Full joint configuration (nq DOF) if IK succeeds, None otherwise
        Works with any scene (welded object: 9 DOF, free object: 16 DOF)
    """
    plant = scene.plant
    plant_context = plant.GetMyMutableContextFromRoot(scene.context)
    ee_frame = _ee_frame(plant)
    
    if q_nominal is None:
        q_nominal = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Franka home (7 DOF)
    
    ik = InverseKinematics(plant, plant_context)
    q_vars = ik.q()  # All plant positions
    prog = ik.prog()
    
    # Constrain non-robot DOF to stay fixed (if they exist)
    # Robot is always first 9 DOF (7 arm + 2 gripper)
    # Additional DOF (e.g., free-floating block) should remain fixed during IK
    q_current = plant.GetPositions(plant_context)
    nq = len(q_current)
    for i in range(9, nq):  # Fix any DOF beyond robot
        prog.AddBoundingBoxConstraint(q_current[i], q_current[i], q_vars[i])
    
    # Position constraint (matching reference notebook: 1.5cm tolerance)
    p_WG = target_pose.translation()
    pos_tol = 0.015
    ik.AddPositionConstraint(
        frameB=ee_frame,
        p_BQ=np.array([0.0, 0.0, 0.0]),  # End-effector origin
        frameA=plant.world_frame(),
        p_AQ_lower=p_WG - pos_tol,
        p_AQ_upper=p_WG + pos_tol,
    )
    
    # Orientation constraint (matching reference notebook: 0.01*pi ~= 1.8 degrees)
    theta_bound = 0.01 * np.pi
    ik.AddOrientationConstraint(
        frameAbar=ee_frame,
        R_AbarA=RotationMatrix(),
        frameBbar=plant.world_frame(),
        R_BbarB=target_pose.rotation(),
        theta_bound=theta_bound,
    )
    
    # Joint-centering cost for robot joints only (first 9 DOF)
    # Use SMALL weight so orientation constraint dominates
    q_full = q_current.copy()
    if len(q_nominal) == 7:
        q_full[:7] = q_nominal  # Set arm joints to nominal
        # Keep gripper at current position (indices 7-8)
    
    # Cost only on robot joints (not block/object) with SMALL weight
    # Adapt to actual DOF count
    Q = np.zeros((nq, nq))
    robot_dof = min(9, nq)  # First 9 DOF are robot (or fewer if plant has less)
    Q[:robot_dof, :robot_dof] = 0.01 * np.eye(robot_dof)  # Small weight
    prog.AddQuadraticErrorCost(Q, q_full, q_vars)
    prog.SetInitialGuess(q_vars, q_full)
    
    result = Solve(prog)
    if result.is_success():
        return result.GetSolution(q_vars)  # Return full state (nq DOF)
    return None

def plan_rrt_connect(
    scene,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    rng: np.random.Generator,
    max_iters: int = 5000,
    step: float = np.pi / 180 * 1.5,  # 1.5 degrees (matching reference notebook)
    exclude_block: bool = False,
    weld_transform: Optional[RigidTransform] = None,
) -> Optional[List[np.ndarray]]:
    """Plan collision-free path using RRT-Connect algorithm (reference notebook approach).
    
    Matches the reference notebook's RRT implementation with:
    - Step size: 1.5 degrees per step
    - Max iterations: 5000 (can be increased to 8000 for difficult segments)
    - Bidirectional tree search
    
    IMPORTANT: Works ONLY with robot DOF (9), never touches block DOF (7).
    
    Args:
        scene: SceneHandles object
        q_start: Start configuration (16 DOF, but only first 9 used)
        q_goal: Goal configuration (16 DOF, but only first 9 used)
        rng: Random number generator
        max_iters: Maximum iterations (default 5000, increase for difficult paths)
        step: Step size for extending tree (default 1.5 degrees)
        exclude_block: If True, ignore collisions with block (for grasping)
        weld_transform: If provided, block position is computed from gripper pose using this transform (X_GB)
    
    Returns:
        List of waypoints (16 DOF each) if successful, None otherwise
    """
    plant = scene.plant
    plant_context = plant.GetMyMutableContextFromRoot(scene.context)
    q_lower, q_upper = _get_joint_limits(plant)
    
    # Extract ONLY robot joints (9 DOF) - never touch block DOF
    q_robot_start = q_start[:9].copy()
    q_robot_goal = q_goal[:9].copy()
    q_robot_lower = q_lower[:9]
    q_robot_upper = q_upper[:9]
    
    # Store block state (will be appended to all waypoints)
    q_block_fixed = q_start[9:].copy()
    
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
            dists = [np.linalg.norm(q - n) for n in self.nodes]
            return int(np.argmin(dists))
        
        def path_to_root(self, idx):
            path = []
            while idx != -1:
                path.append(self.nodes[idx])
                idx = self.parents[idx]
            path.reverse()
            return path
    
    def collision_free_along(q_robot_from, q_robot_to, local_step):
        """Check collision along path (9-DOF robot only)."""
        d = q_robot_to - q_robot_from
        L = np.linalg.norm(d)
        if L < 1e-9:
            return True
        n = max(1, int(math.ceil(L / local_step)))
        for i in range(1, n + 1):
            q_robot = q_robot_from + d * (i / n)
            
            # Compute block position based on weld constraint if provided
            if weld_transform is not None:
                # Set robot positions to compute gripper pose
                q_temp = np.concatenate([q_robot, q_block_fixed])
                plant.SetPositions(plant_context, q_temp)
                
                # Compute block pose from gripper pose
                gripper_body_name = _get_gripper_body_name(plant)
                gripper_body = plant.GetBodyByName(gripper_body_name)
                X_WG = plant.EvalBodyPoseInWorld(plant_context, gripper_body)
                X_WB = X_WG @ weld_transform
                
                # Update block state
                q_block = np.concatenate([
                    X_WB.rotation().ToQuaternion().wxyz(),
                    X_WB.translation()
                ])
                q_full = np.concatenate([q_robot, q_block])
            else:
                # Block position is fixed
                q_full = np.concatenate([q_robot, q_block_fixed])
            
            plant.SetPositions(plant_context, q_full)
            if not _is_collision_free(scene, scene.context, exclude_block=exclude_block):
                return False
        return True
    
    T_a = Tree(q_robot_start)
    T_b = Tree(q_robot_goal)
    
    def has_valid_orientation(q_robot):
        """Check if gripper orientation is reasonable (9-DOF robot only)."""
        # Construct full 16-DOF state for FK
        q_full = np.concatenate([q_robot, q_block_fixed])
        plant.SetPositions(plant_context, q_full)
        ee_frame = _ee_frame(plant)
        X_WE = plant.CalcRelativeTransform(plant_context, plant.world_frame(), ee_frame)
        z_axis = X_WE.rotation().matrix()[:, 2]
        # Z-axis MUST point downward: z-component should be NEGATIVE
        # z = -1.0 is perfect down, z = 0.0 is horizontal, z = +1.0 is up
        # Require z < -0.5 (at least 60 degrees from horizontal, pointing down)
        return z_axis[2] < -0.5  # ONLY accept downward-pointing orientations
    
    def try_connect(tree_a, tree_b):
        # Sample random robot configuration (9-DOF ONLY)
        if rng.random() < 0.5:
            q_robot_rand = rng.uniform(q_robot_lower, q_robot_upper)
        else:
            q_robot_rand = tree_b.nodes[rng.integers(len(tree_b.nodes))]
        
        idx = tree_a.nearest(q_robot_rand)
        q_robot_near = tree_a.nodes[idx]
        q_robot_new, _ = steer(q_robot_near, q_robot_rand, step)
        
        # Check orientation constraint before adding to tree
        if not has_valid_orientation(q_robot_new):
            return None
        
        if collision_free_along(q_robot_near, q_robot_new, step):
            new_idx = tree_a.add(q_robot_new, idx)
            # Try to connect to tree_b
            idx_b = tree_b.nearest(q_robot_new)
            q_robot_near_b = tree_b.nodes[idx_b]
            if collision_free_along(q_robot_new, q_robot_near_b, step):
                # Success: assemble path (9-DOF robot configs)
                path_robot = tree_a.path_to_root(new_idx) + list(reversed(tree_b.path_to_root(idx_b)))
                # Construct full path with block state (welded or fixed)
                path_full = []
                for q_robot in path_robot:
                    if weld_transform is not None:
                        # Compute block pose from gripper pose
                        q_temp = np.concatenate([q_robot, q_block_fixed])
                        plant.SetPositions(plant_context, q_temp)
                        gripper_body_name = _get_gripper_body_name(plant)
                        gripper_body = plant.GetBodyByName(gripper_body_name)
                        X_WG = plant.EvalBodyPoseInWorld(plant_context, gripper_body)
                        X_WB = X_WG @ weld_transform
                        q_block = np.concatenate([
                            X_WB.rotation().ToQuaternion().wxyz(),
                            X_WB.translation()
                        ])
                        path_full.append(np.concatenate([q_robot, q_block]))
                    else:
                        path_full.append(np.concatenate([q_robot, q_block_fixed]))
                return path_full
        return None
    
    for it in range(max_iters):
        path = try_connect(T_a, T_b)
        if path is not None:
            return path
        # Swap roles
        path = try_connect(T_b, T_a)
        if path is not None:
            return path
    
    return None

def plan_and_rollout(scene, subgoals: List[Subgoal], episode_length_sec=10.0, hz=4, use_rrt: bool = False, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """Plan and execute pick-and-place trajectory.
    
    Args:
        scene: SceneHandles object
        subgoals: List of subgoal poses
        episode_length_sec: Duration of episode
        hz: Control frequency
        use_rrt: If True, use RRT-Connect for motion planning
        rng: Random number generator (required if use_rrt=True)
    
    Returns:
        Dictionary with episode data
    """
    plant = scene.plant
    sim = scene.simulator
    root_context = scene.context
    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    
    # Detect manipuland body name (works for both scene C and scene D)
    manipuland_body_name = _get_manipuland_body_name(plant)
    dt = 1.0/float(hz)
    steps = int(episode_length_sec*hz)

    nq = plant.num_positions(); nv = plant.num_velocities()
    # Use current positions from context (should already be at home configuration)
    q = plant.GetPositions(plant_context).copy()
    v = plant.GetVelocities(plant_context).copy()
    # Positions and velocities are already set in the context from build_scene
    
    # If using RRT, pre-compute waypoints
    waypoints = None
    if use_rrt and rng is not None:
        # Save initial scene state before planning
        q_initial_scene = plant.GetPositions(plant_context).copy()
        
        waypoints = []
        q_current = q.copy()  # Use full state (16 DOF)
        weld_transform_planning = None  # Will be set after grasp
        
        for i, sg in enumerate(subgoals):
            R_WG = RotationMatrix(np.asarray(sg.pose_world[:3, :3], dtype=np.float64))
            p_WG = np.asarray(sg.pose_world[:3, 3], dtype=np.float64)
            T_WG = RigidTransform(R_WG, p_WG.tolist())
            
            # Solve IK for subgoal (matching reference notebook approach)
            # Use optimization-based IK with sequential solving
            # Each solution becomes the nominal for the next subgoal
            q_target = solve_ik_for_pose(scene, T_WG, q_current[:7])  # Pass only arm joints as nominal
            if q_target is None:
                print(f"IK failed for subgoal {sg.name}")
                continue
            
            # After grasp subgoal, compute weld constraint for subsequent planning
            if sg.name == "grasp" and weld_transform_planning is None:
                # Compute weld transform from the IK solution (q_target)
                # This represents where the gripper will be relative to block at grasp
                # We compute this WITHOUT modifying the scene context
                gripper_body_name = _get_gripper_body_name(plant)
                gripper_body = plant.GetBodyByName(gripper_body_name)
                block_body = plant.GetBodyByName(manipuland_body_name)
                
                # Save current scene state
                q_scene_backup = plant.GetPositions(plant_context).copy()
                
                # Temporarily set grasp configuration to compute transform
                q_grasp = q_target.copy()
                q_grasp[7:9] = [0.0, 0.0]  # Close gripper
                plant.SetPositions(plant_context, q_grasp)
                
                # Compute weld transform X_GB
                X_WG = plant.EvalBodyPoseInWorld(plant_context, gripper_body)
                X_WB = plant.EvalBodyPoseInWorld(plant_context, block_body)
                weld_transform_planning = X_WG.inverse() @ X_WB
                # Restore scene state immediately
                plant.SetPositions(plant_context, q_scene_backup)
            
            # Plan path from current to target (both are 16-DOF)
            # Exclude block from collision checking for grasp/lift motions
            exclude_block = sg.name in ["grasp", "lift"]
            
            # Use different max_iters for different segments (matching reference notebook)
            # Place segment is hardest (with letter welded to gripper) so needs more iterations
            max_iters_for_segment = {
                "pre_grasp": 3000,
                "grasp": 3000,
                "place": 8000,  # Hardest segment
            }.get(sg.name, 5000)
            
            print(f"Planning RRT for {sg.name} (max_iters={max_iters_for_segment})...")
            path = plan_rrt_connect(
                scene, q_current, q_target, rng, 
                max_iters=max_iters_for_segment, 
                exclude_block=exclude_block,
                weld_transform=weld_transform_planning
            )
            if path is None:
                print(f"RRT failed for subgoal {sg.name} after {max_iters_for_segment} iterations")
                print(f"Reference notebook uses separate scenario with letter welded to gripper")
                # Fall back to direct interpolation
                path = [q_current, q_target]
            else:
                print(f"RRT succeeded for {sg.name}: {len(path)} waypoints")
            
            waypoints.extend(path)
            q_current = q_target
        
        # Restore initial scene state after planning
        plant.SetPositions(plant_context, q_initial_scene)
        print(f"RRT planning complete, {len(waypoints)} waypoints generated")

    images_front, images_wrist, states, actions, contacts = [], [], [], [], []

    def render_rgb(port):
        sys = port.get_system()
        port_context = sys.GetMyContextFromRoot(root_context)
        img = port.Eval(port_context).data
        # Copy only RGB channels and convert to uint8 immediately to save memory
        return np.ascontiguousarray(img[:,:,:3], dtype=np.uint8)

    eeF = _ee_frame(plant)
    J_gain = 0.5
    max_step = 0.02
    goal_idx = 0
    gripper_closed = False
    gripper_close_time = None  # Track when gripper closed
    block_attached = False
    block_released = False
    X_GB = None  # Relative transform from gripper to block (computed when grasping)
    
    # Initialize subgoal indices (used for gripper control)
    grasp_subgoal_idx = next((i for i, sg in enumerate(subgoals) if sg.name == "grasp"), None)
    place_subgoal_idx = next((i for i, sg in enumerate(subgoals) if sg.name == "place"), None)
    
    # If using RRT waypoints, interpolate them over time
    if waypoints is not None and len(waypoints) > 0:
        t_waypoints = np.linspace(0, steps - 1, len(waypoints))
        waypoints_array = np.array(waypoints)

    # Start Meshcat recording for HTML playback (if meshcat is available)
    if scene.meshcat is not None:
        scene.meshcat.StartRecording()
        scene.meshcat.SetProperty("/Background", "visible", False)  # Clean background for recording

    for t in range(steps):
        plant_context = plant.GetMyMutableContextFromRoot(root_context)
        
        # Check block position only periodically to reduce output
        if t == 0:
            block_body = plant.GetBodyByName(manipuland_body_name)
            X_WB_debug = plant.EvalBodyPoseInWorld(plant_context, block_body)
            print(f"[t={t}] Starting trajectory, block at: {X_WB_debug.translation()}")
        
        # If using RRT waypoints, interpolate between them
        if waypoints is not None and len(waypoints) > 0:
            # SMOOTH trajectory following with lookahead
            # Instead of tracking nearest waypoint, look ahead for smoother motion
            
            # Find lookahead point (3-5 timesteps ahead for smooth anticipation)
            lookahead_steps = min(5, len(t_waypoints) - 1)
            t_lookahead = min(t + lookahead_steps, t_waypoints[-1])
            
            # Interpolate ONLY robot joint positions (first 9 DOF: arm + gripper)
            q_target = plant.GetPositions(plant_context).copy()
            for i in range(9):  # Only arm + gripper joints
                q_target[i] = np.interp(t_lookahead, t_waypoints, waypoints_array[:, i])
            
            # SMOOTH PD control with heavy damping to reduce jitter
            q_current = plant.GetPositions(plant_context)
            v_current = plant.GetVelocities(plant_context)
            
            q_err = np.zeros(nq)
            q_err[:9] = q_target[:9] - q_current[:9]
            
            # Use LOWER gains with MORE damping for smooth motion
            error_norm = np.linalg.norm(q_err[:9])
            if error_norm < 0.02:  # Very close - minimal motion
                gain_p = 0.5
                gain_d = 0.5
            elif error_norm < 0.10:  # Close - gentle approach
                gain_p = 1.5
                gain_d = 0.4
            else:  # Far - moderate speed with damping
                gain_p = 3.0
                gain_d = 0.3
            
            # PD control with strong damping for smooth motion
            v_desired = gain_p * q_err - gain_d * plant.MapVelocityToQDot(plant_context, v_current)
            
            # Convert to velocity-space
            v = plant.MapQDotToVelocity(plant_context, v_desired)
            
            # Gentler velocity limits for smooth motion
            max_vel = max_step / dt * 0.8  # 80% max speed for smoothness
            v = np.clip(v, -max_vel, max_vel)
        else:
            # Original Jacobian-based controller
            target = subgoals[min(goal_idx, len(subgoals)-1)]
            R_WG = RotationMatrix(np.asarray(target.pose_world[:3, :3], dtype=np.float64))
            p_WG = np.asarray(target.pose_world[:3, 3], dtype=np.float64)
            T_WG = RigidTransform(R_WG, p_WG.tolist())
            X_WE = plant.CalcRelativeTransform(plant_context, plant.world_frame(), eeF)
            p_err = T_WG.translation() - X_WE.translation()
            dist = np.linalg.norm(p_err)

            J = plant.CalcJacobianSpatialVelocity(
                plant_context,
                JacobianWrtVariable.kV,
                eeF,
                [0, 0, 0],
                plant.world_frame(),
                plant.world_frame(),
            )
            Jv = J[:3, :nv]
            dv = J_gain * p_err
            qdot, *_ = np.linalg.lstsq(Jv, dv, rcond=1e-3)
            qdot = np.clip(qdot, -max_step / dt, max_step / dt)
            v = qdot
        
        # Check if we should close gripper (when AT grasp position)
        # Following reference notebook: gripper closes AFTER reaching grasp pose
        # Only grasp once - prevent re-grasping after release
        if waypoints is not None and grasp_subgoal_idx is not None and not gripper_closed and not block_released:
            # Check if we're close to grasp subgoal
            X_WE = plant.CalcRelativeTransform(plant_context, plant.world_frame(), eeF)
            grasp_pos = subgoals[grasp_subgoal_idx].pose_world[:3, 3]
            dist_to_grasp = np.linalg.norm(X_WE.translation() - grasp_pos)
            
            # print distance periodically (every 4 steps = 1 seconds)
            if t % 4 == 0:
                X_WB = plant.EvalBodyPoseInWorld(plant_context, block_body)
                block_pos_current = X_WB.translation()
                print(f"[t={t}] Approaching grasp: dist={dist_to_grasp:.3f}m")
            
            # Close gripper when arm is near grasp position
            # RRT paths are naturally jittery, so we use distance-based triggering
            # Allow gripper to close when within 1.5cm OR after reasonable time near target
            arm_velocity = np.linalg.norm(v[:7])  # Check arm joint velocities
            
            # Track how long we've been near the grasp
            if not hasattr(plan_and_rollout, '_near_grasp_start'):
                plan_and_rollout._near_grasp_start = None
            
            if dist_to_grasp < 0.02:  # Within 2cm
                if plan_and_rollout._near_grasp_start is None:
                    plan_and_rollout._near_grasp_start = t
                
                # Close if: very close (1cm) OR stayed near grasp for 1 sec (20 timesteps)
                time_near_grasp = t - plan_and_rollout._near_grasp_start
                if dist_to_grasp < 0.01 or time_near_grasp >= 20:
                    # Start closing gripper
                    gripper_closed = True
                    gripper_close_time = t
                    plan_and_rollout._near_grasp_start = None  # Reset
                    print(f"Arm near grasp pose (dist={dist_to_grasp:.4f}m, vel={arm_velocity:.4f})")
                    print(f"Closing gripper at t={t}")
            else:
                plan_and_rollout._near_grasp_start = None  # Reset if we move away
        
        # Gripper force control will be applied AFTER arm control velocities are computed
        # This ensures gripper closing force is not overridden by the PD controller
        
        # After gripper closes, wait for physics to settle, then create weld constraint
        # Wait at least 5 timesteps (0.25 sec) after gripper closes for contact to establish
        if gripper_closed and not block_attached and not block_released and gripper_close_time is not None:
            timesteps_since_close = t - gripper_close_time
            
            if timesteps_since_close >= 5:  # Wait 0.25 seconds for physics to settle
                block_body = plant.GetBodyByName(manipuland_body_name)
                gripper_body_name = _get_gripper_body_name(plant)
                gripper_body = plant.GetBodyByName(gripper_body_name)
                
                X_WG = plant.EvalBodyPoseInWorld(plant_context, gripper_body)
                X_WB = plant.EvalBodyPoseInWorld(plant_context, block_body)
                
                # Check if gripper has actually grasped the block
                dist_gripper_to_block = np.linalg.norm(X_WG.translation() - X_WB.translation())
                block_z = X_WB.translation()[2]
                
                # Check for ACTUAL contact between gripper and block using physics
                sg_context = scene.scene_graph.GetMyContextFromRoot(root_context)
                query = scene.scene_graph.get_query_output_port().Eval(sg_context)
                pairs = query.ComputePointPairPenetration()
                
                # Check if there's contact between gripper fingers and block
                inspector = scene.scene_graph.model_inspector()
                gripper_frame_id = plant.GetBodyFrameIdOrThrow(gripper_body.index())
                block_frame_id = plant.GetBodyFrameIdOrThrow(block_body.index())
                gripper_geom_ids = set(inspector.GetGeometries(gripper_frame_id))
                block_geom_ids = set(inspector.GetGeometries(block_frame_id))
                
                has_gripper_block_contact = False
                for p in pairs:
                    geom_a = p.id_A
                    geom_b = p.id_B
                    # Check if one geometry is from gripper and other is from block
                    if ((geom_a in gripper_geom_ids and geom_b in block_geom_ids) or
                        (geom_b in gripper_geom_ids and geom_a in block_geom_ids)):
                        if p.depth > 0.0:  # Actual penetration/contact
                            has_gripper_block_contact = True
                            break
                
                # Get gripper finger positions
                q_current = plant.GetPositions(plant_context)
                gripper_left = q_current[7]
                gripper_right = q_current[8]
                
                # HYBRID GRASP: Check if block is within gripper finger region
                # Transform block position to gripper frame to check if it's between fingers
                X_GB_current = X_WG.inverse() @ X_WB
                block_in_gripper_frame = X_GB_current.translation()
                
                # Gripper geometry (Panda gripper):
                # Fingers extend in +/- Y direction from gripper center
                # Finger width: ~4cm each (0.04m), so total span when open: ~8cm
                # Grasp region: block must be within finger span in Y, close in X and Z
                
                # Check if block is in valid grasp region (between fingers)
                # Use block size to determine valid region tolerances
                from drake_env.scenes import BLOCK_SIZE, TABLE_SURFACE_Z, WELD_OFFSET_Z
                
                # X, Y: Block must be well-centered with gripper for reliable grasp
                # Use tight tolerance: within 0.5x block size (3.35cm) for X,Y alignment
                # Z: Block should be at the target weld offset depth
                translation_raw = block_in_gripper_frame
                tolerance_xy = BLOCK_SIZE * 0.5  # Tight centering: ±3.35cm
                tolerance_z_min = WELD_OFFSET_Z * 0.85  # 85-115% of target depth
                tolerance_z_max = WELD_OFFSET_Z * 1.15
                
                in_grasp_region = (
                    abs(translation_raw[0]) < tolerance_xy and
                    abs(translation_raw[1]) < tolerance_xy and
                    tolerance_z_min < translation_raw[2] < tolerance_z_max  # Block at correct depth below gripper
                )
                
                # Distance check: gripper very close to block (within 1x block size)
                # Height check: block should be above table surface
                grasp_geometry_valid = (
                    dist_gripper_to_block < BLOCK_SIZE * 1.2 and
                    block_z > TABLE_SURFACE_Z * 0.9 and  # Above table (with margin for block on table)
                    in_grasp_region and
                    abs(translation_raw[2] - WELD_OFFSET_Z) < BLOCK_SIZE * 0.3  # Block at target grasp depth ±2cm
                )
                
                # Compute SAFE weld offset to prevent penetration
                # Use WELD_OFFSET_Z constant for consistency
                if has_gripper_block_contact and grasp_geometry_valid:
                    # Create ideal weld transform with block positioned safely
                    # In gripper frame: X forward, Y left/right, Z down (gripper pointing down)
                    # Block center positioned at WELD_OFFSET_Z below gripper origin
                    # This matches the grasp height computation in compute_subgoals
                    from pydrake.math import RollPitchYaw
                    
                    # Use IDEAL weld offset for proper grasp geometry
                    # Only weld when block is properly positioned in grasp pocket
                    safe_offset = np.array([0.0, 0.0, WELD_OFFSET_Z])  # Use target grasp offset
                    R_GB = X_GB_current.rotation()  # Keep block orientation from contact
                    X_GB = RigidTransform(R_GB, safe_offset)
                    
                    block_attached = True
                    
                    block_pos_before = X_WB.translation()
                    gripper_pos = X_WG.translation()
                    print(f"Block grasped at t={t}")
                    print(f"Distance: {dist_gripper_to_block:.4f}m")
                    print(f"Actual weld offset: [{safe_offset[0]:.3f}, {safe_offset[1]:.3f}, {safe_offset[2]:.3f}]")
                    print(f"Block before: [{block_pos_before[0]:.3f}, {block_pos_before[1]:.3f}, {block_pos_before[2]:.3f}]")
                    print(f"Gripper at: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
                elif timesteps_since_close > 50:
                    # If we've waited too long (2.5 sec) and still no valid grasp, give up
                    print(f"Failed after {timesteps_since_close} steps")
                    print(f"Contact: {has_gripper_block_contact}")
                    print(f"In grasp region: {in_grasp_region}")
                    print(f"Distance gripper-block: {dist_gripper_to_block:.3f}m (threshold: {BLOCK_SIZE * 1.2:.3f}m)")
                    print(f"Block in gripper frame: [{translation_raw[0]:.3f}, {translation_raw[1]:.3f}, {translation_raw[2]:.3f}]")
                    print(f"XY tolerance: {tolerance_xy:.3f}m, Z range: [{tolerance_z_min:.3f}, {tolerance_z_max:.3f}]")
                    print(f"Geometry valid: {grasp_geometry_valid}")
                    gripper_closed = False  # Reset to try again
        
        # Check if we should release block (when near place position)
        if waypoints is not None and place_subgoal_idx is not None and block_attached and not block_released:
            X_WE = plant.CalcRelativeTransform(plant_context, plant.world_frame(), eeF)
            place_pos = subgoals[place_subgoal_idx].pose_world[:3, 3]
            dist_to_place = np.linalg.norm(X_WE.translation() - place_pos)
            
            # Release when within reasonable distance of place position
            # Use block size as threshold (allows some tolerance)
            from drake_env.scenes import BLOCK_SIZE
            if dist_to_place < BLOCK_SIZE * 1.5:  # Within 1.5x block size
                # Open gripper to release block
                q = plant.GetPositions(plant_context).copy()
                q[7:9] = [0.04, 0.04]  # Open gripper fingers
                plant.SetPositions(plant_context, q)
                block_released = True
                block_attached = False  # Detach weld
                print(f"Opening gripper at t={t}, distance={dist_to_place:.4f}m")
                print(f"Block released - now subject to physics only")
        
        # Apply velocities only to robot joints; leave block DOFs to simulator physics
        v_full = plant.GetVelocities(plant_context).copy()
        v_full[:9] = v[:9]
        
        # Intelligent gripper control to prevent over-closing and penetration
        # Only apply closing force if fingers haven't reached minimum safe width
        if gripper_closed and not block_released:
            q_current = plant.GetPositions(plant_context)
            gripper_left = q_current[7]
            gripper_right = q_current[8]
            min_finger_width = 0.005  # Minimum 5mm gap - prevents over-closing/penetration
            
            # Check if we have contact with block (stop applying force if we do)
            sg_context = scene.scene_graph.GetMyContextFromRoot(root_context)
            query = scene.scene_graph.get_query_output_port().Eval(sg_context)
            pairs = query.ComputePointPairPenetration()
            
            # Check for finger-block contact
            has_finger_contact = False
            if not block_attached:  # Only check before weld is created
                block_body = plant.GetBodyByName(manipuland_body_name)
                inspector = scene.scene_graph.model_inspector()
                block_frame_id = plant.GetBodyFrameIdOrThrow(block_body.index())
                block_geom_ids = set(inspector.GetGeometries(block_frame_id))
                
                for p in pairs:
                    if p.id_A in block_geom_ids or p.id_B in block_geom_ids:
                        geom_a_name = inspector.GetName(p.id_A)
                        geom_b_name = inspector.GetName(p.id_B)
                        # Check if contact is with fingers (not palm)
                        if 'finger' in geom_a_name.lower() or 'finger' in geom_b_name.lower():
                            has_finger_contact = True
                            break
            
            # Gripper closing strategy for weld-based grasping:
            # 1. Before attachment: close gently until contact is made
            # 2. After attachment: gripper is welded, no more closing needed
            # 3. Stop at minimum width to prevent over-closing
            
            if (gripper_left > min_finger_width or gripper_right > min_finger_width) and not block_attached:
                if not has_finger_contact:
                    # APPROACHING: Close gently until we make contact
                    v_full[7] = -0.5  # Moderate closing velocity
                    v_full[8] = -0.5  # Moderate closing velocity
                # else: has_finger_contact but not attached yet - wait for weld to engage
        
        plant.SetVelocities(plant_context, v_full)

        qdot_conf = plant.MapVelocityToQDot(plant_context, v_full)
        q = plant.GetPositions(plant_context).copy()
        q[:9] = q[:9] + qdot_conf[:9] * dt
        plant.SetPositions(plant_context, q)
        
        # KINEMATIC WELD CONSTRAINT - Standard approach for pick-and-place
        # Once grasped, the block is rigidly attached to the gripper
        # This is the same approach used in Drake manipulation tutorials
        # Advantages: stable, no slipping, computationally efficient
        if block_attached:
            gripper_body_name = _get_gripper_body_name(plant)
            gripper_body = plant.GetBodyByName(gripper_body_name)
            block_body = plant.GetBodyByName(manipuland_body_name)
            
            # Get current gripper pose
            X_WG = plant.EvalBodyPoseInWorld(plant_context, gripper_body)
            
            # Compute desired block pose: X_WB = X_WG * X_GB
            # X_GB is the block's pose relative to gripper (set when grasp detected)
            X_WB_desired = X_WG @ X_GB
            
            # Update block position in q to maintain grasp
            # Block state is at indices 9-15: [qw, qx, qy, qz, x, y, z]
            quat = X_WB_desired.rotation().ToQuaternion().wxyz()
            pos = X_WB_desired.translation()
            q[9:13] = quat  # Quaternion
            q[13:16] = pos  # Position
            
            # Zero out block velocity (it moves with gripper)
            v_updated = plant.MapQDotToVelocity(plant_context, plant.MapVelocityToQDot(plant_context, v))
            v_updated[9:15] = 0.0  # Zero block velocities
            plant.SetVelocities(plant_context, v_updated)
            
            # Reduced debug output (only on first weld to confirm it's working)
            if t == gripper_close_time + 10:
                print(f"[t={t}] Block attached - Gripper Z: {X_WG.translation()[2]:.3f}, Block Z: {pos[2]:.3f}")
        
        plant.SetPositions(plant_context, q)
        # Advance the simulator - use fewer substeps to reduce memory usage
        # With 0.002s plant timestep, we get 5 substeps per 0.01s control step
        # This is sufficient for stable contact simulation
        physics_dt = 0.01  # 100Hz control rate (matches 0.002s plant timestep × 5)
        steps_per_control = int(dt / physics_dt)
        for _ in range(steps_per_control):
            sim.AdvanceTo(sim.get_context().get_time() + physics_dt)

        # Now sample the rendered images and states from the updated context.
        # Only render images if camera ports exist (scene C has cameras, scene D doesn't)
        if "rgb_front" in scene.ports:
            images_front.append(render_rgb(scene.ports["rgb_front"]))
        if "rgb_wrist" in scene.ports:
            images_wrist.append(render_rgb(scene.ports["rgb_wrist"]))
        states.append({"q": q.copy(), "v": v.copy()})
        actions.append({"qdot": v.copy()})
        contacts.append({"min_distance": _min_signed_distance(scene, root_context)})

        # Update goal index for non-RRT mode
        if waypoints is None:
            if dist < 0.01 and goal_idx < len(subgoals)-1:
                goal_idx += 1

    # Stop Meshcat recording after simulation completes
    if scene.meshcat is not None:
        scene.meshcat.StopRecording()
        scene.meshcat.PublishRecording()
        print(f"Meshcat recording saved (frames: {steps})")

    return {
        "images_front": images_front,
        "images_wrist": images_wrist,
        "states": states,
        "actions": actions,
        "contacts": contacts,
        "language": "place the red block in the green bin",
        "subgoals": [
            {
                "name": sg.name,
                "pose_world": sg.pose_world.tolist(),
            }
            for sg in subgoals
        ],
    }
