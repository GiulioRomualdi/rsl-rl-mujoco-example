"""
Termination conditions for the musculoskeletal simulation environment.
Determines if the episode should end based on conditions such as falling,
COM Y deviation, excessive pelvis/torso angle deviations and high contact forces.
@author: YAKE
"""

import numpy as np
import logging
from functools import partial
from typing import Any, Tuple, Dict, List

from .state import get_GRF_info, get_joint_kinematics
from .common_utils import inverse_convert_ref_traj_qpos, compute_site_kinematics

# Configure module-level logger
logger = logging.getLogger(__name__)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def check_termination(env: Any, conditions: List[str] = None) -> Tuple[bool, Dict[str, bool]]:
    """
    Check if the episode should be terminated early based on selected conditions.

    Args:
        env: The environment instance.
        conditions (List[str], optional): List of condition names to check. 
            If None, defaults to all available conditions.

    Returns:
        Tuple[bool, Dict[str, bool], float]: 
            - terminated: True if any termination condition is met.
            - termination_conditions: Dictionary with each condition's boolean result.
    """
    available_conditions: Dict[str, Any] = {
        "has_fallen": has_fallen,
        "comY_deviated": is_comY_deviated,
        "pelvis_angle_exceeded": partial(is_angle_exceeded, body="pelvis"),
        "torso_angle_exceeded": partial(is_angle_exceeded, body="torso"),
        "contact_force_exceeded": is_contact_force_exceeded,
        "site_deviation_exceeded": is_site_deviation_exceeded
    }
    
    if conditions is None:
        conditions = list(available_conditions.keys())
        
    invalid_conditions = set(conditions) - set(available_conditions.keys())
    if invalid_conditions:
        raise ValueError(f"[ERROR] Invalid termination conditions: {invalid_conditions}")

    termination_conditions: Dict[str, bool] = {}
    for cond in conditions:
        try:
            result = available_conditions[cond](env)
            if isinstance(result, tuple):
                flag = bool(result[0])
            else:
                flag = bool(result)
            termination_conditions[cond] = flag
        except Exception as e:
            logger.error("Error evaluating termination condition '%s': %s", cond, e)
            termination_conditions[cond] = True

    terminated = any(termination_conditions.values())

    return terminated, termination_conditions

def has_fallen(env: Any,
               pelvis_height_range: Tuple[float, float] = (0.6, 1.2)) -> bool:
    """
    Check if the character has fallen by verifying that the pelvis height.

    Args:
        env: The simulation environment that contains the model and data.
        pelvis_height_range (Tuple[float, float]): Allowed pelvis height range as (min, max). 

    Returns:
        bool: True if the pelvis height is not in the specificed range, False otherwise.
    """
    try:
        pelvis_id = env.model.body("pelvis").id
        if pelvis_id is None:
            raise ValueError("Pelvis body not found in the model.")
        
        # Retrieve the pelvis height (z coordinate) from simulation data.
        pelvis_height = env.data.xpos[pelvis_id, 2].copy()
        
        # Check if the pelvis height is below the allowed minimum.
        if pelvis_height < pelvis_height_range[0] or pelvis_height > pelvis_height_range[1]:
            logger.debug("has_fallen: pelvis height %.3f out of range %s", pelvis_height, pelvis_height_range)
            return True
        return False

    except Exception as e:
        # Log the error and conservatively assume fallen.
        logger.error("has_fallen check failed: %s", e)
        return True

def is_comY_deviated(env: Any, com_y_threshold: float = 0.2) -> bool:
    """
    Check if the center-of-mass (COM) deviation in the Y direction exceeds a specified threshold.

    Args:
        env: The simulation environment, which must include:
             - env.init_COMY: The initial COM Y coordinate.
             - env.data.subtree_com: A numpy array containing COM information.
        com_y_threshold (float): The allowable deviation threshold along the Y axis.

    Returns:
        bool: True if the COM Y deviation exceeds the threshold; otherwise, False.
    """
    try:
        # Retrieve the initial COM Y value from the environment.
        init_COMY = getattr(env, "init_COMY", None)
        if init_COMY is None:
            raise ValueError("Initial COM Y value (env.init_COMY) is not set.")
            
        # Access the subtree center-of-mass array and validate its shape.
        subtree_com = env.data.subtree_com
        current_comY = subtree_com[1, 1]

        deviation = abs(current_comY - init_COMY)
        return deviation > com_y_threshold
    
    except Exception as e:
        logger.error("COM Y deviation check failed: %s", e)
        # In case of error, return True to conservatively indicate a potential deviation.
        return True

def is_angle_exceeded(env: Any, 
                      body: str = 'pelvis', 
                      angle_thresholds: Tuple[float, float, float] = (0.35, 0.35, 0.8)) -> bool:
    """
    Check if the specified body's orientation angles deviate from the reference beyond allowed thresholds.

    Args:
        env: The simulation environment instance containing simulation state and reference trajectory.
        body (str): Body part to check ('pelvis' or 'torso').
        angle_thresholds (Tuple[float, float, float]): Thresholds (in radians) for deviation 
            along each axis (e.g., tilt, list, rotation).

    Returns:
        bool: True if the absolute deviation for any angle exceeds its corresponding threshold; False otherwise.
    """
    try:
        valid_bodies = {'pelvis': (3, 6), 'torso': (20, 23)}
        if body not in valid_bodies:
            raise ValueError(f"[ERROR] Invalid body name: {body}. Expected 'pelvis' or 'torso'.")
            
        start, end = valid_bodies[body]
        
        # Convert the simulation's generalized positions into a reference-compatible representation.
        sim_qpos = env.data.qpos.copy()
        sim_qpos = inverse_convert_ref_traj_qpos(sim_qpos)
        
        # Validate that the simulation qpos array has enough elements.
        if sim_qpos.shape[0] < end:
            raise ValueError(f"Expected sim_qpos to have at least {end} elements, got {sim_qpos.shape[0]}")
            
        # Extract the orientation angles for the specified body.
        sim_angles = sim_qpos[start:end]
        
        # Retrieve reference orientation angles.
        if body == 'pelvis':
            ref_angles = env.ref_traj.get_pelvis_ang()[3:6]
        else:
            ref_angles = env.ref_traj.get_torso_ang()
            
        angle_diff = np.arctan2(np.sin(sim_angles - ref_angles), np.cos(sim_angles - ref_angles))
        abs_diff = np.abs(angle_diff)
        
        exceeded = np.any(abs_diff > np.array(angle_thresholds))
        return exceeded
    
    except Exception as e:
        logger.error("Angle check failed for %s: %s", body, e)
        # In case of any error, return True to conservatively indicate the angle might be exceeded.
        return True  
    
def is_contact_force_exceeded(env: Any, max_coef: float = 5.0) -> bool:
    """
    Check if the maximum contact force from the feet exceeds a specified threshold.

    Args:
        env: The simulation environment instance containing the model and simulation data.
        max_coef (float): Coefficient used to scale the total mass and gravity to compute the force threshold.
    
    Returns:
        bool: True if the maximum contact force exceeds the threshold; False otherwise.
    """
    try:
        # Retrieve the total mass; if env has no attribute 'total_mass', compute it as the sum of body masses.
        total_mass = getattr(env, "total_mass", np.sum(env.model.body_mass))
        
        # Get the ground reaction force information.
        _, GRF_info = get_GRF_info(env)
        
        # Safely extract the GRF force vectors for the right and left feet.
        force_right = np.linalg.norm(GRF_info.get("right", {}).get('GRF_force', np.zeros(3)))
        force_left  = np.linalg.norm(GRF_info.get("left",  {}).get('GRF_force', np.zeros(3)))
        
        # Determine the maximum force observed between the two feet.
        max_force = max(force_right, force_left)

        # Calculate the force threshold using the mass, gravity, and max_coef; ensure a minimal threshold of 1000.
        threshold = max(max_coef * total_mass * 9.81, 1e3)
        
        return max_force > threshold

    except Exception as e:
        logger.error("Failed to check contact force exceeded: %s", e)
        # In case of any error, conservatively indicate that the contact force is exceeded.
        return True
    
def is_site_deviation_exceeded(env: Any, threshold: float = 0.25, relative_to_pelvis: bool = False) -> Tuple[bool, Dict[str, float]]:
    """
    Check if any sensor site's position deviates from its reference position by more than a specified threshold.

    Args:
        env (Any): The simulation environment. Must provide:
            - env.ref_traj: reference trajectory object with get_qpos() and get_qvel()
            - env.model: a MuJoCo model object
            - env.data: simulation data including joint‐space sensor positions
        threshold (float): Distance threshold in meters above which a deviation is considered excessive.
        relative_to_pelvis: If True, subtract the pelvis translation and velocity.

    Returns:
        exceeded (bool): True if at least one site exceeds the threshold, False otherwise.
        details  (dict): Mapping from each over‐threshold site name to its actual deviation distance.
    """
    exceeded_sites: Dict[str, float] = {}
    
    if not hasattr(env, 'ref_traj') or not hasattr(env, 'model') or not hasattr(env.data, 'qpos'):
        logger.error("Environment is missing required attributes for site deviation check.")
        return True, {}
    
    # Retrieve current sensor positions from the simulation
    try:
        _, sim_kin = get_joint_kinematics(env, include_pelvis=True, relative_to_pelvis=relative_to_pelvis)
        sim_sites = sim_kin.get("joint_space_pos", {})
        if not sim_sites:
            return False, {}
    except Exception as e:
        logger.error("Failed to get simulated site positions: %s", e)
        return True, {}
    
    # Compute reference sensor positions using the reference trajectory state
    try:
        ref_qpos = env.ref_traj.get_qpos()
        ref_qvel = env.ref_traj.get_qvel()
        ref_kin  = compute_site_kinematics(ref_qpos, ref_qvel, env.model, relative_to_pelvis=relative_to_pelvis)
    except Exception as e:
        logger.error("Failed to compute reference site positions: %s", e)
        return True, {}

    for site, sim_pos in sim_sites.items():
        if site not in ref_kin:
            continue
        ref_pos = ref_kin[site].get("pos", None)
        if ref_pos is None:
            continue
        distance = np.linalg.norm(sim_pos - ref_pos)
        if distance > threshold:
            exceeded_sites[site] = distance
    
    if not exceeded_sites:
        return False, {}
    return True, exceeded_sites
        
        
        