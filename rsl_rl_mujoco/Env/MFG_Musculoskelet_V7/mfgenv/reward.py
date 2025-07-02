"""
Reward functions for musculoskeletal simulation environment.
Provides imitation reward components and (currently stubbed) goal reward.
@author: YAKE
"""

import numpy as np
from .common_utils import compute_site_kinematics, inverse_convert_ref_traj_qpos, inverse_convert_ref_traj_qvel, get_penalty
from .state import get_joint_kinematics, get_COM_kinematics, get_GRF_info
import logging
from typing import Any, Tuple, Dict, Optional

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def compute_reward(env: Any) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the total reward for the current simulation step as a weighted sum of three key components:
      - Imitation reward: Measures how well the simulated motion tracks the desired trajectories,
        including site (position and velocity) and joint (angle and angular velocity) tracking errors.
      - Goal reward: Designed for task-specific objectives; here it is implemented as a COM speed reward,
        encouraging the agent to maintain a steady forward speed.
      - Smooth reward: Encourages smooth motion by penalizing abrupt changes or excessive forces;
        it is computed as a weighted sum of GRF, torque, and action smoothness rewards.
    
    The weights for each component should be provided in env.reward_weights as a dictionary with the keys:
      "imitation", "goal", and "smooth".
      
    Returns:
        Tuple[float, Dict[str, Any]]:
            - total_reward (float): The overall reward computed for the current step.
            - breakdown (dict): A dictionary providing a detailed breakdown of each reward component,
              with keys "imitation", "goal", and "smooth".
    """
    try:
        # Validate that env.reward_weights exists and contains required keys.
        if not hasattr(env, "reward_weights") or not isinstance(env.reward_weights, dict):
            raise ValueError("env.reward_weights is missing or invalid. It must be a dict with keys: 'imitation', 'goal', 'smooth'.")
        
        required_keys = {"imitation", "goal", "smooth"}
        missing_keys = required_keys - set(env.reward_weights.keys())
        if missing_keys:
            raise ValueError(f"Missing reward weights for keys: {missing_keys}")
        
        # Compute individual reward components.
        imitation_reward, imitation_breakdown = get_imitation_reward(env)
        goal_reward, goal_breakdown = get_goal_reward(env)
        smooth_reward, smooth_breakdown = get_smooth_reward(env)
        
        # Compute the total reward as a weighted sum.
        total_reward = (env.reward_weights["imitation"] * imitation_reward +
                        env.reward_weights["goal"] * goal_reward +
                        env.reward_weights["smooth"] * smooth_reward)
        
        # Consolidate the breakdown details.
        breakdown = {
            "imitation": imitation_breakdown,
            "goal": goal_breakdown,
            "smooth": smooth_breakdown
        }
        
        return total_reward, breakdown
    
    except Exception as e:
        logger.error(f"Failed to compute total reward: {e}")
        return 0.0, {}

def get_imitation_reward(env: Any) -> Tuple[float, Dict[str, float]]:
    """
    Compute the imitation reward as a weighted sum of individual components.
    
    The imitation reward is composed of:
        - Site tracking reward:
            * site_pos: based on the global position error of sensor sites.
            * site_vel: based on the linear velocity error of sensor sites.
        - Joint tracking reward:
            * joint_angle: based on the joint angle tracking error (using periodic error calculation).
            * joint_angvel: based on the joint angular velocity tracking error.

    Args:
        env: The environment instance.

    Returns:
        Tuple[float, Dict[str, float]]:
            - total imitation reward (float)
            - a breakdown dictionary with keys "site_pos", "site_vel", "joint_angle", "joint_angvel"
    """
    try:
        # Validate that imitation_weights exists and contains the required keys.
        if not hasattr(env, "imitation_weights") or not isinstance(env.imitation_weights, dict):
            raise ValueError("imitation_weights is missing or invalid. It must be a dict with keys: 'site_pos', 'site_vel', 'joint_angle', 'joint_angvel'.")
    
        required_keys = {"site_pos", "site_vel", "joint_angle", "joint_angvel"}
        missing_keys = required_keys - set(env.imitation_weights.keys())
        if missing_keys:
            raise ValueError(f"Missing imitation reward weights for keys: {missing_keys}")
        
        # Compute individual imitation reward components.
        site_pos_reward, site_vel_reward, _ = get_site_tracking_reward(env)
        joint_angle_reward, joint_angvel_reward, _ = get_joint_tracking_reward(env)
    
        # Calculate the weighted sum of reward components. Direct multiplications are used for efficiency.
        total_reward = (env.imitation_weights["site_pos"] * site_pos_reward +
                        env.imitation_weights["site_vel"] * site_vel_reward +
                        env.imitation_weights["joint_angle"] * joint_angle_reward +
                        env.imitation_weights["joint_angvel"] * joint_angvel_reward)
    
        # Build a detailed breakdown dictionary.
        breakdown = {
            "site_pos": site_pos_reward,
            "site_vel": site_vel_reward,
            "joint_angle": joint_angle_reward,
            "joint_angvel": joint_angvel_reward
            }
    
        return total_reward, breakdown

    except Exception as e:
        # Log the error to help with debugging while ensuring the function remains robust.
        logger.error("Failed to compute imitation reward: %s", e)
        return 0.0, {}

def get_goal_reward(env: Any) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the goal reward based on task-specific objectives.
    Currently not implemented.

    Args:
        env: The environment instance.

    Returns:
      Tuple[float, Dict[str, Any]]:
         - goal_reward (float): The overall goal reward.
         - breakdown (dict): Detailed breakdown of the COM speed reward.
    """
    try:
        com_speed_reward, com_breakdown = get_com_speed_reward(env)
        goal_reward = com_speed_reward
        breakdown = {
            "com_speed_reward": com_speed_reward,}
        
        return goal_reward, breakdown
        
    except Exception as e:
        logger.error("Failed to compute goal reward: %s", e)
        return 0.0, {}

def get_smooth_reward(env: Any) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the overall smooth reward as a weighted sum of three components that encourage smooth motion:
      - GRF reward: Penalizes excessive ground reaction forces.
      - Torque reward: Penalizes excessive joint actuator forces.
      - Action reward: Penalizes abrupt changes in the RL-generated actions.

    Returns:
        Tuple[float, Dict[str, Any]]:
          - The overall smooth reward as a float.
          - A breakdown dictionary containing the individual breakdowns for each component.
    """
    try:
        # Validate that smooth_weights is provided and contains the required keys.
        if not hasattr(env, "smooth_weights") or not isinstance(env.smooth_weights, dict):
            raise ValueError("smooth_weights is missing or invalid. It must be a dictionary with keys: 'grf', 'torque', 'action'.")
        
        required_keys = {"grf", "torque", "action"}
        missing_keys = required_keys - set(env.smooth_weights.keys())
        if missing_keys:
            raise ValueError(f"Missing smooth reward weights for: {missing_keys}")
        
        # Compute individual smooth reward components.
        grf_reward, grf_breakdown = get_grf_reward(env)
        torque_reward, torque_breakdown = get_torque_reward(env)
        action_reward, action_breakdown = get_action_reward(env)
        
        # Compute the overall smooth reward as the weighted sum.
        total_smooth_reward = (
            env.smooth_weights["grf"] * grf_reward +
            env.smooth_weights["torque"] * torque_reward +
            env.smooth_weights["action"] * action_reward
        )
        
        # Build the breakdown dictionary.
        breakdown = {
            "grf": grf_breakdown,
            "torque": torque_breakdown,
            "action": action_breakdown,
        }
        
        return total_smooth_reward, breakdown
    
    except Exception as e:
        logger.error(f"Failed to compute smooth reward: {e}")
        return 0.0, {}

def get_site_tracking_reward(env: Any,
                             relative_to_pelvis: bool = False,
                             site_weights: Optional[Dict[str, float]] = None,
                             k_pos: float = 20.0,
                             k_vel: float = 0.05) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute the site tracking reward based on the global positions and linear velocities of joint anchor (site) sensors.
    
    The function first retrieves the reference joint state from env.ref_traj and computes reference site kinematics
    using compute_site_kinematics(ref_qpos, ref_qvel, model).
    It then obtains simulation site kinematics (including all sensor sites, e.g. 19 in total) using get_joint_kinematics.
    
    For each sensor site present in the reference data, the function computes the squared Euclidean error for 
    both the global position and linear velocity. Each error is weighted according to site_weights (defaulting to 1.0),
    and the weighted errors are summed and then divided by the total weight. The average errors are then converted into
    rewards using an exponential decay with scaling factors k_pos and k_vel.
    
    Args:
        env (Any): The simulation environment. Expected attributes include:
            - env.ref_traj.get_qpos() and env.ref_traj.get_qvel() for reference states.
            - env.data.qpos and env.data.qvel for current states.
            - env.config: a dictionary that must contain a "model_path" key.
            - env.reward_coefficients: a dict containing keys "site_pos" and "site_vel"
              (if not provided, the default scaling factors k_pos and k_vel are used).
            - (Optional) env.EPSILON: a minimum reward threshold.
        relative_to_pelvis: If True, subtract the pelvis translation and velocity.
        site_weights (Optional[Dict[str, float]]): A dictionary mapping sensor site names to their weights.
            If not provided, a default value of 1.0 is used for every site.
        k_pos (float): Scaling factor for the position error reward.
        k_vel (float): Scaling factor for the velocity error reward.
    
    Returns:
        Tuple[float, float, Dict[str, float]]:
            A tuple containing:
              - site_pos_reward (float): Reward computed from the global position tracking error.
              - site_vel_reward (float): Reward computed from the linear velocity tracking error.
              - breakdown (dict): A dictionary with keys "site_pos" and "site_vel" containing the overall rewards.
    
    Raises:
        ValueError: If required attributes are missing, the model path is not provided, or no sensor sites are found.
    """
    # Validate required attributes.
    if not hasattr(env, 'ref_traj'):
        raise ValueError("Environment must have a 'ref_traj' attribute.")
    if not (hasattr(env, 'data') and hasattr(env.data, 'qpos') and hasattr(env.data, 'qvel')):
        raise ValueError("Environment 'data' must have valid 'qpos' and 'qvel' attributes.")
        
    # Override k_pos and k_vel if provided in env.reward_coefficients.
    reward_coeffs = getattr(env, 'reward_coefficients', {})
    k_pos = reward_coeffs.get("site_pos", k_pos)
    k_vel = reward_coeffs.get("site_vel", k_vel)
    
    # Retrieve reference joint state.
    ref_qpos = env.ref_traj.get_qpos()
    ref_qvel = env.ref_traj.get_qvel()
    
    # Compute reference site kinematics.
    try:
        ref_kin = compute_site_kinematics(ref_qpos, ref_qvel, env.model, relative_to_pelvis=relative_to_pelvis)
    except Exception as e:
        raise ValueError(f"Error computing reference site kinematics: {e}")
    
    # Retrieve simulation site kinematics (including all sensor sites) from the state module.
    try:
        _, sim_kin = get_joint_kinematics(env, include_pelvis=True, relative_to_pelvis=relative_to_pelvis)
    except Exception as e:
        raise ValueError(f"Error obtaining simulation site kinematics via get_joint_kinematics: {e}")

    total_pos_error = 0.0
    total_vel_error = 0.0
    total_weight = 0.0
    site_weights = site_weights or {}
    # Iterate over each sensor site returned in the reference kinematics.
    for site, ref_data in ref_kin.items():
        if site not in sim_kin['joint_space_pos']:
            continue
        
        # Retrieve per-site weight (default 1.0 if not provided)
        w = site_weights.get(site, 1.0)
        
        # Get simulation and reference position and velocity for this sensor.
        sim_pos = sim_kin['joint_space_pos'][site]
        ref_pos = ref_data['pos']
        sim_vel = sim_kin['joint_lin_vel'][site]
        ref_vel = ref_data['vel']
        
        # Compute squared error for positions and velocities.
        pos_err = np.linalg.norm(sim_pos - ref_pos) ** 2
        vel_err = np.linalg.norm(sim_vel - ref_vel) ** 2

        total_pos_error += w * pos_err
        total_vel_error += w * vel_err
        total_weight += w

    if total_weight == 0:
        return 0.0, 0.0, {"site_pos": 0.0, "site_vel": 0.0}

    avg_pos_error = total_pos_error / total_weight
    avg_vel_error = total_vel_error / total_weight

    # Use the scaling factors directly (or override using env.reward_coefficients if needed)
    site_pos_reward = np.exp(-k_pos * avg_pos_error)
    site_vel_reward = np.exp(-k_vel * avg_vel_error)

    if hasattr(env, "EPSILON"):
        if site_pos_reward < env.EPSILON:
            site_pos_reward = 0.0
        if site_vel_reward < env.EPSILON:
            site_vel_reward = 0.0

    breakdown = {"site_pos": site_pos_reward, "site_vel": site_vel_reward,
                 "pos_err": avg_pos_error, "vel_err": avg_vel_error}
    
    return site_pos_reward, site_vel_reward, breakdown

def get_joint_tracking_reward(env: Any,
                              k_pos: float = 20,
                              k_vel: float = 0.05) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute the joint tracking reward for the joint angle and angular velocity in the joint space.
    
    This function compares the simulation state with the reference trajectory state.
    The function returns two separate rewards:
      - joint_angle_reward: reward derived from joint angle tracking.
      - joint_angvel_reward: reward derived from joint angular velocity tracking.
    
    Args:
        env (Any): The simulation environment. Expected attributes include:
            - env.data.qpos: current generalized positions.
            - env.data.qvel: current generalized velocities.
            - env.ref_traj.get_qpos(): reference generalized positions.
            - env.ref_traj.get_qvel(): reference generalized velocities.
            - env.reward_coefficients: dict containing keys "joint_angle" and "joint_angvel".
            - (Optional) env.EPSILON: minimum threshold for reward.
        k_pos (float): Scaling factor for the position error reward.
        k_vel (float): Scaling factor for the velocity error reward.
    
    Returns:
        Tuple[float, float, Dict[str, float]]:
            A tuple containing:
              - joint_angle_reward (float): Reward from joint angle tracking.
              - joint_angvel_reward (float): Reward from joint angular velocity tracking.
              - breakdown (dict): Dictionary with keys "joint_angle" and "joint_angvel" for detailed rewards.
    
    Raises:
        ValueError: If required data is missing or if there is a dimension mismatch.
    """
    # Validate that environment provides necessary state and reference trajectory methods.
    if not hasattr(env, 'data') or env.data is None:
        raise ValueError("Environment must have a valid 'data' attribute.")
    if not hasattr(env, 'ref_traj'):
        raise ValueError("Environment must have a valid 'ref_traj' attribute.")
    if not hasattr(env.ref_traj, 'get_qpos') or not callable(env.ref_traj.get_qpos):
        raise ValueError("env.ref_traj must implement a callable 'get_qpos' method.")
    if not hasattr(env.ref_traj, 'get_qvel') or not callable(env.ref_traj.get_qvel):
        raise ValueError("env.ref_traj must implement a callable 'get_qvel' method.")
    
    # Override k_pos and k_vel if provided in env.reward_coefficients.
    reward_coeffs = getattr(env, 'reward_coefficients', {})
    k_pos = reward_coeffs.get("joint_angle", k_pos)
    k_vel = reward_coeffs.get("joint_angvel", k_vel)
    
    # Retrieve simulation and reference data with error checking.
    try:
        sim_qpos = np.copy(env.data.qpos)
        sim_qvel = np.copy(env.data.qvel)
    except Exception as e:
        raise ValueError(f"Error accessing simulation state: {e}")
    
    try:
        ref_qpos = np.copy(env.ref_traj.get_qpos())
        ref_qvel = np.copy(env.ref_traj.get_qvel())
    except Exception as e:
        raise ValueError(f"Error accessing reference trajectory state: {e}")
        
    # Convert simulation qpos and qvel from reference trajectory representation using provided conversion functions.
    try:
        sim_qpos = inverse_convert_ref_traj_qpos(sim_qpos)
        sim_qvel = inverse_convert_ref_traj_qvel(sim_qvel, sim_qpos)
    except Exception as e:
        raise ValueError(f"Error during inverse conversion of simulation state: {e}")
    
    # Check that the state arrays have at least 3 DOFs to allow exclusion of translation.
    if sim_qpos.shape[0] < 3 or ref_qpos.shape[0] < 3:
        raise ValueError("qpos arrays must have at least 3 elements to exclude translational DOFs.")
    
    # Exclude the first three translational DOFs.
    sim_angles = sim_qpos[3:]
    ref_angles = ref_qpos[3:]
    sim_angvel = sim_qvel[3:]
    ref_angvel = ref_qvel[3:]
    
    # Ensure the dimensions match between simulation and reference rotational components.
    if sim_angles.shape != ref_angles.shape:
        raise ValueError(f"Mismatch in angle dimensions: sim_angles {sim_angles.shape} vs ref_angles {ref_angles.shape}")
    if sim_angvel.shape != ref_angvel.shape:
        raise ValueError(f"Mismatch in angular velocity dimensions: sim_angvel {sim_angvel.shape} vs ref_angvel {ref_angvel.shape}")
    
    # Define the weight vector
    num_joints = sim_angles.shape[0]
    upper_count = 14
    weights = np.ones(num_joints, dtype=np.float32)
    weights[-upper_count:] *= 0.2
    total_weight = np.sum(weights)
    
    # Compute the periodic difference for joint angles to account for wrap-around.
    jnt_ang_diff = np.arctan2(np.sin(sim_angles - ref_angles), np.cos(sim_angles - ref_angles))
    ang_mse = np.sum(weights * (jnt_ang_diff ** 2)) / total_weight
    jnt_ang_reward = np.exp(-k_pos * ang_mse)
    
    # Compute the difference for joint angular velocities.
    jnt_angvel_diff = sim_angvel - ref_angvel
    angvel_mse = np.sum(weights * (jnt_angvel_diff ** 2)) / total_weight
    jnt_angvel_reward = np.exp(-k_vel * angvel_mse)
    
    # base_diff = sim_qpos[:3] - ref_qpos[:3]
    # base_penalty = get_penalty(base_diff, np.zeros_like(base_diff),
    #                            env.reward_coefficients.get("base_trans", 1.0), False)
    # base_reward = np.exp(-base_penalty)
    
    # Apply minimum threshold if defined
    if hasattr(env, "EPSILON"):
        if jnt_ang_reward < env.EPSILON:
            jnt_ang_reward = 0.0
        if jnt_angvel_reward < env.EPSILON:
            jnt_angvel_reward = 0.0
            
        breakdown = {"joint_angle": jnt_ang_reward, "joint_angvel": jnt_angvel_reward,
                 "ang_mse": ang_mse, "angvel_mse": angvel_mse}
        
    return jnt_ang_reward, jnt_angvel_reward, breakdown

def get_com_speed_reward(env: Any, k_com: float = 0.5) -> Tuple[float, Dict[str, float]]:
    """
    Compute a center-of-mass (COM) speed goal reward based on the forward component
    of the COM velocity compared to the desired constant speed.

    Parameters:
      env (Any): The simulation environment.
      k_com (float): Scaling factor for the speed error penalty. A higher value penalizes
                     deviations more strongly. Default is 10.0.

    Returns:
      Tuple[float, Dict[str, float]]:
         - com_speed_reward (float): Reward for matching the desired COM speed.
         - breakdown (dict): Detailed info including desired_speed, sim_speed, speed_error,
                             penalty, and com_speed_reward.
    """
    # Validate that the environment has the required reference speed.
    if not hasattr(env, "ref_traj") or not hasattr(env.ref_traj, "current_speed"):
        raise ValueError("Environment must have 'ref_traj' with a 'current_speed' attribute.")
    desired_speed = env.ref_traj.current_speed
    
    # Override k_com if specified in env.reward_coefficients.
    reward_coeffs = getattr(env, 'reward_coefficients', {})
    k_com = reward_coeffs.get("com_speed", k_com)

    # Retrieve the COM kinematics.
    try:
        _, com_info = get_COM_kinematics(env)
    except Exception as e:
        raise ValueError(f"Error getting COM kinematics: {e}")

    # Extract the forward (x-axis) component of COM velocity.
    sim_speed = com_info['com_vel'][0]
    speed_error = sim_speed - desired_speed
    com_speed_reward = np.exp(-k_com * (speed_error ** 2))

    # Apply a minimal threshold if env.EPSILON is defined.
    if hasattr(env, "EPSILON"):
        eps = env.EPSILON
        if com_speed_reward < eps:
            com_speed_reward = 0.0

    breakdown = {
        "com_speed_reward": com_speed_reward,
        "speed_err": speed_error
    }

    return com_speed_reward, breakdown

def get_grf_reward(env: Any, 
                   mode: str = "full", 
                   k_grf: float = 1.0,
                   sigma: float = 0.1) -> Tuple[float, Dict[str, float]]:
    """
    Compute the GRF-based smooth reward by penalizing excessive ground reaction forces.
    
    Args:
        env: The simulation environment. Expected attributes:
            - env.reward_coefficients (optional dict) may override "grf" and "grf_sigma".
            - env.total_mass (optional float). Falls back to sum of model.body_mass.
            - env.EPSILON (optional float) to threshold small rewards to zero.
        mode:  "full" to use the Euclidean norm of the 3D GRF vectors,
               "z" to use only the vertical component.
        k_grf: Scaling factor for the Gaussian exponent.
        sigma: Standard deviation of the Gaussian tolerance band.

    Returns:
        A tuple of:
         - grf_reward: Scalar in [0,1], highest at normalized GRF = 1.
         - breakdown: Dict containing
             * "grf_reward": the computed reward
             * "total_grf_norm": (R+L)/(m·g)
    """
    try:
        # Override k_grf if specified in env.reward_coefficients
        reward_coeffs = getattr(env, 'reward_coefficients', {})
        k_grf = reward_coeffs.get("grf", k_grf)
        sigma_used = float(reward_coeffs.get("grf_sigma", sigma))
        if sigma_used <= 0:
            raise ValueError(f"Invalid sigma {sigma_used}; must be > 0")
        # Get total_mass from the environment, or as a fallback compute from the model's body masses.
        total_mass = getattr(env, "total_mass", np.sum(env.model.body_mass))
            
        # --- fetch GRF info ---
        _, grf_info = get_GRF_info(env)
        R_force = grf_info["right"]["GRF_force"]
        L_force = grf_info["left"]["GRF_force"]

        mode_l = mode.lower()
        if mode_l == "z":
            R_val = float(R_force[2])
            L_val = float(L_force[2])
        elif mode_l == "full":
            R_val = float(np.linalg.norm(R_force))
            L_val = float(np.linalg.norm(L_force))
        else:
            raise ValueError(f"Invalid mode '{mode}'; must be 'full' or 'z'")
        
        # Normalize the total GRF by the product of total_mass and gravity.
        total_grf_norm = (R_val + L_val) / total_mass / 9.81
        exponent = (total_grf_norm - 1) ** 2 / (2 * sigma_used ** 2)
        grf_reward = np.exp(-k_grf * exponent)
        
        if hasattr(env, "EPSILON") and grf_reward < env.EPSILON:
            grf_reward = 0.0
        
        breakdown = {"grf_reward": grf_reward, "total_grf_normalized": total_grf_norm}
        
        return grf_reward, breakdown
    
    except Exception as e:
       logger.error("Failed to compute GRF reward: %s", e)
       return 0.0, {"grf_reward": 0.0, "total_grf_norm": 0.0}
    
def get_torque_reward(env: Any, k_torque: float = 1e-6) -> Tuple[float, Dict[str, float]]:
    """
    Compute the torque reward by penalizing excessive joint actuator forces.
    
    Args:
        env (Any): The simulation environment.
    
    Returns:
        float: The computed torque reward.
    """
    try:
        # Override k_grf if specified in env.reward_coefficients
        reward_coeffs = getattr(env, 'reward_coefficients', {})
        k_torque = reward_coeffs.get("torque", k_torque)
        
        # Retrieve actuator forces (ensure they are in a numpy array with proper shape).
        torque = env.data.qfrc_actuator.copy()[6:]
        if torque.shape[0] != env.num_actuators:
            raise ValueError(f"Torque dimension {torque.shape[0]} does not match expected {env.num_actuators}.")
            
        penalty = get_penalty(torque, np.zeros_like(torque), k_torque, False)
        tor_reward = np.exp(-penalty)

        if hasattr(env, "EPSILON") and tor_reward < env.EPSILON:
            tor_reward = 0.0
        
        breakdown = {"tor_reward": tor_reward}

        return tor_reward, breakdown
    
    except Exception as e:
        logger.error(f"Failed to compute torque reward: {e}")
        return 0.0, {}

def get_action_reward(env: Any, k_action: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """
    Compute the action smoothness reward based on the difference between the current and previous RL actions.
    
    Returns:
        Tuple[float, Dict[str, float]]: A tuple containing:
            - action_reward (float): The computed smooth action reward.
            - breakdown (dict): A dictionary containing "act_reward" and the raw "penalty" for debugging.
    """
    try:
        # Ensure that required action attributes exist.
        if not hasattr(env, "current_action"):
            raise ValueError("Current action (env.current_action) attribute is missing.")
        if not hasattr(env, "prev_action"):
            raise ValueError("Previous action (env.prev_action) attribute is missing.")
        
        # Override k_action if specified in env.reward_coefficients using the key "k_action".
        reward_coeffs = getattr(env, 'reward_coefficients', {})
        k_action = reward_coeffs.get("action", k_action)
        
        # If actions are None, assume no change and return a maximal reward.
        if env.current_action is None or env.prev_action is None:
            return 1.0, {"act_reward": 1.0, "penalty": 0.0}
        
        # Get the current and previous actions.
        current_action = env.current_action.copy()
        previous_action = env.prev_action.copy()
        
        penalty = get_penalty(current_action, previous_action, k_action, False)
        act_reward = np.exp(-k_action * penalty)
        
        # If EPSILON is defined and the computed reward is below it, set reward to zero.
        if hasattr(env, "EPSILON"):
            eps = env.EPSILON
            if act_reward < eps:
                act_reward = 0.0
        
        breakdown = {"act_reward": act_reward}

        return act_reward, breakdown
    
    except Exception as e:
        logger.error(f"Failed to compute action reward: {e}")
        return 0.0, {"act_reward": 0.0, "penalty": 0.0}
    





        