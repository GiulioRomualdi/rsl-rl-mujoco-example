"""
Reward functions for musculoskeletal simulation environment.
Provides imitation reward components and (currently stubbed) goal reward.
@author: YAKE
"""
import math
import numpy as np
from .common_utils import (
    quat_to_mat,
    orient6_to_mat,
    rotation_geodesic,
    inverse_convert_ref_traj_qpos, 
    inverse_convert_ref_traj_qvel, 
    get_penalty
    )
from .state import (
    compute_ref_pelvis_kinematics, 
    compute_ref_site_kinematics, 
    get_site_kinematics, 
    get_COM_kinematics, 
    get_GRF_info
    )
import logging
from typing import Any, Tuple, Dict, Optional, Union

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def compute_reward(
        env: Any, 
        obs_components: Dict[str, Any]
        ) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the total reward for the current step, using only the precomputed
    observation components (obs_components) to avoid redundant state extraction.

    The reward is assembled based on env.phase:
      - IMITATION:  w_imitation * imitation_reward + w_smooth * smooth_reward
      - REACH_GOAL: w_goal      * goal_reward      + w_smooth * smooth_reward

    Each sub-reward is computed via a dedicated function:
      - get_imitation_reward(env, obs_components)
      - get_goal_reward(env, obs_components)
      - get_smooth_reward(env, obs_components)

    Parameters
    ----------
    env : Any
        The environment instance. Must provide:
          - reward_weights : Dict[str, float] with keys "imitation", "goal", "smooth"
          - phase          : Enum with .name in {"IMITATION","REACH_GOAL"}
          - (optional) logger for warnings/errors
    obs_components : Dict[str, Any]
        The `info["obs_components"]` dict produced by `step()`, containing
        precomputed kinematics, contacts, etc.

    Returns
    -------
    total_reward : float
        The weighted sum of the selected sub-rewards.
    breakdown : Dict[str, Any]
        A dict with keys depending on phase:
          - "imitation", "smooth"  if phase=="IMITATION"
          - "goal", "smooth"       if phase=="REACH_GOAL"
        Each value is the sub-dict returned by the corresponding function,
        or an error entry if that sub-reward failed.
    """
    try:
        # Validate required attributes
        weights = getattr(env, "reward_weights", None)
        phase = getattr(env, "phase", None)
        log = getattr(env, "logger", None)
        
        if not isinstance(weights, dict):
            raise TypeError("env.reward_weights must be a dict")
        if phase is None or not hasattr(phase, "name"):
            raise ValueError("env.phase must be defined with a .name")
            
        phase_name = phase.name
        total_reward = 0.0
        breakdown = {}
        
        def safe_reward(fn, label) -> Tuple[float, Dict[str, Any]]:
            try:
                value, info = fn(env, obs_components)
                if not isinstance(value, (float, int)) or math.isnan(value):
                    raise ValueError(f"{label} returned invalid value: {value}")
                return value, info
            except Exception as e:
                if log:
                    log.warning(f"{label} reward computation failed: {e}")
                return 0.0, {f"{label}_error": str(e)}
            
        if phase_name == "IMITATION":
            imit_v, imit_info = safe_reward(get_imitation_reward, "imitation")
            smooth_v, smooth_info = safe_reward(get_smooth_reward, "smooth")
            total_reward = (
                weights.get("imitation", 0.0) * imit_v +
                weights.get("smooth",     0.0) * smooth_v
            )
            breakdown = {"imitation": imit_info, "smooth": smooth_info}

        elif phase_name == "REACH_GOAL":
            goal_v, goal_info = safe_reward(get_goal_reward, "goal")
            smooth_v, smooth_info = safe_reward(get_smooth_reward, "smooth")
            total_reward = (
                weights.get("goal",   0.0) * goal_v +
                weights.get("smooth", 0.0) * smooth_v
            )
            breakdown = {"goal": goal_info, "smooth": smooth_info}

        else:
            raise ValueError(f"Unrecognized phase: {phase_name}")

        return total_reward, breakdown

    except Exception as e:
        if log:
            log.error(f"compute_reward failed entirely: {e}")
        return 0.0, {"error": str(e)}

def get_imitation_reward(
        env: Any,
        obs_components: Dict[str, Any]
        ) -> Tuple[float, Dict[str, Union[float, Dict[str, Any]]]]:
    """
    Compute the imitation reward as a weighted sum of four sub-rewards:
      - site_pos     (body-site position tracking)
      - site_vel     (body-site velocity tracking)
      - joint_angle  (joint-angle tracking)
      - joint_angvel (joint-angular-velocity tracking)

    Each sub-reward is obtained by delegating to:
      get_site_tracking_reward(...)
      get_joint_tracking_reward(env)

    Parameters
    ----------
    env : Any
        Must define:
          - env.imitation_weights : dict with keys
              'site_pos', 'site_vel', 'joint_angle', 'joint_angvel'
          - env.logger (optional) for warnings
    obs_components : dict
        The `info["obs_components"]` dict returned by env.step()/reset(), containing:
          - "joint":   {'pos','linvel','orient','angvel'} → dicts of site→ndarray
          - "pelvis":  {'pos','linvel','orient','angvel'} → pelvis kinematics

    Returns
    -------
    total_reward : float
        Weighted sum:  
          w_sp*r_sp + w_sv*r_sv + w_ja*r_ja + w_jv*r_jv  
    breakdown : dict
        {
          "site_pos":     float,
          "site_vel":     float,
          "joint_angle":  float,
          "joint_angvel": float,
          "site_info":    dict,   # diagnostics from site tracking
          "joint_info":   dict    # diagnostics from joint tracking
        }
    """
    iw = getattr(env, "imitation_weights", None)
    if not isinstance(iw, dict):
        raise TypeError("env.imitation_weights must be a dict.")
    required = {"site_pos", "site_vel", "joint_angle", "joint_angvel"}
    missing = required - set(iw.keys())
    if missing:
        raise KeyError(f"Missing imitation_weights entries: {missing}")
    
    w_sp = float(iw["site_pos"])
    w_sv = float(iw["site_vel"])
    w_ja = float(iw["joint_angle"])
    w_jv = float(iw["joint_angvel"])
    
    logger = getattr(env, "logger", None)
    
    try:
        r_sp, r_sv, _, _, site_info = get_site_tracking_reward(
            env, obs_components,
            include_pelvis=True,
            include_orientation=False,
            site_weights=None
        )
    except Exception as e:
        if logger:
            logger.warning(f"get_site_tracking_reward failed: {e}")
        r_sp = r_sv = 0.0
        site_info = {"site_error": str(e)}
        
    try:
        r_ja, r_jv, joint_info = get_joint_tracking_reward(env)
    except Exception as e:
        if logger:
            logger.warning(f"get_joint_tracking_reward failed: {e}")
        r_ja = r_jv = 0.0
        joint_info = {"joint_error": str(e)}
    
    total = w_sp * r_sp + w_sv * r_sv + w_ja * r_ja + w_jv * r_jv

    breakdown: Dict[str, Union[float, Dict[str, Any]]] = {
        "site_pos":     r_sp,
        "site_vel":     r_sv,
        "joint_angle":  r_ja,
        "joint_angvel": r_jv,
        "site_info":    site_info,
        "joint_info":   joint_info
    }

    return total, breakdown    

def get_site_tracking_reward(
        env: Any,
        obs_components: Dict[str, Any],
        include_pelvis: bool = False,
        include_orientation: bool = False,
        site_weights: Optional[Dict[str, float]] = None
        ) -> Tuple[float, float, Optional[float], Optional[float], Dict[str, Any]]:
    """
    Compute imitation rewards for joint sites (and optionally pelvis) by comparing
    simulation vs reference kinematics.
    
    The four possible rewards are:
      - r_pos     = exp(-k_pos     * mean_weighted_MSE_pos)
      - r_vel     = exp(-k_vel     * mean_weighted_MSE_linvel)
      - r_orn     = exp(-k_orn     * mean_weighted_MSE_orient)    if include_orientation
      - r_angvel  = exp(-k_angvel  * mean_weighted_MSE_angvel)    if include_orientation

    Parameters
    ----------
    env : Any
        Must define env.reward_coefficients dict, and env.ref_traj API.
    obs_components : dict
        Output of get_obs(), with keys:
          - "joint": {"pos", "linvel", "orient", "angvel"} → each a dict site_name→np.ndarray
          - "pelvis":      {"pos", "orient", "linvel", "angvel"} for the pelvis
    include_pelvis : bool
        If True, include pelvis in comparisons.
    include_orientation : bool
        If True, compute orientation-error reward and angvel-error reward.
    site_weights : dict, optional
        Per-site weight multipliers; defaults to 1.0 for all sites.

    Returns
    -------
    r_pos    : float
    r_vel    : float
    r_orn    : float or None
    r_angvel : float or None
    breakdown : dict
        {
          "pos_err": float,
          "vel_err": float,
          "orn_err": float or None,
          "angvel_err": float or None,
          "num_sites": int,
          "weights_sum": float
        }
    """
    EPSILON = 1e-8
    try:
        ref_kin = compute_ref_site_kinematics(env)
        sim_kin = obs_components["joint"]
        
        if include_pelvis:
            ref_pel = compute_ref_pelvis_kinematics(env, use_free_joint=True)
            sim_pel = obs_components["pelvis"]
            for key in ("pos","linvel","orient","angvel"):
                ref_kin[key]["pelvis"] = ref_pel[key]
                sim_kin[key]["pelvis"] = sim_pel[key]
            
        sites = list(ref_kin["pos"].keys() & sim_kin["pos"].keys())
        N = len(sites)
        if N == 0:
            return 0.0, 0.0, None, None, {
                "pos_err":0.0, "vel_err":0.0,
                "orn_err":None, "angvel_err":None,
                "num_sites":0, "weights_sum":0.0
            }
        
        get_w = site_weights.get if site_weights else (lambda s, d=1.0: 1.0)
        
        sum_w        = 0.0
        sum_pos_err  = 0.0
        sum_vel_err  = 0.0
        sum_rot_err  = 0.0
        sum_ang_err  = 0.0
        
        for s in sites:
            w = float(get_w(s, 1.0))
            sum_w += w
            
            # position MSE
            dp = sim_kin["pos"][s] - ref_kin["pos"][s]
            sum_pos_err += w * float(dp.dot(dp))
            
            # velocity MSE
            dv = sim_kin["linvel"][s] - ref_kin["linvel"][s]
            sum_vel_err += w * float(dv.dot(dv))
            
            if include_orientation:
                # orientation geodesic
                if s == "pelvis":
                    Rr = quat_to_mat(ref_kin["orient"][s])
                    Rs = quat_to_mat(sim_kin["orient"][s])
                else:
                    Rr = orient6_to_mat(ref_kin["orient"][s])
                    Rs = orient6_to_mat(sim_kin["orient"][s])
                θ = rotation_geodesic(Rr, Rs)
                sum_rot_err += w * (θ * θ)
                
                # angular velocity MSE (local gyro outputs)
                da = sim_kin["angvel"][s] - ref_kin["angvel"][s]
                sum_ang_err += w * float(da.dot(da))
            
        pos_err    = sum_pos_err / sum_w
        if pos_err <= EPSILON:
            pos_err = 0.0
        vel_err    = sum_vel_err / sum_w
        if vel_err <= EPSILON:
            vel_err = 0.0
            
        rot_err = None
        ang_err = None
        if include_orientation:
            rot_err = sum_rot_err / sum_w
            ang_err = sum_ang_err / sum_w
            if rot_err <= EPSILON:
                rot_err = 0.0
            if ang_err <= EPSILON:
                ang_err = 0.0
        
        coeffs = env.reward_coefficients
        r_pos    = math.exp(-float(coeffs.get("site_pos",    20.0)) * pos_err)
        r_vel    = math.exp(-float(coeffs.get("site_vel",     0.05)) * vel_err)
        if abs(r_pos - 1.0) < EPSILON:
            r_pos = 1.0
        if abs(r_vel - 1.0) < EPSILON:
            r_vel = 1.0
        
        r_orn    = None
        r_angvel = None
        if include_orientation:
            r_orn    = math.exp(-float(coeffs.get("site_orn",    1.0)) * rot_err)
            r_angvel = math.exp(-float(coeffs.get("site_angvel", 1.0)) * ang_err)
            if abs(r_orn - 1.0) < EPSILON:
                r_orn = 1.0
            if abs(r_angvel - 1.0) < EPSILON:
                r_angvel = 1.0
        
        breakdown = {
            "pos_err":    pos_err,
            "vel_err":    vel_err,
            "orn_err":    rot_err,
            "angvel_err": ang_err,
            "num_sites":  N,
            "weights_sum": sum_w
        }
        return r_pos, r_vel, r_orn, r_angvel, breakdown
            
    except Exception as e:
        return 0.0, 0.0, None, None, {"error": str(e)}

def get_joint_tracking_reward(env: Any) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute rewards for joint‐angle and joint‐angular‐velocity tracking by comparing
    the simulated state against the reference trajectory in joint‐space.

    Rewards
    -------
    joint_angle_reward  = exp(-k_angle  * weighted_MSE(angle_error))
    joint_angvel_reward = exp(-k_angvel * weighted_MSE(angvel_error))

    Parameters
    ----------
    env : Any
        Must provide:
          - env.data.qpos   : NDArray, length>=7
          - env.data.qvel   : NDArray, length>=6
          - env.ref_traj.get_extend_refer() -> (ref_qpos, ref_qvel)
          - env.reward_coefficients : dict with optional keys
                "joint_angle", "joint_angvel", 
                "joint_weight_decay", "joint_decay_fraction"
          - Optional env.EPSILON : float

    Returns
    -------
    Tuple[
      joint_angle_reward  : float,
      joint_angvel_reward : float,
      breakdown           : Dict[str, float]
    ]
    breakdown keys:
      "angle_mse", "angvel_mse", "num_joints", "weight_sum"
    """
    coeffs = getattr(env, "reward_coefficients", {})
    k_ang       = float(coeffs.get("joint_angle",   20.0))
    k_vel       = float(coeffs.get("joint_angvel",   0.05))
    decay       = coeffs.get("joint_weight_decay", None)
    frac        = float(coeffs.get("joint_decay_fraction", 0.2))
    EPSILON     = getattr(env, "EPSILON", 1e-8)
    
    data = getattr(env, "data", None)
    ref  = getattr(env, "ref_traj", None)
    if data is None or ref is None:
        return 0.0, 0.0, {"error": "env.data or env.ref_traj missing"}
    
    # Extract raw arrays
    try:
        sim_qpos = data.qpos
        sim_qvel = data.qvel
        ref_qpos, ref_qvel = ref.get_extend_refer()
    except Exception as e:
        return 0.0, 0.0, {"error": f"State retrieval failed: {e}"}
    
    # Convert simulation → reference‐traj space
    try:
        sim_qpos_r = inverse_convert_ref_traj_qpos(sim_qpos)
        sim_qvel_r = inverse_convert_ref_traj_qvel(sim_qvel, sim_qpos_r)
    except Exception as e:
        return 0.0, 0.0, {"error": f"Conversion error: {e}"}
    
    if sim_qpos_r.ndim != 1 or ref_qpos.ndim != 1 or sim_qpos_r.size < 4 or ref_qpos.size < 4:
        return 0.0, 0.0, {"error": "qpos length must be >=4"}
    
    angles_sim = sim_qpos_r[3:]
    angles_ref = ref_qpos[3:]
    angvel_sim = sim_qvel_r[3:]
    angvel_ref = ref_qvel[3:]
    
    if angles_sim.shape != angles_ref.shape or angvel_sim.shape != angvel_ref.shape:
        return 0.0, 0.0, {
            "error": f"Shape mismatch: angles {angles_sim.shape}!={angles_ref.shape}, "
                     f"angvel {angvel_sim.shape}!={angvel_ref.shape}"
        }
    
    num_j = angles_sim.size
    w = np.ones(num_j, dtype=np.float32)
    if decay is not None and 0.0 < decay < 1.0:
        k = max(1, int(num_j * frac))
        w = np.concatenate([
            np.ones(num_j - k, dtype=np.float32),
            np.full(k, decay, dtype=np.float32)
        ])
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return 0.0, 0.0, {"error": "zero joint weight sum"}
    
    # Compute angle‐error MSE
    delta = angles_sim - angles_ref
    dtheta = np.arctan2(np.sin(delta), np.cos(delta))
    angle_mse = float((w * (dtheta * dtheta)).sum() / w_sum)
    
    # Compute angular‐velocity MSE
    domega = angvel_sim - angvel_ref
    angvel_mse = float((w * (domega * domega)).sum() / w_sum)
    
    # Exponential rewards
    r_ang    = math.exp(-k_ang * angle_mse)
    r_angvel = math.exp(-k_vel * angvel_mse)
    if EPSILON is not None:
        if r_ang    < EPSILON: r_ang    = 0.0
        if r_angvel < EPSILON: r_angvel = 0.0

    breakdown = {
        "angle_mse":  angle_mse,
        "angvel_mse": angvel_mse,
        "num_joints": num_j,
        "weight_sum": w_sum
    }
    return r_ang, r_angvel, breakdown

def get_goal_reward(
        env: Any,
        obs_components: Dict[str, Any]
        ) -> Tuple[float, Dict[str, Union[float, Dict[str, Any]]]]:
    return 0.0, {}

def get_smooth_reward(
        env: Any,
        obs_components: Dict[str, Any]
        ) -> Tuple[float, Dict[str, Union[float, Dict[str, Any]]]]:
    return 0.0, {}

def get_com_speed_reward(env: Any, k_com: float = 0.5) -> Tuple[float, Dict[str, float]]:
    return 0, {}

def get_grf_reward(env: Any, 
                   mode: str = "full", 
                   k_grf: float = 0.1,
                   sigma: float = 0.1) -> Tuple[float, Dict[str, float]]:
    return 0, {}

def get_torque_reward(env: Any, k_torque: float = 1e-6) -> Tuple[float, Dict[str, float]]:
    return 0, {}

def get_action_reward(env: Any, k_action: float = 2.0) -> Tuple[float, Dict[str, float]]:
    return 0, {}
    





        