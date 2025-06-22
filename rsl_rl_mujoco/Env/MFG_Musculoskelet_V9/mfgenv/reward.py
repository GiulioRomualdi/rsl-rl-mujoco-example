"""
Reward functions for musculoskeletal simulation environment.
Provides imitation reward components and (currently stubbed) goal reward.
@author: YAKE
"""
import math
import logging
from functools import lru_cache
from typing import Any, Tuple, Dict, Optional, Union

import numpy as np
from numba import njit

from mfgenv.common_utils import (
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


# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

@njit
def _weighted_mse(delta: np.ndarray, w: np.ndarray, w_sum: float) -> float:
    """
    Compute weighted mean squared error between two 1D arrays.
    """
    total = 0.0
    for i in range(delta.shape[0]):
        total += w[i] * delta[i] * delta[i]
    return total / w_sum

@njit
def _mse3d(a: np.ndarray, b: np.ndarray, w: np.ndarray, w_sum: float) -> float:
    total = 0.0
    for i in range(a.shape[0]):
        dx = a[i,0] - b[i,0]
        dy = a[i,1] - b[i,1]
        dz = a[i,2] - b[i,2]
        total += w[i] * (dx*dx + dy*dy + dz*dz)
    return total / w_sum

@lru_cache(maxsize=None)
def _build_joint_weights(num_j: int,
                         decay: float,
                         frac: float) -> Tuple[np.ndarray, float]:
    """
    Build and cache a joint‐weight vector and its sum.
    
    Parameters
    ----------
    num_j : int
        Number of joints.
    decay : float
        Decay factor for the last k joints (0<decay<1), or None.
    frac : float
        Fraction of joints to apply decay (0<frac<1).
    
    Returns
    -------
    w : np.ndarray
        Weight vector of length num_j.
    w_sum : float
        Sum of weights.
    """
    # create uniform weights
    w = np.ones(num_j, dtype=np.float32)
    # apply decay to the last k joints if requested
    if decay is not None and 0.0 < decay < 1.0:
        k = max(1, int(num_j * frac))
        w[-k:] = decay
    return w, float(w.sum())

def compute_reward(
        env: Any, 
        obs_components: Dict[str, Any],
        ) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the overall reward for the current step.

    Expects obs_components to contain exactly these four entries:
      - "joint"   : simulated joint-site kinematics
      - "pelvis"  : simulated pelvis kinematics
      - "ref_kin" : reference joint-site kinematics
      - "ref_pel" : reference pelvis kinematics

    Reward composition:
      - IMITATION:  w_imitation * imitation_reward + w_smooth * smooth_reward
      - REACH_GOAL: w_goal      * goal_reward      + w_smooth * smooth_reward

    Delegates to:
      - get_imitation_reward(env, obs_components)
      - get_goal_reward(env, obs_components)
      - get_smooth_reward(env, obs_components)

    Parameters
    ----------
    env : Any
        Must provide:
          - phase.name in {"IMITATION", "REACH_GOAL"}
          - reward_weights : dict with keys "imitation", "smooth", "goal"
          - logger         : for error logging
    obs_components : dict
        Must include "joint", "pelvis", "ref_kin", and "ref_pel".

    Returns
    -------
    total_reward : float
        Weighted sum of sub-rewards.
    info : dict
        {
          "imitation": {...},  # only if IMITATION
          "smooth": {...},
          "goal": {...}        # only if REACH_GOAL
        }
    """
    try:
        phase = env.phase.name
        rw    = env.reward_weights
        log   = env.logger
        
        do_imitation = get_imitation_reward
        do_smooth    = get_smooth_reward
        do_goal      = get_goal_reward

        info: Dict[str, Any] = {}
        total = 0.0

        if phase == "IMITATION":
            r_imm, imm_info = do_imitation(env, obs_components)
            info["imitation"] = imm_info

            r_smooth, smooth_info = do_smooth(env, obs_components)
            info["smooth"] = smooth_info

            total = rw["imitation"] * r_imm + rw["smooth"]  * r_smooth

        elif phase == "REACH_GOAL":
            r_goal, goal_info = do_goal(env, obs_components)
            info["goal"] = goal_info

            r_smooth, smooth_info = do_smooth(env, obs_components)
            info["smooth"] = smooth_info

            total = rw["goal"] * r_goal + rw["smooth"] * r_smooth

        else:
            raise ValueError(f"Unknown phase {phase!r}")

        return float(total), info

    except Exception as e:
        log = getattr(env, "logger", None)
        if log:
            log.error("compute_reward failed: %s", e)
        return 0.0, {"error": str(e)}

def get_imitation_reward(
    env: Any,
    obs_components: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the imitation-phase reward as a weighted sum of four sub-rewards.

    Sub-rewards:
      - site_pos     : body-site position tracking
      - site_vel     : body-site velocity tracking
      - joint_angle  : joint-angle tracking
      - joint_angvel : joint-angular-velocity tracking

    Parameters
    ----------
    env : Any
        Must provide:
          - env.imitation_weights : dict with keys
                'site_pos', 'site_vel', 'joint_angle', 'joint_angvel'
          - env.logger            : a Logger for error reporting
    obs_components : dict
        Must include:
          - "joint"   : simulated joint sites kin
          - "pelvis"  : simulated pelvis kin
          - "ref_kin" : reference joint sites kin
          - "ref_pel" : reference pelvis kin

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
    try:
        iw = getattr(env, "imitation_weights", None)
        if not isinstance(iw, dict):
            raise TypeError("env.imitation_weights must be a dict")
        w_sp = float(iw["site_pos"])
        w_sv = float(iw["site_vel"])
        w_ja = float(iw["joint_angle"])
        w_jv = float(iw["joint_angvel"])

        sim_kin = obs_components["joint"]
        pel_kin = obs_components["pelvis"]
        ref_kin = obs_components["ref_kin"]
        ref_pel = obs_components["ref_pel"]

        r_sp, r_sv, _, _, site_info = get_site_tracking_reward(
            env,
            sim_kin,
            ref_kin,
            pel_kin,
            ref_pel,
            include_pelvis=True,
            include_orientation=False,
            site_weights=None
        )

        r_ja, r_jv, joint_info = get_joint_tracking_reward(env)

        total = w_sp * r_sp + w_sv * r_sv + w_ja * r_ja + w_jv * r_jv

        breakdown = {
            "site_pos":     r_sp,
            "site_vel":     r_sv,
            "joint_angle":  r_ja,
            "joint_angvel": r_jv,
            "site_info":    site_info,
            "joint_info":   joint_info
        }
        return float(total), breakdown

    except Exception as e:
        env.logger.error("get_imitation_reward failed: %s", e)
        return 0.0, {
            "site_pos":     0.0,
            "site_vel":     0.0,
            "joint_angle":  0.0,
            "joint_angvel": 0.0,
            "site_info":    {"error": str(e)},
            "joint_info":   {"error": str(e)}
        }    

def get_site_tracking_reward(
        env: Any,
        sim_kin: Dict[str, np.ndarray],
        ref_kin: Dict[str, np.ndarray],
        sim_pel_kin: Optional[Dict[str, np.ndarray]] = None,
        ref_pel_kin: Optional[Dict[str, np.ndarray]] = None,
        include_pelvis: bool = True,
        include_orientation: bool = False,
        site_weights: Optional[Dict[str, float]] = None
        ) -> Tuple[float, float, Optional[float], Optional[float], Dict[str, Any]]:
    """
    Compute the four site‐tracking rewards (position, velocity,
    optionally orientation & angular-velocity) by replicating the original loop.

    Parameters
    ----------
    env : Any
        Environment with env.reward_coefficients dict.
    sim_kin : dict
        Simulated site kinematics: keys "pos","linvel","orient","angvel".
    ref_kin : dict of dict
        Reference site kinematics (excluding pelvis): same keys.
    sim_pel_kin : dict
        Simulated pelvis kinematics: keys "pos","linvel","orient","angvel".
    ref_pel_kin : dict
        Reference pelvis kinematics: same keys.
    include_pelvis : bool
        If True, merge pelvis into the site dicts before computing.
    include_orientation : bool
        If True, compute orientation & angvel errors.
    site_weights : dict, optional
        Per-site weight multipliers; defaults to 1.0.

    Returns
    -------
    r_pos : float
    r_vel : float
    r_orient : float or None
    r_angvel : float or None
    breakdown : dict
        {
          "pos_err", "vel_err", "orn_err", "angvel_err",
          "num_sites", "weights_sum"
        }
    """
    EPSILON = 1e-8
    try:
        pos_sim = dict(sim_kin["pos"])
        vel_sim = dict(sim_kin["linvel"])
        pos_ref = dict(ref_kin["pos"])
        vel_ref = dict(ref_kin["linvel"])
        if include_pelvis:
            pos_sim["pelvis"] = sim_pel_kin["pos"]
            vel_sim["pelvis"] = sim_pel_kin["linvel"]
            pos_ref["pelvis"] = ref_pel_kin["pos"]
            vel_ref["pelvis"] = ref_pel_kin["linvel"]

        sites = list(pos_ref.keys() & pos_sim.keys())
        N = len(sites)
        if N == 0:
            return 0.0, 0.0, None, None, {
                "pos_err": 0.0, "vel_err": 0.0,
                "orn_err": None, "angvel_err": None,
                "num_sites": 0, "weights_sum": 0.0
            }

        if site_weights:
            w = np.array([float(site_weights.get(s, 1.0)) for s in sites], dtype=np.float64)
        else:
            w = np.ones(N, dtype=np.float64)
        w_sum = float(w.sum())
        if w_sum <= 0.0:
            raise ValueError("Non-positive total site weight")

        pos_sim_arr = np.vstack([pos_sim[s] for s in sites])
        pos_ref_arr = np.vstack([pos_ref[s] for s in sites])
        vel_sim_arr = np.vstack([vel_sim[s] for s in sites])
        vel_ref_arr = np.vstack([vel_ref[s] for s in sites])
        
        pos_err = _mse3d(pos_sim_arr, pos_ref_arr, w, w_sum)
        vel_err = _mse3d(vel_sim_arr, vel_ref_arr, w, w_sum)

        if pos_err <= EPSILON: pos_err = 0.0
        if vel_err <= EPSILON: vel_err = 0.0

        orn_err = None
        ang_err = None
        if include_orientation:
            orn_err = 0.0
            ang_err = 0.0
            for idx, s in enumerate(sites):
                if s == "pelvis":
                    Rr = quat_to_mat(ref_pel_kin["orient"])
                    Rs = quat_to_mat(sim_pel_kin["orient"])
                else:
                    Rr = orient6_to_mat(ref_kin["orient"][s])
                    Rs = orient6_to_mat(sim_kin["orient"][s])
                θ = rotation_geodesic(Rr, Rs)
                orn_err += w[idx] * (θ * θ)

                da = (sim_kin["angvel"][s] if s != "pelvis" else sim_pel_kin["angvel"]) \
                   - (ref_kin["angvel"][s] if s != "pelvis" else ref_pel_kin["angvel"])
                ang_err += w[idx] * float(da.dot(da))

            orn_err /= w_sum
            ang_err /= w_sum
            if orn_err <= EPSILON: orn_err = 0.0
            if ang_err <= EPSILON: ang_err = 0.0

        coeffs = env.reward_coefficients
        r_pos = math.exp(-float(coeffs.get("site_pos", 20.0)) * pos_err)
        r_vel = math.exp(-float(coeffs.get("site_vel", 0.05)) * vel_err)
        if abs(r_pos-1.0) < EPSILON: r_pos = 1.0
        if abs(r_vel-1.0) < EPSILON: r_vel = 1.0

        r_orn = None
        r_angvel = None
        if include_orientation:
            r_orn = math.exp(-float(coeffs.get("site_orn", 1.0)) * orn_err)
            r_angvel = math.exp(-float(coeffs.get("site_angvel", 1.0)) * ang_err)
            if abs(r_orn-1.0)    < EPSILON: r_orn    = 1.0
            if abs(r_angvel-1.0) < EPSILON: r_angvel = 1.0

        breakdown = {
            "pos_err":    pos_err,
            "vel_err":    vel_err,
            "orn_err":    orn_err,
            "angvel_err": ang_err,
            "num_sites":  N,
            "weights_sum": w_sum
        }
        return r_pos, r_vel, r_orn, r_angvel, breakdown

    except Exception as e:
        env.logger.error("get_site_tracking_reward failed: %s", e)
        return 0.0, 0.0, None, None, {"error": str(e)}

def get_joint_tracking_reward(
        env: Any
        ) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute joint‐space imitation rewards by comparing simulated vs reference.

    Sub‐steps:
      1. Retrieve sim_qpos, sim_qvel, ref_qpos, ref_qvel.
      2. Map sim data into reference joint space.
      3. Slice off the first 3 DOFs; compute deltas.
      4. Build (cached) weight vector and sum.
      5. Compute weighted MSE via JIT helper.
      6. Exponentiate with coefficients and clamp by EPSILON.
    
    Parameters
    ----------
    env : Any
        Environment providing:
          - env.data.qpos : np.ndarray
          - env.data.qvel : np.ndarray
          - env.ref_traj.get_reference_trajectories() -> (ref_qpos, ref_qvel)
          - env.reward_coefficients : dict with keys
                "joint_angle", "joint_angvel", "joint_weight_decay",
                "joint_decay_fraction"
          - Optional env.EPSILON : float
    
    Returns
    -------
    r_angle : float
        Exponential joint‐angle reward.
    r_angvel : float
        Exponential joint‐velocity reward.
    info : dict
        {
          "angle_mse": float,
          "angvel_mse": float,
          "num_joints": int,
          "weight_sum": float
        }
    """
    try:
        data = getattr(env, "data", None)
        ref  = getattr(env, "ref_traj", None)
        if data is None or ref is None:
            raise RuntimeError("env.data or env.ref_traj is missing")
        
        sim_qpos = data.qpos
        sim_qvel = data.qvel
        ref_qpos, ref_qvel = ref.get_reference_trajectories()
        
        sim_rpos = inverse_convert_ref_traj_qpos(sim_qpos)
        sim_rvel = inverse_convert_ref_traj_qvel(sim_qvel, sim_rpos)
        
        angles_sim = sim_rpos[3:]
        angles_ref = ref_qpos[3:]
        angvel_sim = sim_rvel[3:]
        angvel_ref = ref_qvel[3:]
        
        if angles_sim.ndim != 1 or angles_sim.shape != angles_ref.shape:
            raise ValueError(f"Angle shape mismatch: {angles_sim.shape} vs {angles_ref.shape}")
        if angvel_sim.ndim != 1 or angvel_sim.shape != angvel_ref.shape:
            raise ValueError(f"AngVel shape mismatch: {angvel_sim.shape} vs {angvel_ref.shape}")
        
        num_j = angles_sim.size
        
        coeffs = getattr(env, "reward_coefficients", {})
        decay = coeffs.get("joint_weight_decay", None)
        frac  = float(coeffs.get("joint_decay_fraction", 0.2))
        w, w_sum = _build_joint_weights(num_j, decay, frac)
        if w_sum <= 0.0:
            raise ValueError("Sum of joint weights must be positive")
        
        delta_ang   = angles_sim - angles_ref
        delta_ang   = np.arctan2(np.sin(delta_ang), np.cos(delta_ang))
        delta_vel   = angvel_sim - angvel_ref
        
        angle_mse   = _weighted_mse(delta_ang, w, w_sum)
        angvel_mse  = _weighted_mse(delta_vel, w, w_sum)
        
        k_ang       = float(coeffs.get("joint_angle",   20.0))
        k_vel       = float(coeffs.get("joint_angvel",   0.05))
        EPS         = getattr(env, "EPSILON", 1e-8)
        
        r_angle  = math.exp(-k_ang * angle_mse)
        r_angvel = math.exp(-k_vel * angvel_mse)
        if r_angle  < EPS: r_angle  = 0.0
        if r_angvel < EPS: r_angvel = 0.0
        
        info = {
            "angle_mse":  float(angle_mse),
            "angvel_mse": float(angvel_mse),
            "num_joints": num_j,
            "weight_sum": w_sum
        }
        return r_angle, r_angvel, info

    except Exception as e:
        env.logger.error("get_joint_tracking_reward error: %s", e)
        return 0.0, 0.0, {"error": str(e)}

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
    





        