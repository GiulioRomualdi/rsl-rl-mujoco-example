"""
Termination conditions for the musculoskeletal simulation environment.
Determines if the episode should end based on conditions such as falling,
COM Y deviation, excessive pelvis/torso angle deviations and high contact forces.
@author: YAKE
"""

import math
import numpy as np
import logging
from typing import Any, Dict, Tuple, List

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def check_termination(
        env: Any, 
        obs_components: Dict[str, Any],
        conditions: List[str] = None
        ) -> Tuple[bool, Dict[str, bool]]:
    """
    Determine whether the episode should terminate early based on selected conditions.

    Parameters
    ----------
    env : Any
        The environment instance.
    obs_components : dict
        Must contain:
          - "joint":   simulated joint-site kinematics dict
          - "pelvis":  simulated pelvis kinematics dict
          - "ref_kin": reference joint-site kinematics dict
          - "ref_pel": reference pelvis kinematics dict
    conditions : list of str, optional
        Which conditions to check. Supported keys:
          - "has_fallen"
          - "site_deviation_exceeded"
        Defaults to both.

    Returns
    -------
    terminated : bool
        True if any condition is met.
    details : dict
        Mapping each condition to either:
          - False             : condition not met
          - True              : condition met without details
          - dict of details   : e.g. for "site_deviation_exceeded", site→distance
    """
    available = {
        "has_fallen": lambda: _has_fallen(env),
        "site_deviation_exceeded": lambda: _site_deviation_exceeded(
            obs_components,
            threshold=getattr(env, "site_threshold", 0.15)
        ),
    }
    if conditions is None:
        conditions = list(available.keys())

    invalid = set(conditions) - set(available.keys())
    if invalid:
        raise ValueError(f"Invalid termination conditions: {invalid}")

    details: Dict[str, Any] = {}
    for cond in conditions:
        try:
            result = available[cond]()
            if cond == "site_deviation_exceeded":
                exceeded, info = result
                details[cond] = info if exceeded else False
            else:
                details[cond] = bool(result)
        except Exception as e:
            logger.error("Error evaluating '%s': %s", cond, e)
            # On error, conservatively terminate
            details[cond] = True

    # if any condition is True or non-empty dict, we terminate
    terminated = any(
        (v if isinstance(v, bool) else bool(v))
        for v in details.values()
    )
    return terminated, details

def _has_fallen(
    env: Any,
    pelvis_height_range: Tuple[float, float] = (0.6, 1.2)
    ) -> bool:
    """
    Check if the agent has fallen by verifying pelvis height is out of allowed range.

    Parameters
    ----------
    env : Any
    pelvis_height_range : (min, max) allowed z-height for pelvis

    Returns
    -------
    bool : True if fallen or on error, False otherwise.
    """
    try:
        bid = env.model.body("pelvis").id
        z = float(env.data.xpos[bid, 2])
        min_z, max_z = pelvis_height_range
        if z < min_z or z > max_z:
            logger.debug("has_fallen: pelvis z=%.3f outside [%s, %s]", z, min_z, max_z)
            return True
        return False
    except Exception as e:
        logger.error("has_fallen error: %s", e)
        return True

def _site_deviation_exceeded(
    obs_components: Dict[str, Any],
    threshold: float = 0.15
    ) -> Tuple[bool, Dict[str, float]]:
    """
    Check if any body‐site deviates from reference by more than threshold.

    Parameters
    ----------
    obs_components : dict
        Must contain "joint","pelvis","ref_kin","ref_pel".
    threshold : float
        Distance (m) above which a deviation is flagged.

    Returns
    -------
    exceeded : bool
    details  : dict of site_name -> distance
    """
    try:
        sim_kin = obs_components["joint"]
        ref_kin = obs_components["ref_kin"]
        sim_pel = obs_components["pelvis"]
        ref_pel = obs_components["ref_pel"]
        
        sim_pos = dict(sim_kin["pos"])
        ref_pos = dict(ref_kin["pos"])
        sim_pos["pelvis"] = sim_pel["pos"]
        ref_pos["pelvis"] = ref_pel["pos"]
        
        sites = sorted(sim_pos.keys() & ref_pos.keys())
        if not sites:
            return False, {}

        sim_arr = np.vstack([sim_pos[s] for s in sites])
        ref_arr = np.vstack([ref_pos[s] for s in sites])
        dists   = np.linalg.norm(sim_arr - ref_arr, axis=1)

        # collect which exceed
        exceeded = {
            sites[i]: float(d)
            for i, d in enumerate(dists)
            if d > threshold
        }
        return bool(exceeded), exceeded

    except Exception as e:
        logger.error("is_site_deviation_exceeded error: %s", e)
        # On error, conservatively terminate
        return True, {}