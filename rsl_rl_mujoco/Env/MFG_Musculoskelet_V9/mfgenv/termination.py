"""
Termination conditions for the musculoskeletal simulation environment.
Determines if the episode should end based on conditions such as falling,
COM Y deviation, excessive pelvis/torso angle deviations and high contact forces.
@author: YAKE
"""

import numpy as np
import logging
from typing import Any, Tuple, Dict, List

from .state import (
    get_site_kinematics,
    get_pelvis_kinematics,
    compute_ref_site_kinematics,
    compute_ref_pelvis_kinematics,
    )
from .common_utils import inverse_convert_ref_traj_qpos

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def check_termination(
        env: Any, 
        conditions: List[str] = None
        ) -> Tuple[bool, Dict[str, bool]]:
    """
    Determine whether the episode should terminate early based on selected conditions.

    Parameters
    ----------
    env : Any
        The environment instance.
    conditions : list of str, optional
        Which conditions to check. Supported keys:
          - "has_fallen"
          - "site_deviation_exceeded"
        Defaults to all.

    Returns
    -------
    terminated : bool
        True if any condition is met.
    details : dict
        Mapping each condition name to either:
          - False             : condition not met
          - True              : condition met (no extra info)
          - dict of details   : for site_deviation_exceeded, site→distance
    """
    available = {
        "has_fallen": lambda: has_fallen(env),
        "site_deviation_exceeded": lambda: is_site_deviation_exceeded(env),
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
            # conservatively terminate on error
            details[cond] = True

    terminated = any(
        (v if isinstance(v, bool) else bool(v))
        for v in details.values()
    )
    return terminated, details


def has_fallen(
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
        if z < pelvis_height_range[0] or z > pelvis_height_range[1]:
            logger.debug("has_fallen: pelvis height %.3f outside %s", z, pelvis_height_range)
            return True
        return False
    except Exception as e:
        logger.error("has_fallen error: %s", e)
        return True


def is_site_deviation_exceeded(
    env: Any,
    threshold: float = 0.15
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if any site (and pelvis if env.relative_pelvis=False) deviates by more than threshold.

    Parameters
    ----------
    env : Any
    threshold : float
        Distance [m] above which a deviation is flagged.

    Returns
    -------
    exceeded : bool
    details  : dict of site_name -> distance for those exceeding threshold
    """
    try:
        # simulated joint sites
        _, sim_comp = get_site_kinematics(env)
        sim_pos = sim_comp.get("pos", {})
        if not sim_pos:
            return False, {}

        # optionally include pelvis in world-frame
        if not getattr(env, "relative_pelvis", False):
            _, pel_comp = get_pelvis_kinematics(env, use_free_joint=False)
            sim_pos["pelvis"] = pel_comp["pos"]

        # reference joint sites
        ref_kin = compute_ref_site_kinematics(env)
        ref_pos = ref_kin.get("pos", {})
        if not ref_pos:
            return False, {}

        # optionally include reference pelvis
        if not getattr(env, "relative_pelvis", False):
            ref_pel = compute_ref_pelvis_kinematics(env, use_free_joint=False)
            ref_pos["pelvis"] = ref_pel["pos"]

        # common sites
        sites = list(sim_pos.keys() & ref_pos.keys())
        if not sites:
            return False, {}

        # vectorized distance computation
        sim_arr = np.stack([sim_pos[s] for s in sites], axis=0)  # (N,3)
        ref_arr = np.stack([ref_pos[s] for s in sites], axis=0)  # (N,3)
        dists   = np.linalg.norm(sim_arr - ref_arr, axis=1)      # (N,)

        exceeded = {
            sites[i]: float(d) for i, d in enumerate(dists) if d > threshold
        }
        return bool(exceeded), exceeded

    except Exception as e:
        logger.error("is_site_deviation_exceeded error: %s", e)
        return True, {}