"""
Utilities for loading and managing MuJoCo models.
@author: YAKE
"""

import os
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
import logging
from typing import Any, Dict, List, Optional, Tuple

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add a default StreamHandler if no handler exists.
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def load_mujoco_model(
        model_path: str,
        use_muscle: bool = False
        ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Load a MuJoCo model from XML, removing either:
      - all group=2 <position> actuators (if use_muscle=True), or
      - all group=3 <general class="ZXYmuscle"> actuators (if use_muscle=False).

    Parameters
    ----------
    model_path : str
        Path to the MuJoCo XML model file.
    use_muscle : bool, default=False
        If False: keep group=1 & group=2 (position actuators), drop group=3 (muscles).
        If True:  keep group=1 & group=3 (muscles), drop group=2 (low_position).

    Returns
    -------
    model : mujoco.MjModel
        The filtered MuJoCo model.
    data : mujoco.MjData
        Simulation data initialized for the model.

    Raises
    ------
    FileNotFoundError
        If model_path does not exist.
    ValueError
        If the filtered model has zero joints or bodies.
    RuntimeError
        If parsing or loading fails.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")

    try:
        spec = mujoco.MjSpec.from_file(model_path)
        
        keep = {1, 3} if use_muscle else {1, 2}
        for act in list(spec.actuators):
            try:
                grp = int(act.group)
            except Exception:
                continue
            if grp not in keep:
                act.delete()
        
        model = spec.compile()
        if model is None:
            raise ValueError("Model is None after loading. Check the XML file for syntax errors.")

        data = mujoco.MjData(model)
        if data is None:
            raise ValueError("Data object could not be initialized. Check the XML file.")

        if model.njnt == 0:
            raise ValueError("Model contains no joints. Ensure the XML file defines joints correctly.")
        if model.nbody == 0:
            raise ValueError("Model contains no bodies. Ensure at least one rigid body is defined.")

        logger.debug("Successfully loaded MuJoCo model from %s", model_path)
        return model, data

    except mujoco.FatalError as e:
        raise RuntimeError(f"MuJoCo Fatal Error: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load MuJoCo model from {model_path}: {e}")
        
def parse_actuator_prm_from_xml(model_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse actuator parameters from a MuJoCo XML model file, handling both position and muscle actuators.

    Returns a dict mapping actuator name → dict of parameters:
      - For position actuators (class="low_position" or "up_position"):
          { "type": "position",
            "kp": float,
            "forcerange": [float, float],
            "limited": bool,
            "range": [float, float] }

      - For muscle actuators (class="ZXYmuscle"):
          { "type": "muscle",
            "tendon": str,
            "ctrlrange": [float, float],
            "lengthrange": [float, float],
            "gainprm": List[float],
            "biasprm": List[float] }

    If a muscle <general> element lacks a 'ctrlrange', this function falls back
    to the <default class="ZXYmuscle"><general ctrlrange=... /></default> definition.

    Raises:
        FileNotFoundError: If model_path does not exist.
        ValueError: If required attributes are missing or invalid.
        ET.ParseError: If the XML cannot be parsed.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MuJoCo XML model file not found: {model_path}")

    try:
        tree = ET.parse(model_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file: {e}")

    actuator_params: Dict[str, Dict[str, Any]] = {}

    # Locate default <general> for ZXYmuscle (to obtain fallback ctrlrange if missing)
    default_muscle_elem = root.find(".//default[@class='ZXYmuscle']/general")
    default_ctrlrange = None
    if default_muscle_elem is not None:
        ctrl_str = default_muscle_elem.get("ctrlrange")
        if ctrl_str:
            try:
                default_ctrlrange = list(map(float, ctrl_str.split()))
                if len(default_ctrlrange) != 2:
                    default_ctrlrange = None
            except ValueError:
                default_ctrlrange = None

    # 1) Parse <actuator><position .../> elements
    for pos in root.findall(".//actuator/position"):
        act_name = pos.get("name")
        if not act_name:
            raise ValueError("A <position> actuator is missing the 'name' attribute.")

        joint_name = pos.get("joint")
        if not joint_name:
            raise ValueError(f"Actuator '{act_name}' missing 'joint' attribute.")

        kp_str = pos.get("kp")
        if kp_str is None:
            raise ValueError(f"Actuator '{act_name}' missing 'kp' attribute.")
        try:
            kp_val = float(kp_str)
        except ValueError:
            raise ValueError(f"Invalid kp for actuator '{act_name}': '{kp_str}'")

        joint_elem = root.find(f".//joint[@name='{joint_name}']")
        if joint_elem is None:
            raise ValueError(f"No <joint name='{joint_name}'> found for actuator '{act_name}'.")

        # 'actuatorfrcrange' on joint
        afr_str = joint_elem.get("actuatorfrcrange")
        if afr_str is None:
            raise ValueError(f"Joint '{joint_name}' for actuator '{act_name}' missing 'actuatorfrcrange'.")
        try:
            afr_vals = list(map(float, afr_str.split()))
            if len(afr_vals) != 2:
                raise ValueError
        except ValueError:
            raise ValueError(
                f"Invalid 'actuatorfrcrange' for joint '{joint_name}' of actuator '{act_name}': '{afr_str}'"
            )

        limited_str = joint_elem.get("limited", "false")
        limited = limited_str.strip().lower() == "true"

        range_str = joint_elem.get("range", "-1 1")
        try:
            rng_vals = list(map(float, range_str.split()))
            if len(rng_vals) != 2:
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid 'range' for joint '{joint_name}': '{range_str}'")

        actuator_params[act_name] = {
            "type": "position",
            "kp": kp_val,
            "forcerange": afr_vals,
            "limited": limited,
            "range": rng_vals
        }

    # 2) Parse <actuator><general .../> elements of class="ZXYmuscle"
    for gen in root.findall(".//actuator/general"):
        cls = gen.get("class", "")
        if cls.strip() != "ZXYmuscle":
            continue

        act_name = gen.get("name")
        if not act_name:
            raise ValueError("A <general> actuator of class 'ZXYmuscle' is missing 'name'.")

        tendon_name = gen.get("tendon")
        if not tendon_name:
            raise ValueError(f"Muscle actuator '{act_name}' missing 'tendon' attribute.")

        # ctrlrange: if absent, fallback to default_muscle_elem
        ctrl_str = gen.get("ctrlrange")
        if ctrl_str is None:
            if default_ctrlrange is None:
                raise ValueError(f"Muscle actuator '{act_name}' missing 'ctrlrange', and no default found.")
            ctrl_vals = default_ctrlrange
        else:
            try:
                ctrl_vals = list(map(float, ctrl_str.split()))
                if len(ctrl_vals) != 2:
                    raise ValueError
            except ValueError:
                raise ValueError(f"Invalid 'ctrlrange' for muscle '{act_name}': '{ctrl_str}'")

        # lengthrange
        length_str = gen.get("lengthrange")
        if length_str is None:
            raise ValueError(f"Muscle actuator '{act_name}' missing 'lengthrange'.")
        try:
            length_vals = list(map(float, length_str.split()))
            if len(length_vals) != 2:
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid 'lengthrange' for muscle '{act_name}': '{length_str}'")

        # gainprm
        gain_str = gen.get("gainprm")
        if gain_str is None:
            raise ValueError(f"Muscle actuator '{act_name}' missing 'gainprm'.")
        try:
            gain_vals = list(map(float, gain_str.split()))
        except ValueError:
            raise ValueError(f"Invalid 'gainprm' for muscle '{act_name}': '{gain_str}'")

        # biasprm
        bias_str = gen.get("biasprm")
        if bias_str is None:
            raise ValueError(f"Muscle actuator '{act_name}' missing 'biasprm'.")
        try:
            bias_vals = list(map(float, bias_str.split()))
        except ValueError:
            raise ValueError(f"Invalid 'biasprm' for muscle '{act_name}': '{bias_str}'")

        actuator_params[act_name] = {
            "type": "muscle",
            "tendon": tendon_name,
            "ctrlrange": ctrl_vals,
            "lengthrange": length_vals,
            "gainprm": gain_vals,
            "biasprm": bias_vals
        }

    return actuator_params
        
def check_invalid_names(
    jnt_names: List[str],
    actuator_names: List[str],
    use_muscle: bool = False,
    verbose: bool = False
) -> None:
    """
    Validate that:
      1) All position‐based actuators drive a valid joint name.
      2) If not using muscles, the remaining joints correspond to the free‐joint DoFs
         (should be exactly 1 or 7).

    Muscle actuators (whose names don’t appear in `jnt_names`) are always skipped
    from joint‐name validation.

    Args:
        jnt_names (List[str]):
            Names of all joints in the model.
        actuator_names (List[str]):
            Names of all actuators in the model (position + muscle).
        use_muscle (bool):
            If True, skip the “free‐joint count” check since muscles may drive
            those joints. If False, enforce that the number of undriven joints is 1 or 7.
        verbose (bool):
            If True, log detailed info.

    Raises:
        ValueError:
            - A position‐based actuator name is missing from `jnt_names`.
            - (when use_muscle=False) The count of extra (undriven) joints isn’t 1 or 7.
    """
    jnt_set = set(jnt_names)

    # 1) Separate actuators:
    pos_acts = [a for a in actuator_names if a in jnt_set]
    muscle_acts = [a for a in actuator_names if a not in jnt_set]

    # 2) Ensure all position‐based actuators map to a joint
    missing = [a for a in pos_acts if a not in jnt_set]
    if missing:
        raise ValueError(f"Position actuators missing in joint list: {missing}")

    # 3) If not using muscles, enforce free‐joint count
    if not use_muscle:
        driven = set(pos_acts)
        extra_joints = jnt_set - driven
        if len(extra_joints) not in (1, 7):
            raise ValueError(
                f"Expected 1 or 7 free‐joint DoFs, but found {len(extra_joints)}: {sorted(extra_joints)}"
            )

    # 4) Optionally log details
    if verbose:
        logger.info("Position actuators: %s", pos_acts)
        logger.info("Muscle actuators (skipped): %s", muscle_acts)
        if not use_muscle:
            logger.info("Extra joints (free‐joint DoFs): %s", sorted(extra_joints))
        
def check_and_enable_Limit(model: mujoco.MjModel, jnt_names: List[str], actuator_names: List[str], actuator_prm: Dict[str, Any], default_forcerange: np.ndarray = np.array([-100, 100])) -> None:
    """
    Enable force and control limits for actuators and joints in the MuJoCo model.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        jnt_names (List[str]): List of joint names.
        actuator_names (List[str]): List of actuator names.
        actuator_prm (dict): Dictionary of actuator parameters.
        default_forcerange (np.ndarray, optional): Default force range if not specified.

    Raises:
        ValueError: If actuator parameters are invalid.
    """
    if not isinstance(jnt_names, list):
        raise ValueError("jnt_names must be a list.")
    if not isinstance(actuator_names, list):
        raise ValueError("actuator_names must be a list.")
    if not isinstance(actuator_prm, dict):
        raise ValueError("actuator_prm must be a dictionary.")
    
    for i, jnt_name in enumerate(jnt_names):
        # Skip pelvis joints
        if 'floating' in jnt_name.lower() or 'beta' in jnt_name.lower():
            continue
        model.jnt_actfrclimited[i] = 1  # Enable joint force limits
        
        actuator_idx = actuator_names.index(jnt_name) if jnt_name in actuator_names else None
        if actuator_idx is not None:
            if model.actuator_ctrllimited[actuator_idx] != 1:
                model.actuator_ctrllimited[actuator_idx] = 1
            if not np.array_equal(model.actuator_ctrlrange[actuator_idx], actuator_prm[jnt_name]['range']):
                model.actuator_ctrlrange[actuator_idx] = actuator_prm[jnt_name]['range']
            # if model.actuator_forcelimited[actuator_idx] != 1:
            #     model.actuator_forcelimited[actuator_idx] = 1
            # forcerange = actuator_prm[jnt_name].get("forcerange")      
            # if forcerange is not None and not np.array_equal(model.actuator_forcerange[actuator_idx], forcerange):
            #     model.actuator_forcerange[actuator_idx] = forcerange
        
        if jnt_name in actuator_prm:
            # gear = actuator_prm[jnt_name].get("gear")
            forcerange = actuator_prm[jnt_name].get("forcerange")
            if not isinstance(forcerange, (list, tuple)) or len(forcerange) != 2:
                raise ValueError(f"Invalid forcerange for actuator '{jnt_name}': {forcerange}")
            # if not isinstance(gear, (int, float)):
            #     raise ValueError(f"Invalid gear value for actuator '{jnt_name}': {gear}")
            # scaled_range = np.array(forcerange) * gear
            model.jnt_actfrcrange[i] = [min(forcerange), max(forcerange)]
        else:
            model.jnt_actfrcrange[i] = default_forcerange
    
    logger.info("Joint and actuator force limits successfully enabled.")
 
def reset_mujoco_state(model: mujoco.MjModel, data: mujoco.MjData, qpos_init: Optional[np.ndarray] = None, qvel_init: Optional[np.ndarray] = None, randomize: bool = False, noise_std: float = 0.005) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Reset the MuJoCo simulation state to its initial conditions.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The simulation data.
        qpos_init (np.ndarray, optional): Initial joint positions.
        qvel_init (np.ndarray, optional): Initial joint velocities.
        randomize (bool): If True, adds Gaussian noise.
        noise_std (float): Standard deviation of noise.

    Returns:
        tuple: (model, data) after resetting.

    Raises:
        ValueError: If dimensions of qpos_init or qvel_init do not match.
        RuntimeError: If NaN values are detected after reset.
    """
    mujoco.mj_resetData(model, data)
    
    if qpos_init is not None and qpos_init.shape[0] != model.nq:
        raise ValueError(f"qpos_init shape mismatch: expected {model.nq}, got {qpos_init.shape[0]}")
    if qvel_init is not None and qvel_init.shape[0] != model.nv:
        raise ValueError(f"qvel_init shape mismatch: expected {model.nv}, got {qvel_init.shape[0]}")
    
    if qpos_init is not None:
        data.qpos[:] = qpos_init
        if randomize:
            data.qpos += np.random.normal(0, noise_std, size=data.qpos.shape)
        
    if qvel_init is not None:
        data.qvel[:] = qvel_init
        if randomize:
            data.qvel += np.random.normal(0, noise_std, size=data.qvel.shape)
    
    mujoco.mj_forward(model, data)
    
    if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
        raise RuntimeError("Reset resulted in NaN values! The model initialization may be incorrect.")

    return model, data
    
def step_mujoco(model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray) -> None:
    """
    Apply an action to the actuators and advance the simulation.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The simulation data.
        action (np.ndarray): Action vector applied to actuators.

    Raises:
        ValueError: If action shape is incorrect or values are out of bounds.
        RuntimeError: If simulation state contains NaN values.
    """
    if action.shape[0] != model.nu:
        raise ValueError(f"Action shape mismatch: expected {model.nu}, got {action.shape[0]}")

    action_lower, action_upper = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    if np.any(action < action_lower) or np.any(action > action_upper):
        raise ValueError(f"Action out of bounds: expected range {model.actuator_ctrlrange}, got {action}")

    data.ctrl[:] = action
    mujoco.mj_step(model, data)

    if np.isnan(data.qpos).any() or np.isnan(data.qvel).any() or np.isnan(data.qacc).any():
        raise RuntimeError("Simulation state contains NaN values! The model may be unstable.")

    max_actuator_force = np.max(np.abs(data.qfrc_actuator))
    if max_actuator_force > 1e4:
        logger.warning("Unusually high actuator forces detected: %.2e. Consider adjusting model parameters.", max_actuator_force)

def sync_mujoco_with_state(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, qvel: Optional[np.ndarray] = None) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Synchronize the simulation state with provided joint positions and velocities.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The simulation data.
        qpos (np.ndarray): New joint positions.
        qvel (np.ndarray, optional): New joint velocities.

    Returns:
        tuple: (model, data) after synchronization.

    Raises:
        ValueError: If qpos or qvel dimensions do not match.
        RuntimeError: If synchronization leads to NaN values.
    """
    if qpos.shape[0] != model.nq:
        raise ValueError(f"qpos shape mismatch: expected {model.nq}, got {qpos.shape[0]}")
    if qvel is not None and qvel.shape[0] != model.nv:
        raise ValueError(f"qvel shape mismatch: expected {model.nv}, got {qvel.shape[0]}")
        
    data.qpos[:] = qpos
    data.qvel[:] = qvel if qvel is not None else np.zeros(model.nv)
    mujoco.mj_forward(model, data)
    
    if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
        raise RuntimeError("Synchronization resulted in NaN values! Check the input state.")

    return model, data
                    