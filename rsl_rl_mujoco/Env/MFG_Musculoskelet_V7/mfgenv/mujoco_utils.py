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

def load_mujoco_model(model_path: str) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Load a MuJoCo model from an XML file and create an associated data object.

    This function performs several validations:
        - Checks file existence.
        - Parses the XML file.
        - Ensures the model contains at least one joint and one body.

    Args:
        model_path (str): Path to the MuJoCo XML model file.

    Returns:
        tuple: (mjModel, mjData), where mjModel is the MuJoCo model object and mjData is the simulation data.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model has missing components.
        RuntimeError: If loading the model fails due to other errors.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")

    try:
        # Load the MuJoCo model
        model = mujoco.MjModel.from_xml_path(model_path)
        if model is None:
            raise ValueError("Model is None after loading. Check the XML file for syntax errors.")

        # Create the MuJoCo data object
        data = mujoco.MjData(model)
        if data is None:
            raise ValueError("Data object could not be initialized. Check the XML file.")

        # Validate model structure
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
    Parse actuator parameters from a MuJoCo XML model file.

    This function extracts:
        - 'kp': The position controller gain value provided in the actuator definition.
        - 'forcerange': The force limits of the actuator, obtained from the corresponding joint definition's 'actuatorfrcrange' attribute.
        - 'limited': The joint's limited flag (True if limited, else False) from the corresponding joint.
        - 'range': The joint's movement range (angle range) from the joint's 'range' attribute.

    Args:
        model_path (str): Path to the MuJoCo XML model file.

    Returns:
        dict: Mapping from actuator names to a dictionary with keys 'kp' and 'forcerange'.

    Raises:
        FileNotFoundError: If the XML file does not exist.
        ValueError: If required attributes are missing or invalid.
        ET.ParseError: If the XML file cannot be parsed.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MuJoCo XML model file not found: {model_path}")
    
    try:
        tree = ET.parse(model_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file: {e}")
    
    actuator_params: Dict[str, Dict[str, Any]] = {}
    # Find all position actuator elements under the <actuator> tag
    position_actuators = root.findall(".//actuator/position")
    if not position_actuators:
        logger.warning("No position actuators found in the XML file. Ensure the XML defines actuators using <position> tags under <actuator>.")
            
    for actuator in position_actuators:
        actuator_name = actuator.get("name")
        if not actuator_name:
            raise ValueError("A position actuator is missing the 'name' attribute.")
            
        joint_name = actuator.get("joint")
        if not joint_name:
            raise ValueError(f"Actuator '{actuator_name}' is missing the 'joint' attribute.")
            
        kp_str = actuator.get("kp")
        if kp_str is None:
            raise ValueError(f"Actuator '{actuator_name}' is missing the 'kp' attribute.")
        try:
            kp_value = float(kp_str)
        except ValueError:
            raise ValueError(f"Invalid kp value for actuator '{actuator_name}': {kp_str}")
        
        # Locate the corresponding joint element by matching the 'name' attribute
        joint_elem = root.find(f".//joint[@name='{joint_name}']")
        if joint_elem is None:
            raise ValueError(f"No joint definition found for actuator '{actuator_name}' with joint '{joint_name}'.")
            
        actuatorfrcrange_str = joint_elem.get("actuatorfrcrange")
        if actuatorfrcrange_str is None:
            raise ValueError(f"Joint '{joint_name}' for actuator '{actuator_name}' is missing the 'actuatorfrcrange' attribute.")
        try:
            forcerange = list(map(float, actuatorfrcrange_str.split()))
            if len(forcerange) != 2:
                raise ValueError(f"'actuatorfrcrange' must have two values for joint '{joint_name}' associated with actuator '{actuator_name}'.")
        except ValueError:
            raise ValueError(f"Invalid 'actuatorfrcrange' format for joint '{joint_name}' associated with actuator '{actuator_name}': {actuatorfrcrange_str}")
            
        limited_str = joint_elem.get("limited", "false")
        limited = limited_str.strip().lower() == "true"
            
        range_str = joint_elem.get("range", "-1 1")
        try:
            range_vals = list(map(float, range_str.split()))
            if len(range_vals) != 2:
                raise ValueError(f"'range' must have two values for joint '{joint_name}'.")
        except ValueError:
            raise ValueError(f"Invalid 'range' format for joint '{joint_name}': {range_str}")
            
        actuator_params[actuator_name] = {
            "kp": kp_value,
            "forcerange": forcerange,
            "limited": limited,
            "range": range_vals
        }
        
    return actuator_params
        
def check_invalid_names(jnt_names: List[str], actuator_names: List[str], verbose: bool = False) -> None:
    """
    Validate that all actuator names exist in joint names and that the extra joints correspond exactly
    to the 6 pelvis degrees of freedom.

    Args:
        jnt_names (List[str]): List of joint names from the model.
        actuator_names (List[str]): List of actuator names.
        verbose (bool): If True, log detailed information.

    Raises:
        ValueError: If validation fails.
    """
    actuator_set = set(actuator_names)
    joint_set = set(jnt_names)
    
    if not actuator_set.issubset(joint_set):
        missing_actuators = actuator_set - joint_set
        raise ValueError(f"The following actuators are missing in joint names: {missing_actuators}")
    
    extra_joints = joint_set - actuator_set

    if len(extra_joints) != 1:
        raise ValueError(f"Expected 1 extra 'freejoint' DoFs, but found {len(extra_joints)}")

    if verbose:
        logger.info("Joint names correctly contain actuator names. Extra 1 DoF: %s", extra_joints)

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
        if 'floating' in jnt_name.lower():
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
                    