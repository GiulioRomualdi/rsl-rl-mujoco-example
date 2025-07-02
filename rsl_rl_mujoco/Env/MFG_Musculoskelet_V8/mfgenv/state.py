"""
State-related utilities for musculoskeletal simulation environment.
Handles state extraction, normalization, and statistics collection.
@author: YAKE
"""

import numpy as np
import mujoco
from .common_utils import inverse_convert_ref_traj_qpos, inverse_convert_ref_traj_qvel
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def compute_grf(body_id: int, 
                geom_ids: List[int], 
                env: Any, 
                use_body_frame: bool, 
                COM_offset: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ground reaction force (GRF) for a specified foot.

    Parameters
    ----------
    body_id : int
        ID of the reference body (e.g., heel).
    geom_ids : list[int]
        Geometry IDs belonging to the foot.
    env : Any
        MuJoCo environment with model and data attributes.
    use_body_frame : bool
        If True, use data.xpos; else data.xipos for reference.
    COM_offset : np.ndarray
        3-element array to subtract from the GRF position.

    Returns
    -------
    GRF_pos : np.ndarray
        3D contact position after COM adjustment.
    total_force : np.ndarray
        3D net contact force.
    total_torque : np.ndarray
        3D net torque about reference.

    Raises
    ------
    ValueError
        On invalid input shapes or missing data.
    """
    data = env.data
    
    # Retrieve reference position safely
    try:
        ref_pos = data.xpos[body_id] if use_body_frame else data.xipos[body_id]
    except (IndexError, AttributeError) as e:
        raise ValueError(f"Failed to get reference position for body {body_id}: {e}")
    if ref_pos.shape != (3,):
        raise ValueError(f"Reference position must be shape (3,), got {ref_pos.shape}")
    
    # Validate COM_offset.
    COM_offset = np.asarray(COM_offset)
    if COM_offset.shape != (3,):
        raise ValueError(f"COM_offset must be length 3, got {COM_offset.shape}")
    
    # Initialize accumulators
    total_force = np.zeros(3)
    total_torque = np.zeros(3)
    weighted_pos = np.zeros(3)
    geom_set = set(geom_ids)
    temp_force = np.zeros(6)
    
    # Iterate contacts
    for i in range(data.ncon):
        if data.contact.geom2[i] not in geom_set or data.contact.geom1[i] != 0:
            continue
        try:
            mujoco.mj_contactForce(env.model, data, i, temp_force)
        except RuntimeError as e:
            logger.error(f"Contact force error at index {i}: {e}")
            continue
        local_force = temp_force[:3]
        frame_flat = data.contact.frame[i]
        if frame_flat.size != 9:
            raise ValueError(f"Contact frame size invalid at index {i}")
        frame = frame_flat.reshape(3, 3)
        global_force = frame.T @ local_force
        pos_i = data.contact.pos[i]
        torque_i = np.cross(pos_i - ref_pos, global_force)
        total_force += global_force
        total_torque += torque_i
        weighted_pos += pos_i * global_force
        
    if not np.any(total_force):
        return np.zeros(3), np.zeros(3), np.zeros(3)

    GRF_pos = weighted_pos / total_force
    GRF_pos = GRF_pos - COM_offset

    return GRF_pos, total_force, total_torque

def get_GRF_info(env: Any, 
                 use_body_frame: bool = True, 
                 relative_COM: bool = True
                 ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute GRF information for both feet and return flat array with details.

    Parameters
    ----------
    env : Any
        MuJoCo environment with model and data.
    use_body_frame : bool
        If True, reference positions from data.xpos; else data.xipos.
    relative_COM : bool
        If True, subtract root COM (data.subtree_com[0]) from GRF positions.

    Returns
    -------
    concatenated : np.ndarray
        18-element array [r_pos(3), r_force(3), r_torque(3),
                        l_pos(3), l_force(3), l_torque(3)].
    info : dict
        Dictionary with 'right' and 'left' keys, each containing a dict
        with 'GRF_pos', 'GRF_force', 'GRF_torque'.

    Raises
    ------
    ValueError
        If body or geometry IDs cannot be retrieved.
    """
    data = env.data
    
    # COM offset
    if relative_COM:
        try:
            COM_offset = data.subtree_com[0].copy()
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Failed to get COM offset: {e}")
    else:
        COM_offset = np.zeros(3)
    
    # Early exit when no contacts
    if data.ncon == 0:
        zeros = np.zeros(3)
        concat = np.zeros(18)
        info = {'right': {'GRF_pos': zeros, 'GRF_force': zeros, 'GRF_torque': zeros},
                'left':  {'GRF_pos': zeros, 'GRF_force': zeros, 'GRF_torque': zeros}}
        return concat, info
    
    # Retrieve body and geometry IDs
    try:
        r_body = env.model.body('calcn_r').id
        l_body = env.model.body('calcn_l').id
        r_geoms = [env.model.geom(name).id for name in
                   ['C_r_foot1','C_r_foot3','C_r_foot4','C_r_bofoot1','C_r_bofoot2']]
        l_geoms = [env.model.geom(name).id for name in
                   ['C_l_foot1','C_l_foot3','C_l_foot4','C_l_bofoot1','C_l_bofoot2']]
    except Exception as e:
        raise ValueError(f"Failed to get foot IDs: {e}")
    
    # Compute for each foot
    r_pos, r_force, r_torque = compute_grf(r_body, r_geoms, env, use_body_frame, COM_offset)
    l_pos, l_force, l_torque = compute_grf(l_body, l_geoms, env, use_body_frame, COM_offset)

    concatenated = np.hstack((r_pos, r_force, r_torque, l_pos, l_force, l_torque))
    info = {
        'right': {'GRF_pos': r_pos, 'GRF_force': r_force, 'GRF_torque': r_torque},
        'left':  {'GRF_pos': l_pos, 'GRF_force': l_force, 'GRF_torque': l_torque}
    }
    return concatenated, info

def normalize_grf(
        grf_info: np.ndarray, 
        total_mass: float = 75.1646
        ) -> np.ndarray:
    """
    Normalize GRF force and torque components without altering contact positions.

    Parameters
    ----------
    grf_info : np.ndarray
        18-element GRF vector [r_pos(3), r_force(3), r_torque(3),
                                l_pos(3), l_force(3), l_torque(3)].
    total_mass : float, optional
        Total mass of the model for force normalization (kg), by default 75.1646.

    Returns
    -------
    normalized_grf : np.ndarray
        18-element array with:
            - Positions unchanged,
            - Forces divided by (total_mass * 9.81),
            - Torques divided by torque_max and clipped to [-1, 1].

    Raises
    ------
    ValueError
        If grf_info does not have shape (18,).
    """
    grf = np.asarray(grf_info, dtype=np.float64)
    if grf.shape != (18,):
        raise ValueError(f"Expected GRF array of shape (18,), got {grf.shape}")

    # Normalization factors
    g = 9.81
    force_scale = max(total_mass, 1e-6) * g
    torque_max = 100.0
    
    normalized = grf.copy()
    # Right foot: indices 3-5 (force), 6-8 (torque)
    normalized[3:6] /= force_scale
    normalized[6:9] = np.clip(normalized[6:9] / torque_max, -1.0, 1.0)
    # Left foot: indices 12-14 (force), 15-17 (torque)
    normalized[12:15] /= force_scale
    normalized[15:18] = np.clip(normalized[15:18] / torque_max, -1.0, 1.0)

    return np.nan_to_num(normalized)

def get_pelvis_kinematics(env: Any) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Retrieve pelvis kinematics in world frame and Tait-Bryan (tilt-list-rotation) angles.

    Parameters
    ----------
    env : Any
        MuJoCo environment instance providing:
        - data.qpos : array-like with at least 7 elements [tx, ty, tz, tilt, list, rot, ...]
        - data.qvel : array-like with at least 6 elements [vx, vy, vz, tilt_dot, list_dot, rot_dot, ...]
        - Optional attribute remove_x_pos (bool): if True, drop the x-translation.

    Returns
    -------
    pelvis_state : np.ndarray
        1D array concatenating:
        [translation (3 or 2), euler_angles (3), linear_velocity (3), angular_velocity (3)].
    components : dict
        Detailed components:
        {
          'pelvis_trans': np.ndarray,
          'pelvis_rot': np.ndarray,
          'pelvis_lin_vel': np.ndarray,
          'pelvis_ang_vel': np.ndarray
        }

    Raises
    ------
    ValueError
        If qpos/qvel are missing or have insufficient size,
        or if conversion routines fail.
    """
    # Validate presence and size of qpos/qvel
    qpos = getattr(env.data, 'qpos', None)
    qvel = getattr(env.data, 'qvel', None)
    if qpos is None or qvel is None:
        raise ValueError("Missing qpos or qvel in environment data.")
    if qpos.shape[0] < 7 or qvel.shape[0] < 6:
        raise ValueError(f"Expected qpos>=7 and qvel>=6 elements, got {qpos.shape[0]}, {qvel.shape[0]}")
    
    d_qpos = qpos[:7].copy()
    d_qvel = qvel[:6].copy()
    
    # World-frame translation and linear velocity
    trans = d_qpos[:3]
    lin_vel = d_qvel[:3]
    if getattr(env, 'remove_x_pos', False):
        trans = trans[1:]
    
    components = {
        'pelvis_trans': trans,
        'pelvis_lin_vel': lin_vel
    }
    
    # Convert from freejoint quaternion format back to reference Euler angles
    try:
        ref_qpos = inverse_convert_ref_traj_qpos(d_qpos)
    except Exception as e:
        raise ValueError(f"Quaternion conversion failed for qpos: {e}")
    try:
        ref_qvel = inverse_convert_ref_traj_qvel(d_qvel, ref_qpos)
    except Exception as e:
        raise ValueError(f"Angular velocity conversion failed for qvel: {e}")

    euler = ref_qpos[3:6]
    ang_vel = ref_qvel[3:6]
    components.update({
        'pelvis_rot': euler,
        'pelvis_ang_vel': ang_vel
    })
    
    pelvis_state = np.concatenate((trans, euler, lin_vel, ang_vel))
    return pelvis_state, components

def get_joint_kinematics(env: Any, 
                         include_pelvis: bool = False,
                         relative_to_pelvis: bool = False
                         ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Retrieve joint-anchor site kinematics (positions and velocities) and generalized joint state.

    Parameters
    ----------
    env : Any
        MuJoCo environment providing:
        - model: MuJoCo model with sensor/site definitions.
        - data: MuJoCo data with:
            * site_xpos: world-frame site positions
            * site_xmat: site rotation matrices
            * sensordata: sensor measurements
            * qpos: generalized positions
            * qvel: generalized velocities
        - Optional attribute remove_x_pos (bool): if True, drop the first component of 3D vectors.
    include_pelvis : bool, optional
        If False, skip the sensor named "pelvis_sensor" and exclude the root dofs
        (first 7 qpos, 6 qvel) from the returned generalized state.
        Default is False.
    relative_to_pelvis : bool, optional
        If True, subtract pelvis translation/velocity from each site measurement.
        Default is False.

    Returns
    -------
    joint_state : np.ndarray
        Flattened vector concatenating:
        [site_pos_1, site_vel_1, ..., site_pos_N, site_vel_N,
         joint_qpos, joint_qvel]
    components : dict
        {
            'joint_space_pos': dict(site_name -> np.ndarray),
            'joint_lin_vel':   dict(site_name -> np.ndarray),
            'joint_qpos':      np.ndarray,
            'joint_qvel':      np.ndarray
        }

    Raises
    ------
    ValueError
        If no valid velocimeter sensors on sites are found,
        or if qpos/qvel dimensions are insufficient when excluding pelvis.
    """
    model = env.model
    data = env.data
    eps = 1e-6
    
    # Prepare pelvis reference if needed
    if relative_to_pelvis:
        pel_trans = data.qpos[:3].copy()
        pel_vel   = data.qvel[:3].copy()
        if getattr(env, "remove_x_pos", False):
            pel_trans = pel_trans[1:]
    else:
        pel_trans = np.zeros(3)
        pel_vel   = np.zeros(3)

    sensor_entries = []
    for i in range(model.nsensor):
        if (model.sensor_type[i] != mujoco.mjtSensor.mjSENS_VELOCIMETER or
            model.sensor_objtype[i] != mujoco.mjtObj.mjOBJ_SITE):
            continue
        
        site_id = model.sensor_objid[i]
        name = model.site(site_id).name
        if isinstance(name, bytes):
            name = name.decode()
        if not include_pelvis and name == "pelvis_sensor":
            continue
        
        # Position
        pos = data.site_xpos[site_id].copy()
        if getattr(env, "remove_x_pos", False):
            pos = pos[1:]
        if relative_to_pelvis:
            pos -= pel_trans
        pos[np.abs(pos) < eps] = 0.0
        
        # Velocity
        raw = data.sensordata[3*i:3*i+3].copy()
        xmat = data.site_xmat[site_id].copy()
        if xmat.size != 9:
            raise ValueError(f"Site '{name}' rotation matrix has {xmat.size} elements; expected 9.")
        Rmat = xmat.reshape(3,3)
        vel = Rmat @ raw
        if relative_to_pelvis:
            vel -= pel_vel
        vel[np.abs(vel) < eps] = 0.0
        
        sensor_entries.append((name, pos, vel))
    
    if not sensor_entries:
        raise ValueError("No joint site velocimeter sensors found in model.")
    
    # Sort for deterministic ordering
    joint_space_pos = {n: p for n, p, _ in sensor_entries}
    joint_lin_vel   = {n: v for n, _, v in sensor_entries}
    
    pos_arr = np.vstack([p for _,p,_ in sensor_entries])
    vel_arr = np.vstack([v for _,_,v in sensor_entries])
    
    # Generalized qpos/qvel
    qpos_all = data.qpos.copy()
    qvel_all = data.qvel.copy()
    if not include_pelvis:
        if qpos_all.size <= 7 or qvel_all.size <= 6:
            raise ValueError("Cannot exclude pelvis: qpos/qvel too short.")
        qpos_all = qpos_all[7:]
        qvel_all = qvel_all[6:]
    
    # Concatenate full state
    joint_state = np.concatenate([
        pos_arr.ravel(),
        vel_arr.ravel(),
        qpos_all.ravel(),
        qvel_all.ravel()
    ])
    
    components = {
        'joint_space_pos': joint_space_pos,
        'joint_lin_vel':   joint_lin_vel,
        'joint_qpos':      qpos_all,
        'joint_qvel':      qvel_all
    }
    return joint_state, components

def get_traj_info(env: Any, 
                  horizon: Optional[Union[int, List[int]]] = None
                  ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Retrieve future reference trajectory joint positions and velocities.

    Parameters
    ----------
    env : Any
        Environment instance providing:
        - ref_traj.qpos : np.ndarray of shape (n_dofs, traj_frames)
        - ref_traj.qvel : np.ndarray of shape (n_dofs, traj_frames)
        - ref_traj._pos : int, current frame index
        - ref_traj.traj_frames : int, total number of frames
        - ref_traj.increment : int, frame step per call
    horizon : int or list of int, optional
        Frame offsets ahead of current frame to retrieve.
        If None, defaults to [1].

    Returns
    -------
    future_state : np.ndarray
        Flattened array of shape (n_dofs * m * 2,) where m = len(offsets),
        concatenating future_qpos then future_qvel.
    components : dict
        {
            'future_qpos': np.ndarray of shape (n_dofs, m),
            'future_qvel': np.ndarray of shape (n_dofs, m)
        }

    Raises
    ------
    ValueError
        If ref_traj is missing required attributes, or horizon is invalid.
    """
    # Validate ref_traj presence
    if not hasattr(env, 'ref_traj'):
        raise ValueError("Environment has no ref_traj attribute.")
    ref = env.ref_traj
        
    # Validate qpos/qvel arrays
    if not hasattr(ref, 'qpos') or not hasattr(ref, 'qvel'):
        raise ValueError("ref_traj must have 'qpos' and 'qvel' attributes.")
    qpos = ref.qpos
    qvel = ref.qvel
    if qpos.ndim != 2 or qvel.ndim != 2:
        raise ValueError(f"'qpos' and 'qvel' must be 2D arrays; got shapes {qpos.shape} and {qvel.shape}.")
    
    # Retrieve trajectory parameters
    try:
        current = int(ref._pos)
        total_frames = int(ref.traj_frames)
        incr = int(getattr(ref, 'increment', 1))
    except Exception as e:
        raise ValueError(f"Error reading ref_traj attributes: {e}")
    
    # Prepare offsets
    if horizon is None:
        offsets = [1]
    elif isinstance(horizon, int):
        offsets = [horizon]
    elif isinstance(horizon, list):
        offsets = horizon
    else:
        raise ValueError("horizon must be None, an int, or a list of ints.")
        
    # Validate offsets
    for off in offsets:
        if not isinstance(off, int) or off < 0:
            raise ValueError(f"Invalid horizon offset {off}; must be non-negative integer.")

    # Compute future frame indices with wrap-around
    indices = [ (current + off * incr) % total_frames for off in offsets ]
    
    # Extract future trajectories
    try:
        future_qpos = np.array(qpos[:, indices], copy=True)
        future_qvel = np.array(qvel[:, indices], copy=True)
    except Exception as e:
        raise ValueError(f"Error slicing ref_traj data: {e}")

    # Flatten into single state vector
    future_state = np.concatenate([ future_qpos.ravel(), future_qvel.ravel() ])

    components = {
        'future_qpos': future_qpos,
        'future_qvel': future_qvel
    }
    return future_state, components
    
def get_state(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Retrieve the full simulation state by concatenating sub-states and computing contact flags.

    Parameters
    ----------
    env : Any
        MuJoCo environment instance providing:
        - get_pelvis_kinematics()
        - get_joint_kinematics()
        - get_GRF_info()
        - get_traj_info()
        - data: for qpos, qvel if needed
        - Optional attribute remove_x_pos (bool)

    Returns
    -------
    state : np.ndarray
        Flattened array of shape (N,) concatenating:
        [pelvis_state,
         joint_state,
         grf_state,
         future_traj_state,
         foot_contacts (2,)]

    components : dict
        {
            'pelvis': dict,    # output of get_pelvis_kinematics
            'joint': dict,     # output of get_joint_kinematics
            'grf': dict,       # output of get_GRF_info
            'traj': dict,      # output of get_traj_info
            'foot_contacts': np.ndarray(shape=(2,))  # [right_contact, left_contact]
        }

    Raises
    ------
    ValueError
        If any sub-state extraction or concatenation fails.
    """
    # Extract sub-states
    try:
        pelvis_state, pelvis_comp = get_pelvis_kinematics(env)
        joint_state, joint_comp   = get_joint_kinematics(env)
        grf_state, grf_comp       = get_GRF_info(env)
        future_state, future_comp = get_traj_info(env)
    except Exception as e:
        raise ValueError(f"Failed to retrieve sub-states: {e}")
    
    # Compute foot contact flags
    try:
        right_force = grf_comp['right']['GRF_force']
        left_force  = grf_comp['left']['GRF_force']
        contact_thresh = 1e-2
        right_contact = 1.0 if np.linalg.norm(right_force) > contact_thresh else 0.0
        left_contact  = 1.0 if np.linalg.norm(left_force)  > contact_thresh else 0.0
        foot_contacts = np.array([right_contact, left_contact], dtype=np.float32)
    except KeyError as e:
        raise ValueError(f"Missing GRF component for contact computation: {e}")
    except Exception as e:
        raise ValueError(f"Error computing foot contacts: {e}")
    
    # Concatenate full state vector
    try:
        state = np.concatenate([
            pelvis_state,
            joint_state,
            grf_state,
            future_state,
            foot_contacts
        ])
    except Exception as e:
        raise ValueError(f"Error concatenating full state: {e}")

    components = {
        'pelvis':        pelvis_comp,
        'joint':         joint_comp,
        'grf':           grf_comp,
        'traj':          future_comp,
        'foot_contacts': foot_contacts
    }
    return state, components

def get_state_size(env: Any) -> int:
    """
    Compute the dimensionality of the full state vector.

    Parameters
    ----------
    env : Any
        MuJoCo environment instance.

    Returns
    -------
    size : int
        Number of elements in the state returned by get_state().
    """
    state, _ = get_state(env)
    return state.size

def get_relative_joint_positions(env: any) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute joint sensor positions relative to the pelvis translation.

    Parameters
    ----------
    env : Any
        MuJoCo environment instance providing:
        - get_joint_kinematics(): returns (_, components) where components['joint_space_pos'] is a dict of site positions.
        - env.data.qpos: returns the pelvis translation.

    Returns
    -------
    rel_pos_vector : np.ndarray
        1D array of all sensor-site positions minus pelvis translation,
        concatenated in alphabetical order of site names.
    relative_positions : dict
        Mapping from site name (str) to its relative position np.ndarray.

    Raises
    ------
    ValueError
        If no joint sensor positions are available.
    RuntimeError
        If pelvis translation cannot be retrieved.
    """
    # Retrieve sensor positions
    _, joint_comp = get_joint_kinematics(env)
    joint_positions = joint_comp.get('joint_space_pos')
    if not joint_positions:
        raise ValueError("No joint sensor positions available for computing relative positions.")
    
    # Retrieve pelvis translation
    qpos = getattr(env.data, 'qpos', None)
    if qpos is None:
        raise ValueError("Pelvis translation cannot be retrieved.")
    pelvis_trans = qpos[:3].copy()
 
    # Compute relative positions
    relative_positions: Dict[str, np.ndarray] = {
        name: pos - pelvis_trans
        for name, pos in joint_positions.items()
    }

    # Flatten in deterministic order
    ordered_names = sorted(relative_positions.keys())
    rel_pos_vector = np.concatenate([relative_positions[n].ravel() for n in ordered_names])

    return rel_pos_vector, relative_positions  

def get_COM_kinematics(env: any) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute the overall Center-of-Mass (COM) position and velocity of the model.

    Parameters
    ----------
    env : Any
        A MuJoCo environment instance which must provide:
        - env.data.subtree_com: ndarray of shape (n_bodies, 3)
        - env.data.cvel:         ndarray of shape (n_bodies, 6)
        - env.model.body_mass:   ndarray of shape (n_bodies,)
        - Optionally, env.remove_x_pos (bool) to drop the x component from outputs.

    Returns
    -------
    com_state : np.ndarray, shape (6,) or (4,)
        Concatenated [com_pos, com_vel]. If `remove_x_pos=True`, shape is (4,) (y/z + vy/vz).
    components : dict
        - 'com_pos': np.ndarray, shape (3,) or (2,)
        - 'com_vel': np.ndarray, shape (3,) or (2,)

    Raises
    ------
    ValueError
        If required attributes are missing or have unexpected shapes.
    """
    model = env.model
    data = env.data
    
    # Retrieve COM position from subtree_com[0]
    try:
        com_pos = np.array(data.subtree_com[0], copy=True)
    except Exception as e:
        raise ValueError(f"Could not read COM position: {e}")
    if com_pos.shape != (3,):
        raise ValueError(f"Expected COM position of shape (3,), got {com_pos.shape}")
    if getattr(env, "remove_x_pos", False):
        com_pos = com_pos[1:]
    
    # COM velocity: mass-weighted average of each body's COM linear velocity
    try:
        cvel = np.array(data.cvel[1:, 3:6], copy=False)   # shape (n_bodies, 3)
        masses = np.array(model.body_mass[1:], copy=False)   # shape (n_bodies,)
    except Exception as e:
        raise ValueError(f"Could not read cvel or body_mass: {e}")
    if cvel.ndim != 2 or cvel.shape[1] != 3 or masses.ndim != 1 or masses.shape[0] != cvel.shape[0]:
        raise ValueError(f"Unexpected shapes for cvel {cvel.shape} or body_mass {masses.shape}")
    total_mass = masses.sum()
    if total_mass <= 0:
        raise ValueError(f"Total mass must be positive; got {total_mass}")
    com_vel = (masses[:, None] * cvel).sum(axis=0) / total_mass
    
    com_state = np.concatenate((com_pos, com_vel))
    components = {'com_pos': com_pos, 'com_vel': com_vel}

    return com_state, components
    
def get_state_extend(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Retrieve an extended observation vector by augmenting the base state with
    joint-site positions relative to the pelvis and COM kinematics.

    Parameters
    ----------
    env : Any
        A MuJoCo environment instance which must provide:
        - get_state() → (state: np.ndarray, components: dict)
        - get_relative_joint_positions() → (rel_pos: np.ndarray, rel_components: dict)
        - get_COM_kinematics() → (com_state: np.ndarray, com_components: dict)

    Returns
    -------
    extended_state : np.ndarray
        1D array formed by concatenating:
        [base_state,
         rel_site_positions (n_sites*3 or 2),
         com_kinematics]

    components : dict
        A dictionary containing:
          - 'base': dict
              Components returned by get_state().
          - 'relative_joints': dict
              Mapping site_name → relative position vector.
          - 'com': dict
              {
                'com_pos': np.ndarray,
                'com_vel': np.ndarray
              }

    Raises
    ------
    ValueError
        If any of the sub-function calls fail or return inconsistent dimensions.
    """
    try:
        base_state, base_comp = get_state(env)
    except Exception as e:
        raise ValueError(f"Failed to get base state: {e}")

    try:
        rel_pos, rel_comp = get_relative_joint_positions(env)
    except Exception as e:
        raise ValueError(f"Failed to get relative joint positions: {e}")
        
    try:
        com_state, com_comp = get_COM_kinematics(env)
    except Exception as e:
        raise ValueError(f"Failed to get COM kinematics: {e}")
    
    extended_state = np.concatenate((base_state, rel_pos, com_state))
    components: Dict[str, Any] = {
        'base': base_comp,
        'relative_joints': rel_comp,
        'com': com_comp
    }

    return extended_state, components

def get_state_extend_size(env: Any) -> int:
    """
    Compute the dimensionality of the extended state vector.

    Parameters
    ----------
    env : Any
        MuJoCo environment instance.

    Returns
    -------
    size : int
        Number of elements in the extend state returned by get_state_extend().
    """
    state, _ = get_state_extend(env)
    return state.size
    
# joint torques/actuator forces
# foot slip velocity
# muscle/tendon states
# energy/power
# terrain
