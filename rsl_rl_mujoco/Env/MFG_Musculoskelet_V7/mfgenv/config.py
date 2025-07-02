import json
from pathlib import Path
import logging
from typing import Any, Dict, Optional, Union

# Configure logging for the module
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add a default StreamHandler if no handler exists.
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Default path for the configuration file
DEFAULT_CONFIG_PATH: Path = Path(__file__).parent / "config.json"

def deep_merge(source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries. Values in `overrides` will overwrite those in `source`.
    
    Args:
        source (Dict[str, Any]): The base dictionary.
        overrides (Dict[str, Any]): The dictionary with override values.
    
    Returns:
        Dict[str, Any]: The merged dictionary.
    """
    result = source.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(config_path: Optional[Union[str, Path]] = None, overrides: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    Load environment configuration from a JSON file with the option to override specific parameters.
    
    The function loads a JSON configuration from `config_path` if provided (or from the default path),
    then merges it with a set of default parameters, and finally applies additional overrides.
    
    Args:
        config_path (Optional[Union[str, Path]]): Path to the JSON configuration file. If None, uses DEFAULT_CONFIG_PATH.
        overrides (Dict[str, Any]): Dictionary of parameters to override in the loaded configuration.
    
    Returns:
        Dict[str, Any]: The final configuration dictionary.
    
    Raises:
        FileNotFoundError: If the specified config file is not found.
        json.JSONDecodeError: If the JSON file cannot be parsed.
    """
    # Use the provided config path or fallback to the default one
    file_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    loaded_config: Dict[str, Any] = {}
    if file_path.exists():
        try:
            with file_path.open("r", encoding="utf-8") as f:
                loaded_config = json.load(f)
            logger.info("Loaded config from %s", file_path)
        except json.JSONDecodeError as e:
            logger.error("Error decoding JSON from config file %s: %s", file_path, e)
            raise e
    else:
        # logger.warning("Config file %s not found. Using default values.", file_path)
        pass

    # original
    default_config: Dict[str, Any] = {
        "model_path": "Env/MFG_Musculoskelet_V7/MFG_Raj2015_V7.xml",
        "data_path": "Env/MFG_mocap/SUBJECT01_steps.pkl",
        
        "reward_weights": {
            'imitation': 1.00,
            "smooth": 0.00,
            'goal': 0.00,
        },
        "imitation_weights": {
            "site_pos": 0.79,
            "site_vel": 0.01,
            "joint_angle": 0.195,
            "joint_angvel": 0.005
        },
        "smooth_weights": {
            "grf": 0.0,
            "torque": 0.0,
            "action": 1.0
        },
        "reward_coefficients": {
            "site_pos": 20.0,
            "site_vel": 0.05,
            "joint_angle": 20.0,
            "joint_angvel": 0.05,
            "com_speed": 0.5,
            "grf": 0.1,
            "torque": 1e-6,
            "action": 2.0
        },
        "ref_traj_repeat_times": 20,
        "remove_x_pos": False,
        "short_history_max_len": 2,
        "long_history_max_len": 2,

        "random_seed": None,
        "verbose": False
    }

    # Merge default configuration, loaded configuration and overrides using deep merge
    merged_config = deep_merge(default_config, loaded_config)
    merged_config = deep_merge(merged_config, overrides)
    
    return merged_config

def save_config(config: Dict[str, Any], save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Save the provided configuration dictionary to a JSON file.
    
    Args:
        config (Dict[str, Any]): The configuration dictionary to save.
        save_path (Optional[Union[str, Path]]): Path to save the configuration file. If None, saves to DEFAULT_CONFIG_PATH.
    
    Raises:
        IOError: If saving the file fails.
    """
    file_path = Path(save_path) if save_path else DEFAULT_CONFIG_PATH
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        logger.info("Config saved to %s", file_path)
    except Exception as e:
        logger.error("Failed to save config to %s: %s", file_path, e)
        raise e