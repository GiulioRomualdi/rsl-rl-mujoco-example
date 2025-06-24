"""
preprocess_trajs.py

Offline preprocessing for gait reference trajectories.

This script loads raw gait segments from one or more .pkl files,
performs the following steps in batch:
  1. Remove extra knee degrees of freedom (rotation2/3).
  2. Optionally augment with left–right mirroring.
  3. Trim each segment to ~2/3 cycle, center pelvis lateral translation,
     tile/repeat with translation continuity and overlap-smoothing.
  4. Smooth the full trajectory with a Gaussian filter.
  5. Compute joint velocities via finite differences (forward/backward/center).
  6. Expand each 37-DOF trajectory to 43-DOF by inserting three patella-related
     DOFs per leg using a polynomial model (analytic derivative for velocity).
  7. Rebuild a final 43-DOF joint-name → index mapping.
  8. Save each processed segment individually as a compressed .npz with `qpos`
     and `qvel`, and write an index.json summarizing all outputs.

Usage:
    python preprocess_trajs.py \
        --input /path/to/raw_trajs \
        --output /path/to/processed_trajs \
        [--no-mirror] [--method forward] [--sigma 3.0] [--repeat 2] \
        [--overlap 10] [--uniform] [--freq 100.0]
"""
import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d

# configure root logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Base 37‐DOF mapping
BASE_JOINT_MAP: Dict[str,int] = {
    'pelvis_tz':0,'pelvis_ty':1,'pelvis_tx':2,
    'pelvis_tilt':3,'pelvis_list':4,'pelvis_rotation':5,
    'hip_flexion_r':6,'hip_adduction_r':7,'hip_rotation_r':8,
    'knee_angle_r':9,'knee_angle_r_rotation2':10,'knee_angle_r_rotation3':11,
    'ankle_angle_r':12,'subtalar_angle_r':13,'mtp_angle_r':14,
    'hip_flexion_l':15,'hip_adduction_l':16,'hip_rotation_l':17,
    'knee_angle_l':18,'knee_angle_l_rotation2':19,'knee_angle_l_rotation3':20,
    'ankle_angle_l':21,'subtalar_angle_l':22,'mtp_angle_l':23,
    'lumbar_extension':24,'lumbar_bending':25,'lumbar_rotation':26,
    'arm_flex_r':27,'arm_add_r':28,'arm_rot_r':29,
    'elbow_flex_r':30,'pro_sup_r':31,'wrist_flex_r':32,'wrist_dev_r':33,
    'arm_flex_l':34,'arm_add_l':35,'arm_rot_l':36,
    'elbow_flex_l':37,'pro_sup_l':38,'wrist_flex_l':39,'wrist_dev_l':40
}

# polynomial coefficients for beta‐DOFs
BETA_COEF = np.array([
    [-0.0108281, -0.0487847,  0.00927644,  0.0131673,   -0.00349673],
    [ 0.0524192, -0.0150188, -0.0340522,   0.0133393,   -0.000879151],
    [ 0.010506 ,  0.0247615, -1.31647   ,  0.716337 ,   -0.138302  ],
])
TARGET_DOF = 43

class RawTrajectoryProcessor:
    """
    Handles offline preprocessing of raw gait segments into ready-to-use
    reference trajectories with 43 DOFs (qpos, qvel) saved per-segment.
    """
    
    def __init__(self,
                 remove_extra_knee: bool = True,
                 enable_mirroring: bool = True,
                 smoothing_sigma: float = 3.0,
                 repeat_times: int = 1,
                 splice_overlap: int = 10,
                 uniform_length: bool = False,
                 sample_frequency: float = 100.0,
                 velocity_method: str = "center"):
        """
        Args:
            remove_extra_knee: strip knee_rotation2/3 DOFs.
            enable_mirroring: append left–right mirrored segments.
            smoothing_sigma: sigma for Gaussian smoothing.
            repeat_times: number of times to tile each trimmed cycle.
            splice_overlap: frame overlap for splice smoothing.
            uniform_length: crop all trajectories to minimal cycle×repeat length.
            sample_frequency: sampling freq (Hz) for finite differences.
            velocity_method: 'forward', 'backward', or 'center'.
        """
        self.remove_extra_knee = remove_extra_knee
        self.enable_mirroring = enable_mirroring
        self.smoothing_sigma = float(smoothing_sigma)
        self.repeat_times = int(repeat_times)
        self.splice_overlap = int(splice_overlap)
        self.uniform_length = uniform_length
        self.sample_frequency = float(sample_frequency)
        if velocity_method not in {"forward", "backward", "center"}:
            raise ValueError(f"Invalid velocity_method '{velocity_method}'")
        self.velocity_method = velocity_method

        # Dynamic mappings:
        self.jnt_name: Dict[str, int] = {}
        self.base_jnt_name: Dict[str, int] = {}

    def process(self, input_path: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Main entry: load raw .pkl files, preprocess all segments, and save per-segment .npz.
        """
        inp = Path(input_path)
        out_dir = Path(output_dir)
        if not inp.exists():
            raise FileNotFoundError(f"Input path does not exist: {inp}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Gather raw segments
        files = self._collect_pkl_files(inp)
        all_data = self._load_all_segments(files)
        
        # Remove extra knee DOFs
        self.jnt_name = dict(BASE_JOINT_MAP)
        if self.remove_extra_knee:
            all_data = self._remove_extra_knee(all_data)

        # Mirror augmentation
        if self.enable_mirroring:
            all_data = self._augment_mirroring(all_data)

        # Freeze 37-DOF map
        self.base_jnt_name = dict(self.jnt_name)

         # Preprocess each segment
        processed = []
        for stem, idx, seg in all_data:
            qpos37 = self._process_step_trim_repeat(seg)
            qvel37 = self._compute_velocity(qpos37)
            qpos43, qvel43 = self._expand_beta(qpos37, qvel37)
            processed.append((stem, idx, qpos43, qvel43))

        if not processed:
            raise RuntimeError("No segments processed; check your inputs/filters.")
        # self.ref_traj = processed

        # Rebuild full 43-DOF mapping
        self._rebuild_expanded_mapping()

        # Save outputs
        self._save_processed(processed, out_dir)
        logger.info("All trajectories processed successfully.")

    # ---------- helper methods ----------
    def _collect_pkl_files(self, inp: Path) -> List[Path]:
        if inp.is_dir():
            files = sorted(inp.glob("*.pkl"))
            if not files:
                raise FileNotFoundError(f"No .pkl files found in directory: {inp}")
            return files
        elif inp.is_file() and inp.suffix == ".pkl":
            return [inp]
        else:
            raise ValueError(f"Invalid input path: {inp}")
            
    def _load_all_segments(self, files: List[Path]
                           ) -> List[Tuple[str, int, np.ndarray]]:
        """
        Load all segments from each .pkl, return list of (filename, index, data).
        """
        all_data = []
        for f in files:
            try:
                segments = pickle.loads(f.read_bytes())
            except Exception as e:
                logger.error(f"Failed to load {f}: {e}")
                continue
            if not isinstance(segments, list):
                logger.warning(f"{f} did not contain a list; skipping.")
                continue
            for idx, seg in enumerate(segments):
                if not (isinstance(seg, np.ndarray) and seg.ndim == 2):
                    logger.warning(f"Segment {idx} in {f} invalid shape; skipping.")
                    continue
                all_data.append((f.stem, idx, seg.copy()))
        return all_data

    def _remove_extra_knee(self, data: List[Tuple[str, int, np.ndarray]]
                           ) -> List[Tuple[str, int, np.ndarray]]:
        """
        Delete knee_rotation2/3 rows and update self.jnt_name accordingly.
        """
        removal_keys = [
            "knee_angle_r_rotation2", "knee_angle_r_rotation3",
            "knee_angle_l_rotation2", "knee_angle_l_rotation3"
        ]
        remove_idxs = sorted(self.jnt_name[k] for k in removal_keys
                             if k in self.jnt_name)
        if not remove_idxs:
            logger.info("No extra knee DOFs found; skipping removal.")
            return data

        # Remove rows
        pruned = []
        for stem, idx, seg in data:
            pruned_seg = np.delete(seg, remove_idxs, axis=0)
            pruned.append((stem, idx, pruned_seg))

        # Update mapping
        for name, old_idx in list(self.jnt_name.items()):
            shift = sum(1 for r in remove_idxs if r < old_idx)
            new_idx = old_idx - shift
            if name in removal_keys:
                self.jnt_name.pop(name, None)
            else:
                self.jnt_name[name] = new_idx

        logger.info(f"Removed extra knee DOFs at rows {remove_idxs}")
        return pruned

    def _augment_mirroring(self, data: List[Tuple[str, int, np.ndarray]]
                           ) -> List[Tuple[str, int, np.ndarray]]:
        """
        Append left–right mirrored copy of each segment.
        """
        augmented = []
        for stem, idx, seg in data:
            augmented.append((stem, idx, seg))
            augmented.append((f"{stem}_mirror", idx, self._mirror_step(seg)))
        logger.info(f"Mirrored {len(data)}→{len(augmented)} segments")
        return augmented

    def _mirror_step(self, seg: np.ndarray) -> np.ndarray:
        """
        Swap left/right joint rows and flip lateral sign for pelvis/lumbar.
        """
        dofs, _ = seg.shape
        idx_map = np.arange(dofs, dtype=int)
        # left/right swap pairs
        pairs = {
            "hip_flexion_r": "hip_flexion_l", "hip_adduction_r": "hip_adduction_l",
            "hip_rotation_r": "hip_rotation_l", "knee_angle_r": "knee_angle_l",
            "ankle_angle_r": "ankle_angle_l", "subtalar_angle_r": "subtalar_angle_l",
            "mtp_angle_r": "mtp_angle_l", "arm_flex_r": "arm_flex_l",
            "arm_add_r": "arm_add_l", "arm_rot_r": "arm_rot_l",
            "elbow_flex_r": "elbow_flex_l", "pro_sup_r": "pro_sup_l",
            "wrist_flex_r": "wrist_flex_l", "wrist_dev_r": "wrist_dev_l"
        }
        for r, l in pairs.items():
            if r in self.jnt_name and l in self.jnt_name:
                ir, il = self.jnt_name[r], self.jnt_name[l]
                idx_map[ir], idx_map[il] = il, ir

        mirrored = seg[idx_map, :].copy()

        # flip sign on lateral DOFs
        flips = ["pelvis_tz", "pelvis_list", "pelvis_rotation",
                 "lumbar_bending", "lumbar_rotation"]
        for key in flips:
            if key in self.jnt_name:
                mirrored[self.jnt_name[key], :] *= -1

        return mirrored

    def _process_step_trim_repeat(self, seg: np.ndarray) -> np.ndarray:
        """
        Trim to 2/3 cycle, center pelvis lateral, tile & splice-smooth, then
        apply global Gaussian smoothing.
        """
        total = seg.shape[1]
        cut = int(np.ceil(2/3 * total))
        trimmed = seg[:, :cut].copy()

        # center pelvis lateral
        if "pelvis_tz" in self.jnt_name:
            tz = self.jnt_name["pelvis_tz"]
            trimmed[tz, :] -= trimmed[tz, :].mean()

        # tile + splice smoothing
        tiled = self._repeat_step_with_translation(trimmed)

        # final smoothing
        sigma = self.smoothing_sigma
        return gaussian_filter1d(tiled, sigma=sigma, axis=1)

    def _repeat_step_with_translation(self, arr: np.ndarray) -> np.ndarray:
        """
        Tile trimmed cycle, apply translation offsets on pelvis DOFs, then
        smooth only the overlap regions.
        """
        N, L = self.repeat_times, arr.shape[1]
        keys = ["pelvis_tz", "pelvis_ty", "pelvis_tx"]
        t_idx = [self.jnt_name[k] for k in keys if k in self.jnt_name]
        start = arr[t_idx, 0][:, None]
        end = arr[t_idx, -1][:, None]
        offsets = (end - start) @ np.arange(N)[None, :]

        tiled = np.tile(arr, (1, N))
        for i, idx in enumerate(t_idx):
            tiled[idx, :] += np.repeat(offsets[i], L)

        total = L * N
        mask = np.zeros(total, dtype=bool)
        for c in range(1, N):
            b = c * L
            a = max(0, b - self.splice_overlap)
            d = min(total, b + self.splice_overlap)
            mask[a:d] = True

        sm = gaussian_filter1d(tiled, sigma=self.smoothing_sigma, axis=1)
        tiled[:, mask] = sm[:, mask]

        if self.uniform_length:
            desired = L * N
            tiled = tiled[:, :desired]

        return tiled

    def _compute_velocity(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute joint velocities via finite differences.
        """
        method = self.velocity_method
        dt = 1.0 / self.sample_frequency
        if method == "center":
            return np.gradient(pos, dt, axis=1)
        dif = np.diff(pos, axis=1) / dt
        if method == "forward":
            return np.pad(dif, ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
        # backward
        return np.pad(dif, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)

    def _compute_beta(self, x: np.ndarray, xdot: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute patella‐related DOFs and their velocities via analytic derivative.
        """
        P = BETA_COEF.shape[1]
        # angles: BETA_COEF @ [x^0; x^1; ... x^(P-1)]
        powers = np.vstack([x**i for i in range(P)])  # (P, T)
        angles = BETA_COEF @ powers                   # (3, T)

        # derivative w.r.t time: beta'(q) * qdot
        dp = np.vstack([(i * x**(i - 1)) if i > 0 else np.zeros_like(x) for i in range(P)])
        vels = (BETA_COEF @ dp) * xdot                 # (3, T)
        return angles, vels

    def _expand_beta(self, qpos: np.ndarray, qvel: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert right then left β‐DOFs based on frozen base_jnt_name.
        """
        base = self.base_jnt_name
        kr, kl = base["knee_angle_r"], base["knee_angle_l"]
        mr, ml = base["mtp_angle_r"], base["mtp_angle_l"]

        ang_r, vel_r = self._compute_beta(qpos[kr], qvel[kr])
        ang_l, vel_l = self._compute_beta(qpos[kl], qvel[kl])

        # insert right‐leg β
        qpos1 = np.insert(qpos, mr + 1, ang_r, axis=0)
        qvel1 = np.insert(qvel, mr + 1, vel_r, axis=0)

        # left insert index shifts if ml > mr
        left_idx = ml + 1 + (ang_r.shape[0] if ml > mr else 0)
        full_qpos = np.insert(qpos1, left_idx, ang_l, axis=0)
        full_qvel = np.insert(qvel1, left_idx, vel_l, axis=0)

        return full_qpos, full_qvel

    def _rebuild_expanded_mapping(self) -> None:
        """
        After expansion, rebuild self.jnt_name to include the six β‐DOFs.
        """
        # start from sorted base names
        items = sorted(self.base_jnt_name.items(), key=lambda kv: kv[1])
        names = [name for name, _ in items]

        # insert right β
        pos_r = names.index("mtp_angle_r")
        for nm in ["knee_angle_r_beta_translation2",
                   "knee_angle_r_beta_translation1",
                   "knee_angle_r_beta_rotation1"]:
            pos_r += 1
            names.insert(pos_r, nm)

        # insert left β
        pos_l = names.index("mtp_angle_l")
        for nm in ["knee_angle_l_beta_translation2",
                   "knee_angle_l_beta_translation1",
                   "knee_angle_l_beta_rotation1"]:
            pos_l += 1
            names.insert(pos_l, nm)

        self.jnt_name = {name: idx for idx, name in enumerate(names)}
        
    def _save_processed(self,
                        processed: List[Tuple[str, int, np.ndarray, np.ndarray]],
                        out_dir: Path) -> None:
        """
        Save each trajectory segment as compressed .npz and write index.json.
        """
        index = []
        for stem, idx, qpos, qvel in processed:
            fname = f"{stem}_seg{idx:03d}.npz"
            out_f = out_dir / fname
            np.savez_compressed(out_f, qpos=qpos, qvel=qvel)
            index.append({"file": fname, "frames": int(qpos.shape[1])})
            logger.info(f"Saved {fname}")
        with open(out_dir / "index.json", "w") as jf:
            json.dump(index, jf, indent=2)
        logger.info("Wrote index.json")
        
        
input_path  = r"C:/Users/YAKEz/OneDrive/Desktop/MFG_Musculoskelet_V9/MFG_mocap"
output_dir  = r"C:/Users/YAKEz/OneDrive/Desktop/MFG_Musculoskelet_V9/processed_trajs"
remove_knee = True
do_mirror   = True
sigma       = 3.0
repeat_cnt  = 10
splice_ov   = 10
uniform_len = False

processor = RawTrajectoryProcessor(
    remove_extra_knee=remove_knee,
    enable_mirroring=do_mirror,
    smoothing_sigma=sigma,
    repeat_times=repeat_cnt,
    splice_overlap=splice_ov,
    uniform_length=uniform_len,
    velocity_method="center"
)
processor.process(input_path, output_dir)
logger.info("Processing completed.")


# def main():
#     p = argparse.ArgumentParser(
#         description="Offline preprocess raw gait .pkl into 43-DOF .npz trajectories"
#     )
#     p.add_argument("-i", "--input", required=True,
#                    help="Input .pkl file or directory containing .pkl files")
#     p.add_argument("-o", "--output", required=True,
#                    help="Output directory for processed .npz files")
#     p.add_argument("--no-mirror", action="store_true",
#                    help="Disable left–right mirroring")
#     p.add_argument("--method", choices=["forward", "backward", "center"],
#                    default="center", help="Finite difference scheme for velocity")
#     p.add_argument("--sigma", type=float, default=3.0,
#                    help="Gaussian smoothing sigma")
#     p.add_argument("--repeat", type=int, default=10,
#                    help="Number of repeats of trimmed cycle")
#     p.add_argument("--overlap", type=int, default=10,
#                    help="Splice overlap frames")
#     p.add_argument("--uniform", action="store_true",
#                    help="Crop all trajectories to uniform length")
#     p.add_argument("--freq", type=float, default=100.0,
#                    help="Sampling frequency for velocity (Hz)")

#     args = p.parse_args()

#     processor = RawTrajectoryProcessor(
#         remove_extra_knee=True,
#         enable_mirroring=not args.no_mirror,
#         smoothing_sigma=args.sigma,
#         repeat_times=args.repeat,
#         splice_overlap=args.overlap,
#         uniform_length=args.uniform,
#         sample_frequency=args.freq,
#         velocity_method=args.method,
#     )
#     try:
#         processor.process(args.input, args.output)
#     except Exception as e:
#         logger.exception(f"Processing failed: {e}")
#         exit(1)


# if __name__ == "__main__":
#     main()