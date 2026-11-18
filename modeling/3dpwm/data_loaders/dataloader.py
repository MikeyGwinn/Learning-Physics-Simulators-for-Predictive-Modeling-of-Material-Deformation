"""Deformable point-cloud sequences stored in gs-particle-dynamics format."""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from data_loaders.data_utils import calculate_displacements


class Dataset(TorchDataset):
    def __init__(self, phase, cfg):
        self.cfg = cfg
        self.phase = phase
        self.stride = cfg["dataset"]["training_fpt"]
        if self.stride != 1:
            raise ValueError("The sequence loader currently supports training_fpt=1.")
        self.root = (
            Path(cfg["dataset"]["data_dir"]) / cfg["dataset"]["scenario"] / phase
        )

        # Stores the paths to each params_coarse.npz file for each scene
        self.scenes = []

        # 
        self.frames = []
        raw_window = cfg["training"]["rollout_num"] + 1

        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir():
                continue

            gs_dir = scene_dir / "gs"
            if not gs_dir.is_dir():
                continue

            frame_paths = self._get_frame_paths(gs_dir)
            scene_index = len(self.scenes)
            self.scenes.append((scene_dir.name, frame_paths))

            if phase == "train":
                for start in range(len(frame_paths) - raw_window + 1):
                    self.frames.append((scene_index, start))
            elif len(frame_paths) >= 9:
                self.frames.append((scene_index, None))

        if not self.frames:
            raise ValueError(f"No usable {phase} sequences found in {self.root}")

    @staticmethod
    def _get_frame_paths(scene_folder):
        paths_by_frame = {}
        for frame_dir in scene_folder.iterdir():
            if not frame_dir.is_dir():
                continue
            try:
                frame_number = int(frame_dir.name)
            except ValueError:
                continue
            if frame_number in paths_by_frame:
                raise ValueError(
                    f"Duplicate numeric frame {frame_number} in {scene_folder}."
                )

            param_path = frame_dir / "params_coarse.npz"
            if not param_path.is_file():
                raise FileNotFoundError(
                    f"Missing params_coarse.npz for frame {frame_dir}."
                )
            paths_by_frame[frame_number] = param_path

        if not paths_by_frame:
            raise ValueError(f"No point-cloud frames found in {scene_folder}.")

        frame_numbers = sorted(paths_by_frame)
        if len(frame_numbers) > 2:
            steps = {
                right - left
                for left, right in zip(frame_numbers, frame_numbers[1:])
            }
            if len(steps) != 1:
                raise ValueError(
                    f"Frame spacing is not constant in {scene_folder}: {sorted(steps)}"
                )

        return [paths_by_frame[number] for number in frame_numbers]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        scene_index, start = self.frames[index]
        name, all_frame_paths = self.scenes[scene_index]
        if start is None:
            frame_paths = all_frame_paths[:: self.stride]
            start = 0
        else:
            raw_window = self.cfg["training"]["rollout_num"] + 1
            end = start + raw_window * self.stride
            frame_paths = all_frame_paths[start:end:self.stride]

        sequence = []
        zero_action = np.zeros(self.cfg["model"]["action_channels"], dtype=np.float32) # placeholder for actions
        for frame_path in frame_paths:
            positions = load_positions(frame_path)
            sequence.append(
                {
                    "positions": positions,
                    "displacements": None,
                    "actions": zero_action.copy(),
                }
            )

        return {
            "seq_name": name,
            "start_frame": start,
            "pc_data": calculate_displacements(sequence),
            "grid_size": self.cfg["pointcloud"]["grid_size"],
        }


def load_positions(path):
    with np.load(path, allow_pickle=False) as data:
        if "means3D" not in data.files:
            raise KeyError(f"{path} is missing means3D")
        positions = np.asarray(data["means3D"], dtype=np.float32)

    if positions.ndim == 3 and positions.shape[0] == 1:
        positions = positions[0]
    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError(
            f"{path}: expected means3D shape (N, 2), (N, 3), "
            f"(1, N, 2), or (1, N, 3); got {positions.shape}"
        )
    if len(positions) == 0:
        raise ValueError(f"{path}: point cloud is empty")
    if not np.isfinite(positions).all():
        raise ValueError(f"{path}: point cloud contains non-finite values")

    if positions.shape[1] == 2:
        # Embed the 2D cloud in the xy plane
        positions = np.concatenate([positions, np.zeros((len(positions), 1), dtype=np.float32)], axis=1,)
    else:
        # Verify z-coordinate is 0
        if not (positions[:, 2] == 0).all():
            raise ValueError(
                f"{path}: point cloud is not planar. "
            )

    return positions


def collate_fn(examples):
    """Pack variable point counts; each time step becomes a point dictionary."""
    lengths = {len(item["pc_data"]) for item in examples}
    if len(lengths) != 1:
        raise ValueError("Full test trajectories require batch_size=1.")
    counts = [len(item["pc_data"][0]["positions"]) for item in examples]
    offset = np.cumsum(counts).astype(np.int64)
    batch = np.repeat(np.arange(len(examples)), counts).astype(np.int64)
    packed = []
    for step in range(1, len(examples[0]["pc_data"])):
        current = [item["pc_data"][step] for item in examples]
        features = []
        for item, frame in zip(examples, current):
            previous = item["pc_data"][step - 1]
            features.append(
                np.concatenate(
                    [frame["displacements"], previous["displacements"]], axis=1
                )
            )
        packed.append(
            {
                "coord": np.concatenate(
                    [frame["positions"] for frame in current]
                ).astype(np.float32),
                "feat": np.concatenate(features).astype(np.float32),
                "offset": offset.copy(),
                "batch": batch.copy(),
                "action": np.stack([frame["actions"] for frame in current]).astype(
                    np.float32
                ),
                "grid_size": examples[0]["grid_size"],
            }
        )
    return packed
