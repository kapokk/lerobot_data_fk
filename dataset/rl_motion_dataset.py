"""
Simple PyTorch Dataset for loading PKL files containing qpos and ee_pose.

The PKL files contain forward kinematics results with the following structure:
- "qpos": List of arrays of shape [num_traj, num_dim] containing all joint positions
- "ee_pose": List of arrays of shape [num_traj, num_ee, 7] containing xyzqwqxqyqz for each end-effector

This dataset provides access to individual samples for training or inference.
"""

import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Union
import os


class RLMotionDataset(Dataset):
    """Simple PyTorch Dataset for loading PKL files with qpos and ee_pose.

    Args:
        pkl_path: Path to PKL file or directory containing PKL files
        load_all: If True, load all data into memory. If False, load on demand.
        device: Device to store tensors on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        pkl_path: Union[str, List[str]],
        load_all: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()
        self.load_all = load_all
        self.device = device

        # Handle single file or list of files
        if isinstance(pkl_path, str):
            if os.path.isdir(pkl_path):
                # Load all PKL files in directory
                pkl_files = [os.path.join(pkl_path, f) for f in os.listdir(pkl_path)
                            if f.endswith('.pkl')]
                if not pkl_files:
                    raise FileNotFoundError(f"No .pkl files found in directory: {pkl_path}")
                self.pkl_paths = sorted(pkl_files)
            else:
                # Single file
                self.pkl_paths = [pkl_path]
        else:
            # List of files
            self.pkl_paths = pkl_path

        # Validate all files exist
        for path in self.pkl_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"PKL file not found: {path}")

        # Calculate total samples across all files
        self.total_samples = 0
        self.file_offsets = [0]  # Cumulative sample counts

        for pkl_path in self.pkl_paths:
            with open(pkl_path, 'rb') as f:
                data_dict = pickle.load(f)
            num_samples = len(data_dict['qpos'])
            self.total_samples += num_samples
            self.file_offsets.append(self.total_samples)

        # Load data based on load_all flag
        if self.load_all:
            self._load_all_data()
        else:
            self.data = None  # Will load on demand in __getitem__

    def _load_all_data(self):
        """Load all data from all PKL files into memory."""
        all_qpos = []
        all_ee_pose = []

        for pkl_path in self.pkl_paths:
            with open(pkl_path, 'rb') as f:
                data_dict = pickle.load(f)

            # Convert lists to numpy arrays
            qpos_array = np.array(data_dict['qpos'])
            ee_pose_array = np.array(data_dict['ee_pose'])

            all_qpos.append(qpos_array)
            all_ee_pose.append(ee_pose_array)

        # Concatenate all files
        self.qpos = np.concatenate(all_qpos, axis=0)
        self.ee_pose = np.concatenate(all_ee_pose, axis=0)

        # Convert to tensors
        self.qpos_tensor = torch.from_numpy(self.qpos).float().to(self.device)
        self.ee_pose_tensor = torch.from_numpy(self.ee_pose).float().to(self.device)

    def _load_file_data(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from a specific file."""
        with open(self.pkl_paths[file_idx], 'rb') as f:
            data_dict = pickle.load(f)

        qpos_array = np.array(data_dict['qpos'])
        ee_pose_array = np.array(data_dict['ee_pose'])

        return qpos_array, ee_pose_array

    def _get_file_and_sample_idx(self, idx: int) -> Tuple[int, int]:
        """Convert global index to (file_index, sample_index_within_file)."""
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples-1}]")

        # Find which file contains this sample
        for file_idx in range(len(self.pkl_paths)):
            if idx < self.file_offsets[file_idx + 1]:
                sample_idx = idx - self.file_offsets[file_idx]
                return file_idx, sample_idx

        raise IndexError(f"Could not find file for index {idx}")

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample by index.

        Returns:
            Dictionary containing:
                - 'qpos': Joint positions tensor of shape [num_joints]
                - 'ee_pose': End-effector poses tensor of shape [num_ee, 7]
        """
        if self.load_all:
            # Get from pre-loaded tensors
            qpos = self.qpos_tensor[idx]
            ee_pose = self.ee_pose_tensor[idx]
        else:
            # Load on demand
            file_idx, sample_idx = self._get_file_and_sample_idx(idx)
            qpos_array, ee_pose_array = self._load_file_data(file_idx)

            qpos = torch.from_numpy(qpos_array[sample_idx]).float().to(self.device)
            ee_pose = torch.from_numpy(ee_pose_array[sample_idx]).float().to(self.device)

        sample = {
            'qpos': qpos,
            'ee_pose': ee_pose,
        }

        return sample


# Example usage and test
if __name__ == "__main__":
    # Example 1: Load single PKL file
    print("Example 1: Loading single PKL file")
    dataset = RLMotionDataset(
        pkl_path="path/to/your/fk_results.pkl",
        load_all=True,
        device='cpu'
    )

    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"qpos shape: {sample['qpos'].shape}")
    print(f"ee_pose shape: {sample['ee_pose'].shape}")

    # Example 2: Load directory of PKL files
    print("\nExample 2: Loading directory of PKL files")
    # dataset_dir = RLMotionDataset(
    #     pkl_path="path/to/your/fk_results_directory/",
    #     load_all=True,
    #     device='cpu'
    # )
    # print(f"Loaded {len(dataset_dir)} samples from directory")