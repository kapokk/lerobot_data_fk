"""
Forward Kinematics (FK) Pipeline for processing robot joint data using pytorch-kinematics.

This pipeline:
1. Reads URDF file to get robot model
2. Loads joint position data from parquet file
3. Performs forward kinematics in batch for both arms
4. Writes FK results (position + quaternion) back to parquet file
"""

import os
import numpy as np
import torch
import pytorch_kinematics as pk
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd


class FKPipeline:
    """Forward Kinematics pipeline for batch processing of robot joint data using pytorch-kinematics."""

    def __init__(self, urdf_path: str, data_path: str, output_path: str, device: str = None):
        """Initialize the FK pipeline.

        Args:
            urdf_path: Path to URDF file describing robot model
            data_path: Path to input parquet file containing joint position data
            output_path: Path to output parquet file for FK results
            device: Device to use for computation ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.urdf_path = urdf_path
        self.data_path = data_path
        self.output_path = output_path

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Robot models for left and right arms
        self.left_arm_chain = None
        self.right_arm_chain = None

        # Data containers
        self.data = None  # Loaded DataFrame
        self.left_arm_joints = None  # Left arm joint positions tensor
        self.right_arm_joints = None  # Right arm joint positions tensor
        self.torso_joints = None  # Torso joint positions tensor
        self.left_gripper = None  # Left gripper position tensor
        self.right_gripper = None  # Right gripper position tensor

        # FK results
        self.left_fk_results = None  # Left gripper FK results
        self.right_fk_results = None  # Right gripper FK results

        # Configuration
        self.batch_size = 1000  # Default batch size for FK computation
        self.dtype = torch.float32  # Data type for computation

        # Joint column patterns (based on your description)
        self.left_arm_pattern = "observation.state.left_arm_{}"
        self.right_arm_pattern = "observation.state.right_arm_{}"
        self.torso_pattern = "torso_{}"
        self.left_gripper_col = "left_gripper_0"
        self.right_gripper_col = "right_gripper_0"

        # End effector links
        self.left_ee_link = "left_gripper_link"
        self.right_ee_link = "right_gripper_link"

    def load_urdf(self):
        """Load URDF file and initialize robot models for both arms using pytorch-kinematics."""
        print(f"Loading URDF from: {self.urdf_path}")
        print(f"Using device: {self.device}")

        try:
            # Read URDF file
            with open(self.urdf_path, "rb") as f:
                urdf_content = f.read()

            # Build chains for left and right arms
            # Note: Assuming the URDF has separate chains for left and right arms
            # If it's a single chain, we might need to create SerialChains with different end effectors
            self.left_arm_chain = pk.build_serial_chain_from_urdf(
                urdf_content,
                self.left_ee_link
            )
            self.right_arm_chain = pk.build_serial_chain_from_urdf(
                urdf_content,
                self.right_ee_link
            )

            # Move chains to device
            self.left_arm_chain = self.left_arm_chain.to(
                dtype=self.dtype,
                device=self.device
            )
            self.right_arm_chain = self.right_arm_chain.to(
                dtype=self.dtype,
                device=self.device
            )

            # Print chain information
            print("Left arm chain joint names:", self.left_arm_chain.get_joint_parameter_names())
            print("Right arm chain joint names:", self.right_arm_chain.get_joint_parameter_names())

        except Exception as e:
            raise RuntimeError(f"Failed to load URDF: {e}")

    def load_data(self):
        """Load joint position data from parquet file and extract joint arrays."""
        print(f"Loading data from: {self.data_path}")

        try:
            # Import here to avoid circular imports
            from loader.parquent_loader import parquent_loader

            # Load data using parquent_loader
            loader = parquent_loader(self.data_path)
            self.data = loader.load()

            print(f"Loaded {len(self.data)} samples")
            print(f"Columns: {list(self.data.columns)}")

            # Extract joint position arrays
            self._extract_joint_arrays()

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def _extract_joint_arrays(self):
        """Extract joint position arrays from the loaded DataFrame."""
        # Left arm joints (0-6)
        left_arm_cols = [self.left_arm_pattern.format(i) for i in range(7)]
        self.left_arm_joints = self._extract_and_stack_columns(left_arm_cols)

        # Right arm joints (0-6)
        right_arm_cols = [self.right_arm_pattern.format(i) for i in range(7)]
        self.right_arm_joints = self._extract_and_stack_columns(right_arm_cols)

        # Torso joints (0-3)
        torso_cols = [self.torso_pattern.format(i) for i in range(4)]
        self.torso_joints = self._extract_and_stack_columns(torso_cols)

        # Gripper positions
        self.left_gripper = self.data[self.left_gripper_col].values
        self.right_gripper = self.data[self.right_gripper_col].values

        print(f"Left arm joints shape: {self.left_arm_joints.shape}")
        print(f"Right arm joints shape: {self.right_arm_joints.shape}")
        print(f"Torso joints shape: {self.torso_joints.shape}")
        print(f"Left gripper shape: {self.left_gripper.shape}")
        print(f"Right gripper shape: {self.right_gripper.shape}")

    def _extract_and_stack_columns(self, columns: List[str]) -> np.ndarray:
        """Extract and stack multiple columns into a single array."""
        arrays = []
        for col in columns:
            if col in self.data.columns:
                arrays.append(self.data[col].values)
            else:
                raise ValueError(f"Column not found: {col}")

        # Stack columns horizontally: (n_samples, n_joints)
        return np.column_stack(arrays) if len(arrays) > 1 else arrays[0]

    def validate_data(self):
        """Validate that loaded data has required columns and consistent shapes."""
        print("Validating data...")

        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Check all required columns exist
        required_cols = []
        for i in range(7):
            required_cols.append(self.left_arm_pattern.format(i))
            required_cols.append(self.right_arm_pattern.format(i))
        for i in range(4):
            required_cols.append(self.torso_pattern.format(i))
        required_cols.extend([self.left_gripper_col, self.right_gripper_col])

        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check all arrays have same number of samples
        n_samples = len(self.data)
        arrays = [
            self.left_arm_joints, self.right_arm_joints,
            self.torso_joints, self.left_gripper, self.right_gripper
        ]

        for i, arr in enumerate(arrays):
            if arr is not None and len(arr) != n_samples:
                raise ValueError(f"Array {i} has {len(arr)} samples, expected {n_samples}")

        print(f"Data validation passed: {n_samples} samples")

    def compute_fk_batch(self, joint_batch: torch.Tensor, chain: pk.SerialChain) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics for a batch of joint positions.

        Args:
            joint_batch: Tensor of joint positions with shape (batch_size, n_joints)
            chain: pytorch-kinematics SerialChain for the arm

        Returns:
            Tuple of (positions, quaternions) as numpy arrays
            positions: (batch_size, 3) array of x,y,z coordinates
            quaternions: (batch_size, 4) array of qw,qx,qy,qz quaternions
        """
        # Compute forward kinematics
        transforms = chain.forward_kinematics(joint_batch)

        # Get transform matrix (batch_size, 4, 4)
        transform_matrix = transforms.get_matrix()

        # Extract position (batch_size, 3)
        positions = transform_matrix[:, :3, 3]

        # Extract rotation matrix and convert to quaternion (batch_size, 4)
        rotation_matrix = transform_matrix[:, :3, :3]
        quaternions = pk.matrix_to_quaternion(rotation_matrix)

        # Convert to numpy for storage
        return positions.cpu().numpy(), quaternions.cpu().numpy()

    def run_fk_pipeline(self):
        """Run the complete FK pipeline in batches for both arms."""
        print("Running FK pipeline...")

        if self.left_arm_joints is None or self.right_arm_joints is None:
            raise ValueError("Joint data not loaded. Call load_data() first.")

        n_samples = len(self.data)
        print(f"Processing {n_samples} samples in batches of {self.batch_size}")

        # Initialize result arrays
        left_positions = []
        left_quaternions = []
        right_positions = []
        right_quaternions = []

        # Process in batches
        for i in range(0, n_samples, self.batch_size):
            batch_end = min(i + self.batch_size, n_samples)
            batch_size_actual = batch_end - i

            print(f"Processing batch {i//self.batch_size + 1}/{(n_samples + self.batch_size - 1)//self.batch_size} "
                  f"(samples {i} to {batch_end-1})")

            # Extract batch for left arm
            left_batch_np = self.left_arm_joints[i:batch_end]
            left_batch_torch = torch.from_numpy(left_batch_np).to(
                dtype=self.dtype,
                device=self.device
            )

            # Extract batch for right arm
            right_batch_np = self.right_arm_joints[i:batch_end]
            right_batch_torch = torch.from_numpy(right_batch_np).to(
                dtype=self.dtype,
                device=self.device
            )

            # Compute FK for left arm
            left_pos_batch, left_quat_batch = self.compute_fk_batch(
                left_batch_torch,
                self.left_arm_chain
            )

            # Compute FK for right arm
            right_pos_batch, right_quat_batch = self.compute_fk_batch(
                right_batch_torch,
                self.right_arm_chain
            )

            # Store results
            left_positions.append(left_pos_batch)
            left_quaternions.append(left_quat_batch)
            right_positions.append(right_pos_batch)
            right_quaternions.append(right_quat_batch)

        # Concatenate all batches
        self.left_fk_results = (
            np.concatenate(left_positions, axis=0),
            np.concatenate(left_quaternions, axis=0)
        )
        self.right_fk_results = (
            np.concatenate(right_positions, axis=0),
            np.concatenate(right_quaternions, axis=0)
        )

        print("FK pipeline completed successfully!")
        print(f"Left FK results: positions {self.left_fk_results[0].shape}, "
              f"quaternions {self.left_fk_results[1].shape}")
        print(f"Right FK results: positions {self.right_fk_results[0].shape}, "
              f"quaternions {self.right_fk_results[1].shape}")

    def prepare_output_data(self) -> pd.DataFrame:
        """Prepare output DataFrame with FK results added as new columns."""
        print("Preparing output data...")

        if self.left_fk_results is None or self.right_fk_results is None:
            raise ValueError("FK results not computed. Call run_fk_pipeline() first.")

        # Create a copy of the original data
        output_df = self.data.copy()

        # Extract position and quaternion arrays
        left_pos, left_quat = self.left_fk_results
        right_pos, right_quat = self.right_fk_results

        # Add left gripper FK results
        output_df['left_gripper_x'] = left_pos[:, 0]
        output_df['left_gripper_y'] = left_pos[:, 1]
        output_df['left_gripper_z'] = left_pos[:, 2]
        output_df['left_gripper_qw'] = left_quat[:, 0]
        output_df['left_gripper_qx'] = left_quat[:, 1]
        output_df['left_gripper_qy'] = left_quat[:, 2]
        output_df['left_gripper_qz'] = left_quat[:, 3]

        # Add right gripper FK results
        output_df['right_gripper_x'] = right_pos[:, 0]
        output_df['right_gripper_y'] = right_pos[:, 1]
        output_df['right_gripper_z'] = right_pos[:, 2]
        output_df['right_gripper_qw'] = right_quat[:, 0]
        output_df['right_gripper_qx'] = right_quat[:, 1]
        output_df['right_gripper_qy'] = right_quat[:, 2]
        output_df['right_gripper_qz'] = right_quat[:, 3]

        print(f"Output DataFrame shape: {output_df.shape}")
        print(f"New columns added: {[col for col in output_df.columns if 'gripper_' in col]}")

        return output_df

    def save_results(self):
        """Save FK results to parquet file."""
        print(f"Saving results to: {self.output_path}")

        # Prepare output data
        output_df = self.prepare_output_data()

        # Save using parquent_loader
        from loader.parquent_loader import parquent_loader
        loader = parquent_loader(self.output_path)
        loader.data = output_df
        loader.save()

        print(f"Results saved successfully to {self.output_path}")

    def run(self):
        """Run the complete pipeline from start to finish."""
        print("Starting FK pipeline...")

        try:
            self.load_urdf()
            self.load_data()
            self.validate_data()
            self.run_fk_pipeline()
            self.save_results()

            print("FK pipeline completed successfully!")
            print(f"Results saved to: {self.output_path}")

        except Exception as e:
            print(f"Error in FK pipeline: {e}")
            raise

    def set_batch_size(self, batch_size: int):
        """Set batch size for FK computation.

        Args:
            batch_size: Number of samples to process in each batch
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        self.batch_size = batch_size
        print(f"Batch size set to: {batch_size}")


def main():
    """Example usage of the FK pipeline."""
    # TODO: Set your actual file paths
    urdf_path = "path/to/robot.urdf"
    data_path = "path/to/input_data.parquet"
    output_path = "path/to/output_data.parquet"

    # Create and run pipeline
    pipeline = FKPipeline(urdf_path, data_path, output_path)
    pipeline.set_batch_size(500)  # Optional: adjust batch size
    pipeline.run()


if __name__ == "__main__":
    main()