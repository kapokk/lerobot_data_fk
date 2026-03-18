"""
Forward Kinematics (FK) Pipeline for processing robot joint data using pytorch-kinematics.

This pipeline:
1. Reads URDF file to get robot model
2. Loads joint position data from parquet file
3. Performs forward kinematics in batch for both arms
4. Writes structured FK results back to parquet file with columns:
   - "qpos": Array of shape [num_traj, num_dim] containing all joint positions
   - "joint_names": List of joint names corresponding to qpos dimensions
   - "ee_pose": Array of shape [num_traj, num_ee, 7] containing xyzqwqxqyqz for each end-effector
   - "ee_name": List of end-effector names
   - "pose_format": String "xyzqwqxqyqz" indicating the pose format
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
            data_path: Path to input parquet file or list of files containing joint position data
            output_path: Path to output parquet file or list of files for FK results
            device: Device to use for computation ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.urdf_path = urdf_path
        self.data_path = data_path
        self.output_path = output_path
        self.is_batch_mode = False  # Track if we're processing multiple files

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
        self.left_arm_pattern = "observation.state.left_arm"
        self.right_arm_pattern = "observation.state.right_arm"
        self.torso_pattern = "observation.state.torso"
        self.left_gripper_col = "observation.state.left_gripper"
        self.right_gripper_col = "observation.state.right_gripper"

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
        left_arm_cols = [self.left_arm_pattern]
        self.left_arm_joints = np.vstack(self._extract_and_stack_columns(left_arm_cols))

        # Right arm joints (0-6)
        right_arm_cols = [self.right_arm_pattern]
        self.right_arm_joints = np.vstack(self._extract_and_stack_columns(right_arm_cols))

        # Torso joints (0-3)
        torso_cols = [self.torso_pattern]
        self.torso_joints = np.vstack(self._extract_and_stack_columns(torso_cols))

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

            # Extract batch for torso
            torso_batch_np = self.torso_joints[i:batch_end]
            torso_batch_torch = torch.from_numpy(torso_batch_np).to(
                dtype=self.dtype,
                device=self.device
            )

            # Concatenate torso with left arm joints
            left_full_batch = torch.cat([torso_batch_torch, left_batch_torch], dim=1)

            # Concatenate torso with right arm joints
            right_full_batch = torch.cat([torso_batch_torch, right_batch_torch], dim=1)

            # Compute FK for left arm
            left_pos_batch, left_quat_batch = self.compute_fk_batch(
                left_full_batch,
                self.left_arm_chain
            )

            # Compute FK for right arm
            right_pos_batch, right_quat_batch = self.compute_fk_batch(
                right_full_batch,
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
        """Prepare output DataFrame with structured columns as requested.

        Creates columns:
        - "qpos": Array of shape [num_traj, num_dim] containing all joint positions
        - "joint_names": List of joint names corresponding to qpos dimensions
        - "ee_pose": Array of shape [num_traj, num_ee, 7] containing xyzqwqxqyqz for each end-effector
        - "ee_name": List of end-effector names
        - "pose_format": String "xyzqwqxqyqz" indicating the pose format
        """
        print("Preparing output data with structured columns...")

        if self.left_fk_results is None or self.right_fk_results is None:
            raise ValueError("FK results not computed. Call run_fk_pipeline() first.")

        n_samples = len(self.data)

        # 1. Create "qpos" column: [num_traj, num_dim] array
        # Collect all joint positions
        qpos_arrays = []
        joint_names_list = []

        # Left arm joints (0-6)
        for i in range(7):
            col_name = f"observation.state.left_arm_{i}"
            if col_name in self.data.columns:
                qpos_arrays.append(self.data[col_name].values.reshape(-1, 1))
                joint_names_list.append(f"left_arm_{i}")

        # Right arm joints (0-6)
        for i in range(7):
            col_name = f"observation.state.right_arm_{i}"
            if col_name in self.data.columns:
                qpos_arrays.append(self.data[col_name].values.reshape(-1, 1))
                joint_names_list.append(f"right_arm_{i}")

        # Torso joints (0-3)
        for i in range(4):
            col_name = f"torso_{i}"
            if col_name in self.data.columns:
                qpos_arrays.append(self.data[col_name].values.reshape(-1, 1))
                joint_names_list.append(f"torso_{i}")

        # Grippers
        if "left_gripper_0" in self.data.columns:
            qpos_arrays.append(self.data["left_gripper_0"].values.reshape(-1, 1))
            joint_names_list.append("left_gripper")

        if "right_gripper_0" in self.data.columns:
            qpos_arrays.append(self.data["right_gripper_0"].values.reshape(-1, 1))
            joint_names_list.append("right_gripper")

        # Stack all joints horizontally: (n_samples, n_joints)
        qpos_array = np.hstack(qpos_arrays) if qpos_arrays else np.zeros((n_samples, 0))

        # 2. Create "ee_pose" column: [num_traj, num_ee, 7] array
        # Extract position and quaternion arrays
        left_pos, left_quat = self.left_fk_results
        right_pos, right_quat = self.right_fk_results

        # Combine position and quaternion: [x, y, z, qw, qx, qy, qz]
        left_pose = np.hstack([left_pos, left_quat])  # (n_samples, 7)
        right_pose = np.hstack([right_pos, right_quat])  # (n_samples, 7)

        # Stack for both end-effectors: (n_samples, 2, 7)
        ee_pose_array = np.stack([left_pose, right_pose], axis=1)

        # 3. Create output DataFrame with structured columns
        output_data = {
            "qpos": list(qpos_array),  # Store as list of arrays for parquet compatibility
            "joint_names": [joint_names_list] * n_samples,  # Same for all rows
            "ee_pose": list(ee_pose_array),  # Store as list of arrays
            "ee_name": [["left_gripper", "right_gripper"]] * n_samples,  # Same for all rows
            "pose_format": ["xyzqwqxqyqz"] * n_samples  # Same for all rows
        }

        # Create DataFrame
        output_df = pd.DataFrame(output_data)

        print(f"Output DataFrame shape: {output_df.shape}")
        print(f"Columns: {list(output_df.columns)}")
        print(f"qpos shape: {qpos_array.shape} (n_samples={n_samples}, n_joints={len(joint_names_list)})")
        print(f"ee_pose shape: {ee_pose_array.shape} (n_samples={n_samples}, n_ee=2, pose_dim=7)")
        print(f"Joint names: {joint_names_list}")
        print(f"End-effector names: {['left_gripper', 'right_gripper']}")
        print(f"Pose format: xyzqwqxqyqz")

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
            # Check if we're processing multiple files
            if isinstance(self.data_path, list) or isinstance(self.output_path, list):
                return self.run_batch()

            # Single file processing
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

    def run_batch(self):
        """Run pipeline for multiple input/output files."""
        print("Starting batch FK pipeline...")

        import os

        # Validate inputs and prepare file pairs
        if isinstance(self.data_path, list) and isinstance(self.output_path, list):
            if len(self.data_path) != len(self.output_path):
                raise ValueError(f"Number of input files ({len(self.data_path)}) "
                               f"does not match number of output files ({len(self.output_path)})")
            file_pairs = list(zip(self.data_path, self.output_path))
        elif isinstance(self.data_path, list):
            # If only data_path is a list
            if os.path.isdir(str(self.output_path)):
                # output_path is a directory - create output filenames based on input filenames
                output_dir = str(self.output_path)
                file_pairs = []
                for input_file in self.data_path:
                    # Get filename from input path
                    filename = os.path.basename(input_file)
                    # Create output path in directory
                    output_file = os.path.join(output_dir, filename)
                    file_pairs.append((input_file, output_file))
            else:
                # output_path is a single file pattern (not supported for multiple inputs)
                raise ValueError("For multiple input files, output_path must be a list or a directory")
        elif isinstance(self.output_path, list):
            # If only output_path is a list, use same data path (not typical but supported)
            file_pairs = [(self.data_path, path) for path in self.output_path]
        else:
            raise ValueError("Batch mode requires at least one of data_path or output_path to be a list")

        print(f"Processing {len(file_pairs)} file pairs")

        # Load URDF once (shared for all files)
        self.load_urdf()

        # Process each file pair
        for i, (input_file, output_file) in enumerate(file_pairs):
            print(f"\n{'='*60}")
            print(f"Processing file {i+1}/{len(file_pairs)}")
            print(f"Input: {input_file}")
            print(f"Output: {output_file}")
            print(f"{'='*60}")

            try:
                # Set paths for this file
                self.data_path = input_file
                self.output_path = output_file

                # Load data, validate, compute FK, and save
                self.load_data()
                self.validate_data()
                self.run_fk_pipeline()
                self.save_results()

                print(f"Successfully processed: {input_file} -> {output_file}")

            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                # Continue with next file
                continue

            finally:
                # Clear data for next iteration
                self.data = None
                self.left_arm_joints = None
                self.right_arm_joints = None
                self.torso_joints = None
                self.left_gripper = None
                self.right_gripper = None
                self.left_fk_results = None
                self.right_fk_results = None

        print(f"\n{'='*60}")
        print(f"Batch processing completed!")
        print(f"Processed {len(file_pairs)} files")
        print(f"{'='*60}")

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
    """FK pipeline CLI that processes parquet files with forward kinematics.

    Usage examples:
        # Process a single parquet file
        python fk_pipeline.py --urdf robot.urdf --input data.parquet --output results.parquet

        # Process all parquet files in a folder
        python fk_pipeline.py --urdf robot.urdf --input /path/to/data_folder --output /path/to/output_folder

        # Process folder with custom batch size
        python fk_pipeline.py --urdf robot.urdf --input /path/to/data_folder --batch-size 1000
    """
    import argparse
    import os
    import glob

    parser = argparse.ArgumentParser(
        description="Forward Kinematics pipeline for processing robot joint data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single parquet file
  python fk_pipeline.py --urdf robot.urdf --input data.parquet --output results.parquet

  # Process all parquet files in folder, output to subfolder 'fk_results'
  python fk_pipeline.py --urdf robot.urdf --input /path/to/data_folder

  # Process folder with custom batch size
  python fk_pipeline.py --urdf robot.urdf --input /path/to/data_folder --batch-size 1000

  # Process folder and specify output folder
  python fk_pipeline.py --urdf robot.urdf --input /path/to/data_folder --output /path/to/output_folder
        """
    )
    parser.add_argument("--urdf", required=True, help="Path to URDF file")
    parser.add_argument("--input", required=True, help="Input parquet file or folder containing parquet files")
    parser.add_argument("--output", help="Output parquet file or folder. If input is a folder and output is not specified, creates 'fk_results' subfolder.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for FK computation (default: 1000)")

    args = parser.parse_args()

    # Check URDF file exists
    if not os.path.exists(args.urdf):
        raise FileNotFoundError(f"URDF file not found: {args.urdf}")

    # Determine if input is a file or folder
    if os.path.isfile(args.input):
        # Single file processing
        input_files = [args.input]
        if args.output:
            if os.path.isdir(args.output):
                # Output is a directory, create output filename based on input filename
                filename = os.path.basename(args.input)
                output_path = os.path.join(args.output, filename)
            else:
                # Output is a file path
                output_path = args.output
        else:
            # Default: add '_fk' suffix to input filename
            base, ext = os.path.splitext(args.input)
            output_path = f"{base}_fk{ext}"

        print(f"Processing single file:")
        print(f"  URDF: {args.urdf}")
        print(f"  Input: {args.input}")
        print(f"  Output: {output_path}")
        print(f"  Batch size: {args.batch_size}")

        pipeline = FKPipeline(args.urdf, input_files, output_path)
        pipeline.set_batch_size(args.batch_size)
        pipeline.run()

    elif os.path.isdir(args.input):
        # Folder processing: find all parquet files
        parquet_files = glob.glob(os.path.join(args.input, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in folder: {args.input}")

        # Determine output directory
        if args.output:
            output_dir = args.output
        else:
            # Create 'fk_results' subfolder in input folder
            output_dir = os.path.join(args.input, "fk_results")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing folder:")
        print(f"  URDF: {args.urdf}")
        print(f"  Input folder: {args.input}")
        print(f"  Found {len(parquet_files)} parquet files")
        print(f"  Output folder: {output_dir}")
        print(f"  Batch size: {args.batch_size}")

        # Use batch mode with output directory
        pipeline = FKPipeline(args.urdf, parquet_files, output_dir)
        pipeline.set_batch_size(args.batch_size)
        pipeline.run()

    else:
        raise FileNotFoundError(f"Input path does not exist: {args.input}")


if __name__ == "__main__":
    main()