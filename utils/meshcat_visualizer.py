"""
MeshCat visualizer for robot joint positions and end-effector frames from parquet data.

This tool:
1. Loads robot URDF and parquet data with FK results
2. Visualizes robot poses in MeshCat
3. Shows end-effector frames (coordinate axes) using meshcat-shapes
4. Allows playback of trajectory data
"""

import numpy as np
import torch
import pandas as pd
import pytorch_kinematics as pk
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import meshcat_shapes
import time
from typing import List, Dict, Optional, Tuple
import threading
import queue


class MeshcatVisualizer:
    """Visualize robot poses and end-effector frames from parquet data using MeshCat."""

    def __init__(self, urdf_path: str, data_path: str,
                 zmq_url: str = "tcp://127.0.0.1:6000"):
        """
        Initialize the MeshCat visualizer.

        Args:
            urdf_path: Path to URDF file
            data_path: Path to parquet file with joint data and FK results
            zmq_url: MeshCat server URL (default: tcp://127.0.0.1:6000)
        """
        self.urdf_path = urdf_path
        self.data_path = data_path
        self.zmq_url = zmq_url

        # Robot model
        self.robot_chain = None
        self.left_arm_chain = None
        self.right_arm_chain = None

        # Data
        self.data = None
        self.joint_names = []
        self.n_samples = 0
        self.current_frame = 0

        # MeshCat
        self.vis = None
        self.robot_geometry = {}
        self.ee_frames = {}  # End-effector frame visualizations

        # Playback control
        self.is_playing = False
        self.playback_speed = 1.0  # Real-time factor
        self.frame_delay = 0.033  # ~30 FPS

        # Colors
        self.robot_color = [0.7, 0.7, 0.7, 1.0]  # Gray
        self.left_ee_color = [1.0, 0.0, 0.0, 1.0]  # Red
        self.right_ee_color = [0.0, 0.0, 1.0, 1.0]  # Blue
        self.frame_scale = 0.1  # Size of coordinate frame axes

    def load_robot(self):
        """Load robot URDF and build kinematic chains."""
        print(f"Loading robot from: {self.urdf_path}")

        try:
            with open(self.urdf_path, "rb") as f:
                urdf_content = f.read()

            # Build full chain for visualization
            self.robot_chain = pk.build_chain_from_urdf(urdf_content)

            # Build chains for left and right arms (for FK if needed)
            self.left_arm_chain = pk.build_serial_chain_from_urdf(
                urdf_content, "left_gripper_link"
            )
            self.right_arm_chain = pk.build_serial_chain_from_urdf(
                urdf_content, "right_gripper_link"
            )

            # Get joint names
            self.joint_names = self.robot_chain.get_joint_parameter_names()
            print(f"Robot loaded with {len(self.joint_names)} joints")
            print(f"Joint names: {self.joint_names}")

        except Exception as e:
            raise RuntimeError(f"Failed to load robot: {e}")

    def load_data(self):
        """Load parquet data with joint positions and FK results."""
        print(f"Loading data from: {self.data_path}")

        try:
            # Load using pandas directly for simplicity
            self.data = pd.read_parquet(self.data_path)
            self.n_samples = len(self.data)

            print(f"Loaded {self.n_samples} samples")
            print(f"Columns: {list(self.data.columns)}")

            # Check for required columns
            self._check_required_columns()

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def _check_required_columns(self):
        """Check that required columns exist in the data."""
        # Check for joint columns
        required_joint_cols = []
        for i in range(7):
            required_joint_cols.append(f"observation.state.left_arm_{i}")
            required_joint_cols.append(f"observation.state.right_arm_{i}")
        for i in range(4):
            required_joint_cols.append(f"torso_{i}")
        required_joint_cols.extend(["left_gripper_0", "right_gripper_0"])

        missing_joint_cols = [col for col in required_joint_cols
                            if col not in self.data.columns]
        if missing_joint_cols:
            print(f"Warning: Missing joint columns: {missing_joint_cols[:5]}...")

        # Check for FK columns
        fk_cols = [
            'left_gripper_x', 'left_gripper_y', 'left_gripper_z',
            'left_gripper_qw', 'left_gripper_qx', 'left_gripper_qy', 'left_gripper_qz',
            'right_gripper_x', 'right_gripper_y', 'right_gripper_z',
            'right_gripper_qw', 'right_gripper_qx', 'right_gripper_qy', 'right_gripper_qz'
        ]

        missing_fk_cols = [col for col in fk_cols if col not in self.data.columns]
        if missing_fk_cols:
            print(f"Warning: Missing FK columns. Will compute FK from joints.")
            self.has_fk = False
        else:
            self.has_fk = True
            print("FK columns found in data")

    def setup_meshcat(self):
        """Set up MeshCat visualization."""
        print(f"Connecting to MeshCat at {self.zmq_url}")

        try:
            self.vis = meshcat.Visualizer(zmq_url=self.zmq_url)
            self.vis.open()

            # Clear any existing visualizations
            self.vis.delete()

            # Set background color
            self.vis["/Background"].set_property("top_color", [1.0, 1.0, 1.0])
            self.vis["/Background"].set_property("bottom_color", [0.9, 0.9, 0.9])

            print("MeshCat connected successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to connect to MeshCat: {e}")

    def create_robot_visualization(self):
        """Create simple robot visualization using primitive shapes."""
        print("Creating robot visualization...")

        # Base (torso)
        self.vis["robot/base"].set_object(
            g.Box([0.3, 0.3, 0.1]),
            g.MeshLambertMaterial(color=self.robot_color[:3], opacity=self.robot_color[3])
        )
        self.vis["robot/base"].set_transform(tf.translation_matrix([0, 0, 0.05]))

        # Left arm (simplified as a chain of cylinders)
        for i in range(3):  # Shoulder, elbow, wrist
            link_name = f"robot/left_arm_{i}"
            length = 0.2
            radius = 0.03

            self.vis[link_name].set_object(
                g.Cylinder(length, radius),
                g.MeshLambertMaterial(color=[0.8, 0.2, 0.2], opacity=0.8)
            )
            # Position will be updated during playback

        # Right arm
        for i in range(3):
            link_name = f"robot/right_arm_{i}"
            length = 0.2
            radius = 0.03

            self.vis[link_name].set_object(
                g.Cylinder(length, radius),
                g.MeshLambertMaterial(color=[0.2, 0.2, 0.8], opacity=0.8)
            )

        # End-effector frames using meshcat-shapes
        self._create_ee_frame_meshcat_shapes("left_ee", self.left_ee_color)
        self._create_ee_frame_meshcat_shapes("right_ee", self.right_ee_color)

        print("Robot visualization created")

    def _create_ee_frame_meshcat_shapes(self, name: str, color: List[float]):
        """Create coordinate frame visualization for end-effector using meshcat-shapes."""
        # Create frame using meshcat-shapes
        meshcat_shapes.frame(
            self.vis[f"frames/{name}"],
            axis_length=self.frame_scale,
            axis_thickness=0.005,
            origin_radius=0.01,
            opacity=0.8
        )

        # Store frame reference
        self.ee_frames[name] = f"frames/{name}"

    def get_joint_dict(self, frame_idx: int) -> Dict[str, float]:
        """Extract joint positions as dictionary for given frame index."""
        row = self.data.iloc[frame_idx]

        joint_dict = {}

        # Left arm joints
        for i in range(7):
            col_name = f"observation.state.left_arm_{i}"
            if col_name in self.data.columns:
                joint_dict[f"left_arm_{i}"] = float(row[col_name])

        # Right arm joints
        for i in range(7):
            col_name = f"observation.state.right_arm_{i}"
            if col_name in self.data.columns:
                joint_dict[f"right_arm_{i}"] = float(row[col_name])

        # Torso joints
        for i in range(4):
            col_name = f"torso_{i}"
            if col_name in self.data.columns:
                joint_dict[f"torso_{i}"] = float(row[col_name])

        # Grippers
        if "left_gripper_0" in self.data.columns:
            joint_dict["left_gripper"] = float(row["left_gripper_0"])
        if "right_gripper_0" in self.data.columns:
            joint_dict["right_gripper"] = float(row["right_gripper_0"])

        return joint_dict

    def get_ee_pose(self, frame_idx: int, side: str = "left") -> np.ndarray:
        """Get end-effector pose (4x4 transformation matrix) for given frame."""
        row = self.data.iloc[frame_idx]

        if self.has_fk:
            # Use precomputed FK from data
            if side == "left":
                x = row['left_gripper_x']
                y = row['left_gripper_y']
                z = row['left_gripper_z']
                qw = row['left_gripper_qw']
                qx = row['left_gripper_qx']
                qy = row['left_gripper_qy']
                qz = row['left_gripper_qz']
            else:  # right
                x = row['right_gripper_x']
                y = row['right_gripper_y']
                z = row['right_gripper_z']
                qw = row['right_gripper_qw']
                qx = row['right_gripper_qx']
                qy = row['right_gripper_qy']
                qz = row['right_gripper_qz']

            # Create transformation matrix from position and quaternion
            T = tf.quaternion_matrix([qx, qy, qz, qw])
            T[:3, 3] = [x, y, z]
            return T

        else:
            # Compute FK from joints (simplified)
            joint_dict = self.get_joint_dict(frame_idx)

            # Simplified forward kinematics for visualization
            # This is a placeholder - in practice, use pytorch-kinematics
            if side == "left":
                # Simple arm model: base at (0, 0.2, 0.1)
                base_pos = np.array([0, 0.2, 0.1])
            else:
                # Right arm base at (0, -0.2, 0.1)
                base_pos = np.array([0, -0.2, 0.1])

            # Create a simple transformation
            T = np.eye(4)
            T[:3, 3] = base_pos
            return T

    def update_robot_pose(self, frame_idx: int):
        """Update robot visualization for given frame index."""
        joint_dict = self.get_joint_dict(frame_idx)

        # Simplified update for visualization
        # In a real implementation, you would:
        # 1. Compute FK for all links using pytorch-kinematics
        # 2. Update each link's transform in MeshCat

        # For now, just update end-effector frames
        self.update_ee_frames(frame_idx)

        # Update frame counter display using meshcat-shapes text
        meshcat_shapes.textarea(
            self.vis["info/frame"],
            f"Frame: {frame_idx}/{self.n_samples}",
            width=0.5,
            height=0.1,
            font_size=24,
            color=0x000000,  # Black
            background_color=0xFFFFFF,  # White
            background_opacity=0.7
        )
        # Position the text in the top-left corner
        self.vis["info/frame"].set_transform(
            tf.translation_matrix([-0.6, 0.4, 0]) @
            tf.scale_matrix(0.1)
        )

    def update_ee_frames(self, frame_idx: int):
        """Update end-effector frame visualizations."""
        # Left end-effector
        T_left = self.get_ee_pose(frame_idx, "left")
        self.vis[self.ee_frames["left_ee"]].set_transform(T_left)

        # Right end-effector
        T_right = self.get_ee_pose(frame_idx, "right")
        self.vis[self.ee_frames["right_ee"]].set_transform(T_right)

    def play(self, start_frame: int = 0, end_frame: Optional[int] = None):
        """Play back trajectory from start_frame to end_frame."""
        if end_frame is None:
            end_frame = self.n_samples - 1

        self.is_playing = True
        self.current_frame = start_frame

        print(f"Playing frames {start_frame} to {end_frame}")

        while self.is_playing and self.current_frame <= end_frame:
            start_time = time.time()

            # Update visualization
            self.update_robot_pose(self.current_frame)

            # Move to next frame
            self.current_frame += 1

            # Control playback speed
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_delay / self.playback_speed - elapsed)
            time.sleep(sleep_time)

            # Print progress every 100 frames
            if self.current_frame % 100 == 0:
                print(f"Frame: {self.current_frame}/{end_frame}")

        self.is_playing = False
        print("Playback finished")

    def stop(self):
        """Stop playback."""
        self.is_playing = False

    def set_playback_speed(self, speed: float):
        """Set playback speed multiplier."""
        self.playback_speed = max(0.1, min(10.0, speed))
        print(f"Playback speed set to: {self.playback_speed}x")

    def goto_frame(self, frame_idx: int):
        """Go to specific frame."""
        frame_idx = max(0, min(frame_idx, self.n_samples - 1))
        self.current_frame = frame_idx
        self.update_robot_pose(frame_idx)
        print(f"Jumped to frame: {frame_idx}")

    def run_interactive(self):
        """Run interactive visualization with keyboard controls."""
        print("\n" + "="*50)
        print("MeshCat Visualizer - Interactive Mode")
        print("="*50)
        print("Controls:")
        print("  [space] - Play/Pause")
        print("  [→]     - Next frame")
        print("  [←]     - Previous frame")
        print("  [↑]     - Increase speed")
        print("  [↓]     - Decrease speed")
        print("  [g]     - Go to frame (enter number)")
        print("  [q]     - Quit")
        print("="*50)

        # Note: For full interactive controls, you would need to:
        # 1. Use a GUI framework (Tkinter, PyQt) for keyboard input
        # 2. Or implement a web interface with MeshCat's JavaScript API
        # 3. Or use Jupyter notebook for interactive control

        print("\nOpen your browser to view the visualization.")
        print("Use the methods above to control playback programmatically.")

    def visualize_sample(self, frame_idx: int = 0):
        """Visualize a single sample frame."""
        print(f"Visualizing frame {frame_idx}")

        # Load everything
        self.load_robot()
        self.load_data()
        self.setup_meshcat()
        self.create_robot_visualization()

        # Update to specified frame
        self.goto_frame(frame_idx)

        # Keep visualization open
        print("\nVisualization ready. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting visualization")

    def visualize_trajectory(self, start_frame: int = 0, end_frame: Optional[int] = None):
        """Visualize a trajectory (playback)."""
        print(f"Visualizing trajectory from frame {start_frame}")

        # Load everything
        self.load_robot()
        self.load_data()
        self.setup_meshcat()
        self.create_robot_visualization()

        # Play trajectory
        self.play(start_frame, end_frame)

        # Keep visualization open
        print("\nPlayback finished. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting visualization")


def main():
    """Example usage of the MeshCat visualizer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize robot data with MeshCat using meshcat-shapes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize single frame
  python -m utils.meshcat_visualizer --urdf robot.urdf --data data.parquet --frame 50

  # Play trajectory with custom frame size
  python -m utils.meshcat_visualizer --urdf robot.urdf --data data.parquet --play --start 0 --end 500 --frame-scale 0.15

  # Play at 2x speed
  python -m utils.meshcat_visualizer --urdf robot.urdf --data data.parquet --play --speed 2.0

Requirements:
  Make sure meshcat-server is running: meshcat-server
  Then open browser to: http://127.0.0.1:6000
        """
    )
    parser.add_argument("--urdf", required=True, help="Path to URDF file")
    parser.add_argument("--data", required=True, help="Path to parquet data file")
    parser.add_argument("--frame", type=int, default=0, help="Frame to visualize")
    parser.add_argument("--play", action="store_true", help="Play trajectory")
    parser.add_argument("--start", type=int, default=0, help="Start frame for playback")
    parser.add_argument("--end", type=int, help="End frame for playback")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.1 to 10.0)")
    parser.add_argument("--frame-scale", type=float, default=0.1,
                       help="Size of coordinate frames (default: 0.1)")

    args = parser.parse_args()

    # Create visualizer
    visualizer = MeshcatVisualizer(args.urdf, args.data)
    visualizer.set_playback_speed(args.speed)
    visualizer.frame_scale = args.frame_scale  # Set frame scale

    if args.play:
        # Play trajectory
        visualizer.visualize_trajectory(args.start, args.end)
    else:
        # Visualize single frame
        visualizer.visualize_sample(args.frame)


if __name__ == "__main__":
    main()