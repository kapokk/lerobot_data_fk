# Forward Kinematics (FK) Pipeline Documentation

## Overview

The FK Pipeline is a tool for computing forward kinematics for dual-arm robots using `pytorch-kinematics`. It processes joint position data from parquet files, computes end-effector poses in batch, and saves the results back to parquet files with new columns.

## Features

- **Dual-arm support**: Processes left and right arms simultaneously
- **Batch processing**: Efficient handling of large datasets with configurable batch size
- **GPU acceleration**: Auto-detects CUDA availability, falls back to CPU
- **Position + Quaternion output**: Stores FK results as 7 values per gripper (x,y,z + qw,qx,qy,qz)
- **Data validation**: Checks for required columns and consistent data shapes
- **Error handling**: Comprehensive error checking and informative messages

## Installation

### Prerequisites
```bash
pip install pytorch-kinematics pandas numpy torch
```

### Optional: Install from source
```bash
git clone https://github.com/UM-ARM-Lab/pytorch_kinematics.git
cd pytorch_kinematics
pip install -e .
```

## Data Format

### Input Data Structure
The pipeline expects parquet files with the following column naming convention:

| Column Pattern | Description | Number of Columns |
|----------------|-------------|-------------------|
| `observation.state.left_arm_{0~6}` | Left arm joint positions | 7 |
| `observation.state.right_arm_{0~6}` | Right arm joint positions | 7 |
| `torso_{0~3}` | Torso joint positions | 4 |
| `left_gripper_0` | Left gripper position | 1 |
| `right_gripper_0` | Right gripper position | 1 |

### Output Data Structure
The pipeline adds the following columns to the output:

| Column | Description | Shape |
|--------|-------------|-------|
| `left_gripper_x`, `left_gripper_y`, `left_gripper_z` | Left gripper position (meters) | (n_samples, 3) |
| `left_gripper_qw`, `left_gripper_qx`, `left_gripper_qy`, `left_gripper_qz` | Left gripper orientation (unit quaternion) | (n_samples, 4) |
| `right_gripper_x`, `right_gripper_y`, `right_gripper_z` | Right gripper position (meters) | (n_samples, 3) |
| `right_gripper_qw`, `right_gripper_qx`, `right_gripper_qy`, `right_gripper_qz` | Right gripper orientation (unit quaternion) | (n_samples, 4) |

## URDF Requirements

The pipeline expects a URDF file with:
- End effector links named: `left_gripper_link` and `right_gripper_link`
- Separate kinematic chains for left and right arms (or a single chain with these end effectors)
- Proper joint definitions matching the joint names in the data

## Usage

### Basic Usage
```python
from fk_pipeline import FKPipeline

# Create pipeline
pipeline = FKPipeline(
    urdf_path="path/to/robot.urdf",
    data_path="path/to/input_data.parquet",
    output_path="path/to/output_data.parquet"
)

# Run complete pipeline
pipeline.run()
```

### Advanced Usage
```python
from fk_pipeline import FKPipeline

# Create pipeline with specific device
pipeline = FKPipeline(
    urdf_path="robot.urdf",
    data_path="input.parquet",
    output_path="output.parquet",
    device="cuda"  # Force CUDA, or "cpu" for CPU-only
)

# Adjust batch size (default: 1000)
pipeline.set_batch_size(500)

# Run pipeline step by step
pipeline.load_urdf()
pipeline.load_data()
pipeline.validate_data()
pipeline.run_fk_pipeline()
pipeline.save_results()
```

### Command Line Interface
```bash
# Example script to run pipeline
python -c "
from fk_pipeline import FKPipeline
pipeline = FKPipeline('robot.urdf', 'input.parquet', 'output.parquet')
pipeline.run()
"
```

## API Reference

### `FKPipeline` Class

#### `__init__(urdf_path, data_path, output_path, device=None)`
Initialize the FK pipeline.

**Parameters:**
- `urdf_path` (str): Path to URDF file describing robot model
- `data_path` (str): Path to input parquet file containing joint position data
- `output_path` (str): Path to output parquet file for FK results
- `device` (str, optional): Device to use for computation ('cuda' or 'cpu'). Auto-detects if None.

#### `load_urdf()`
Load URDF file and initialize robot models for both arms using pytorch-kinematics.

**Raises:**
- `RuntimeError`: If URDF loading fails

#### `load_data()`
Load joint position data from parquet file and extract joint arrays.

**Raises:**
- `RuntimeError`: If data loading fails
- `ValueError`: If required columns are missing

#### `validate_data()`
Validate that loaded data has required columns and consistent shapes.

**Raises:**
- `ValueError`: If data validation fails

#### `run_fk_pipeline()`
Run the complete FK pipeline in batches for both arms.

**Raises:**
- `ValueError`: If joint data not loaded

#### `save_results()`
Save FK results to parquet file.

#### `set_batch_size(batch_size)`
Set batch size for FK computation.

**Parameters:**
- `batch_size` (int): Number of samples to process in each batch

**Raises:**
- `ValueError`: If batch size is not positive

#### `run()`
Run the complete pipeline from start to finish (convenience method).

## Performance Tips

### Batch Size Tuning
- **Small datasets** (< 10,000 samples): Use batch size 100-500
- **Medium datasets** (10,000-100,000 samples): Use batch size 500-2000
- **Large datasets** (> 100,000 samples): Use batch size 2000-5000

### GPU Usage
- The pipeline auto-detects CUDA availability
- For optimal GPU performance, ensure batch size is large enough to fully utilize GPU
- Monitor GPU memory usage with `nvidia-smi`

### Memory Management
- Large datasets are processed in batches to avoid memory issues
- Intermediate tensors are moved to CPU after computation
- Consider using `torch.cuda.empty_cache()` if processing multiple files

## Error Handling

### Common Errors and Solutions

1. **URDF Loading Error**
   ```
   RuntimeError: Failed to load URDF: [error details]
   ```
   **Solution**: Check URDF file path and format. Ensure end effector links exist.

2. **Missing Columns Error**
   ```
   ValueError: Missing required columns: ['observation.state.left_arm_0', ...]
   ```
   **Solution**: Verify input data has all required columns with correct naming.

3. **Shape Mismatch Error**
   ```
   ValueError: Array X has Y samples, expected Z
   ```
   **Solution**: Check data consistency. All joint arrays should have same number of samples.

4. **CUDA Out of Memory**
   ```
   torch.cuda.OutOfMemoryError
   ```
   **Solution**: Reduce batch size or use CPU mode.

## Examples

### Example 1: Processing a Single File
```python
from fk_pipeline import FKPipeline

pipeline = FKPipeline(
    urdf_path="models/robot.urdf",
    data_path="data/robot_data.parquet",
    output_path="data/robot_data_with_fk.parquet"
)

pipeline.set_batch_size(1000)
pipeline.run()
```

### Example 2: Processing Multiple Files
```python
import glob
from fk_pipeline import FKPipeline

urdf_path = "models/robot.urdf"
input_files = glob.glob("data/*.parquet")

for input_file in input_files:
    output_file = input_file.replace(".parquet", "_with_fk.parquet")

    pipeline = FKPipeline(urdf_path, input_file, output_file)
    pipeline.set_batch_size(500)
    pipeline.run()

    print(f"Processed: {input_file} -> {output_file}")
```

### Example 3: Custom Configuration
```python
from fk_pipeline import FKPipeline

# Custom device and dtype
pipeline = FKPipeline(
    urdf_path="robot.urdf",
    data_path="data.parquet",
    output_path="output.parquet",
    device="cpu"  # Force CPU computation
)

# Small batch size for testing
pipeline.set_batch_size(100)

# Step-by-step execution with error handling
try:
    pipeline.load_urdf()
    pipeline.load_data()
    pipeline.validate_data()
    pipeline.run_fk_pipeline()
    pipeline.save_results()
    print("Pipeline completed successfully!")
except Exception as e:
    print(f"Pipeline failed: {e}")
```

## Integration with Other Tools

### Using with `parquent_loader`
```python
from fk_pipeline import FKPipeline
from loader.parquent_loader import parquent_loader

# Run FK pipeline
pipeline = FKPipeline("robot.urdf", "input.parquet", "output.parquet")
pipeline.run()

# Load and inspect results
loader = parquent_loader("output.parquet")
df = loader.load()

# Extract FK results as tensor
fk_keys = [
    'left_gripper_x', 'left_gripper_y', 'left_gripper_z',
    'left_gripper_qw', 'left_gripper_qx', 'left_gripper_qy', 'left_gripper_qz',
    'right_gripper_x', 'right_gripper_y', 'right_gripper_z',
    'right_gripper_qw', 'right_gripper_qx', 'right_gripper_qy', 'right_gripper_qz'
]
fk_tensor = loader.get_tensor(fk_keys)
print(f"FK tensor shape: {fk_tensor.shape}")
```

### Using with PyTorch Datasets
```python
import torch
from torch.utils.data import Dataset
from fk_pipeline import FKPipeline

class RobotDataset(Dataset):
    def __init__(self, parquet_path, urdf_path):
        # Run FK pipeline if needed
        output_path = parquet_path.replace(".parquet", "_with_fk.parquet")
        pipeline = FKPipeline(urdf_path, parquet_path, output_path)
        pipeline.run()

        # Load data with FK results
        from loader.parquent_loader import parquent_loader
        loader = parquent_loader(output_path)
        self.data = loader.load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return joint positions and FK results
        sample = self.data.iloc[idx]
        joints = torch.tensor([
            *[sample[f'observation.state.left_arm_{i}'] for i in range(7)],
            *[sample[f'observation.state.right_arm_{i}'] for i in range(7)],
            *[sample[f'torso_{i}'] for i in range(4)],
            sample['left_gripper_0'],
            sample['right_gripper_0']
        ])

        fk_left = torch.tensor([
            sample['left_gripper_x'], sample['left_gripper_y'], sample['left_gripper_z'],
            sample['left_gripper_qw'], sample['left_gripper_qx'],
            sample['left_gripper_qy'], sample['left_gripper_qz']
        ])

        fk_right = torch.tensor([
            sample['right_gripper_x'], sample['right_gripper_y'], sample['right_gripper_z'],
            sample['right_gripper_qw'], sample['right_gripper_qx'],
            sample['right_gripper_qy'], sample['right_gripper_qz']
        ])

        return joints, fk_left, fk_right
```

## Troubleshooting

### Q: The pipeline is slow. How can I speed it up?
**A**:
1. Use GPU if available (`device="cuda"`)
2. Increase batch size (but watch GPU memory)
3. Ensure `pytorch-kinematics` is compiled with `torch.compile` support

### Q: I'm getting shape errors. What should I check?
**A**:
1. Verify all joint columns exist in the input data
2. Check that all arrays have the same number of samples
3. Ensure URDF joint names match data column patterns

### Q: How do I handle different URDF structures?
**A**: Modify the `load_urdf()` method in `fk_pipeline.py` to match your URDF structure. You may need to adjust how chains are built based on your robot's kinematic tree.

### Q: Can I process only one arm?
**A**: Yes, modify the pipeline to skip the other arm's computation. You'll need to adjust the data extraction and FK computation methods.

## Contributing

To contribute to the FK pipeline:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This pipeline is part of the lerobot_data_fk project. See the main project for license details.

## References

- [pytorch-kinematics documentation](pytorch_kinematics.md)
- [PyTorch Kinematics GitHub](https://github.com/UM-ARM-Lab/pytorch_kinematics)
- [URDF Documentation](http://wiki.ros.org/urdf)