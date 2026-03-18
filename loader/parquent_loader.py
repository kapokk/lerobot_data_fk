import pandas as pd
import numpy as np
import torch

class parquent_loader:
    """Simple loader for parquet files."""

    def __init__(self, file_path):
        """Initialize with file path.

        Args:
            file_path: Path to parquet file.
        """
        self.file_path = file_path
        self.data = None

    def load(self):
        """Load parquet file into pandas DataFrame and store in self.data.

        Returns:
            pandas.DataFrame: Loaded data (also stored in self.data).
        """
        self.data = pd.read_parquet(self.file_path)
        return self.data

    def get_tensor(self, keys):
        """Extract columns by keys and merge them into a torch tensor.

        Args:
            keys: List of column names to extract from the loaded data.

        Returns:
            torch.Tensor: Combined tensor of shape (n_samples, total_features)
            where total_features is the sum of feature dimensions across all keys.

        Raises:
            ValueError: If data is not loaded or keys are not found in data.
            ValueError: If extracted arrays have different lengths.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")

        # Check all keys exist
        missing_keys = [key for key in keys if key not in self.data.columns]
        if missing_keys:
            raise ValueError(f"Keys not found in data: {missing_keys}")

        # Extract arrays for each key
        arrays = []
        for key in keys:
            array = self.data[key].values
            arrays.append(array)

        # Check all arrays have same length
        lengths = [len(arr) for arr in arrays]
        if len(set(lengths)) > 1:
            raise ValueError(f"Arrays have different lengths: {lengths}")

        # Stack arrays along feature dimension
        # Convert to torch tensor and ensure float32 dtype
        stacked = np.column_stack(arrays) if len(arrays) > 1 else arrays[0]
        tensor = torch.from_numpy(stacked).float()

        return tensor

    def save(self, file_path: str = None, **kwargs):
        """Save the loaded data to a parquet file.

        Args:
            file_path: Path to save the parquet file. If None, uses the original file path.
            **kwargs: Additional arguments passed to pandas.DataFrame.to_parquet()

        Raises:
            ValueError: If data is not loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")

        save_path = file_path if file_path is not None else self.file_path
        self.data.to_parquet(save_path, **kwargs)
        print(f"Data saved to: {save_path}")