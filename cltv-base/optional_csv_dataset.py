# Solutions_CLTV_A2A\cltv-base\src\cltv_base\optional_csv_dataset.py

import pandas as pd
from kedro.io import AbstractDataset
from pathlib import Path
from typing import Any, Dict

class OptionalCSVDataset(AbstractDataset):
    """
    A DataSet that checks for the existence of the CSV file.
    If the file exists, it loads it as a pandas DataFrame.
    If the file does not exist, it logs a warning and returns an empty DataFrame
    with no columns, allowing the pipeline to proceed gracefully.
    """
    def __init__(self, filepath: str, save_args: Dict[str, Any] = None, load_args: Dict[str, Any] = None):
        """
        Initializes the OptionalCSVDataSet.
        
        Args:
            filepath: The location of the CSV file.
            save_args: Optional arguments for pandas.DataFrame.to_csv.
            load_args: Optional arguments for pandas.read_csv.
        """
        self._filepath = filepath
        self._save_args = save_args or {}
        self._load_args = load_args or {}

    def _load(self) -> pd.DataFrame:
        path = Path(self._filepath)
        if not path.is_file():
            # Crucial for optional data: return an empty DataFrame
            print(f"[WARN] Optional file not found at: {self._filepath}. Returning empty DataFrame.")
            return pd.DataFrame()
            
        print(f"[INFO] Loading file from: {self._filepath}")
        return pd.read_csv(path, **self._load_args)

    def _save(self, data: pd.DataFrame) -> None:
        path = Path(self._filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, **self._save_args)

    def _exists(self) -> bool:
        return Path(self._filepath).is_file()

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, load_args=self._load_args, save_args=self._save_args)