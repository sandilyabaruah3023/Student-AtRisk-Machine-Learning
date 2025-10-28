# src/load_data.py
import pandas as pd
from pathlib import Path
from typing import Dict

def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)

def load_oulad(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load OULAD CSV files from data_dir. Returns dict of DataFrames.
    Expects files:
      - assessments.csv
      - courses.csv
      - studentAssessment.csv
      - studentInfo.csv
      - studentRegistration.csv
      - studentVle.csv
      - vle.csv
    """
    data_dir = Path(data_dir)
    files = {
        "assessments": data_dir / "assessments.csv",
        "courses": data_dir / "courses.csv",
        "studentAssessment": data_dir / "studentAssessment.csv",
        "studentInfo": data_dir / "studentInfo.csv",
        "studentRegistration": data_dir / "studentRegistration.csv",
        "studentVle": data_dir / "studentVle.csv",
        "vle": data_dir / "vle.csv",
    }
    dfs = {}
    for name, path in files.items():
        if path.exists():
            print(f"Loading {path.name} ...")
            dfs[name] = read_csv_safe(path)
        else:
            raise FileNotFoundError(f"Required file not found: {path}")
    return dfs

if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "../OULAD"
    dfs = load_oulad(d)
    for k,v in dfs.items():
        print(f"{k}: {v.shape}")
