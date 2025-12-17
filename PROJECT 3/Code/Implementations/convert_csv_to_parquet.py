# convert_csv_to_parquet.py
# Convert large CSV files to Parquet format (much smaller and faster)

import pandas as pd
from pathlib import Path

# ----------------------------
# Project paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "Data_Processed"

# Files to convert (with nat_ prefix)
FILES_TO_CONVERT = [
    "nat_nn_tabular_dataset.csv",
    "nat_meta.csv",
]

print("Converting CSV files to Parquet format...")
print(f"Output directory: {OUT_DIR}\n")

for csv_file in FILES_TO_CONVERT:
    csv_path = OUT_DIR / csv_file
    
    if not csv_path.exists():
        print(f"[SKIP] Skipping {csv_file} (not found)")
        continue
    
    print(f"[READ] Reading {csv_file}...")
    df = pd.read_csv(csv_path)
    print(f"       Shape: {df.shape}")
    
    # Get file size
    csv_size = csv_path.stat().st_size / (1024 * 1024)  # MB
    print(f"       CSV size: {csv_size:.2f} MB")
    
    # Convert to parquet
    parquet_file = csv_file.replace(".csv", ".parquet")
    parquet_path = OUT_DIR / parquet_file
    
    print(f"[SAVE] Saving to {parquet_file}...")
    df.to_parquet(parquet_path, compression='snappy', index=False)
    
    # Get parquet file size
    parquet_size = parquet_path.stat().st_size / (1024 * 1024)  # MB
    reduction = (1 - parquet_size / csv_size) * 100
    
    print(f"       Parquet size: {parquet_size:.2f} MB")
    print(f"       Size reduction: {reduction:.1f}%")
    print()

print("[DONE] Conversion complete!")
print(f"\nParquet files saved to: {OUT_DIR}")
print("You can now add CSV files to .gitignore and commit Parquet files instead.")

