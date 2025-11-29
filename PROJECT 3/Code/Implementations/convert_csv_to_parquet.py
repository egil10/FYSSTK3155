"""Convert existing CSV files to Parquet format for better compression.

This utility script converts CSV files to Parquet format, which typically
reduces file size by 50-90% while preserving all data and types.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_csv_to_parquet(csv_path: Path, output_path: Path | None = None, delete_csv: bool = False) -> Path:
    """Convert CSV file to Parquet format.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path for output Parquet file (default: same name with .parquet extension)
        delete_csv: Whether to delete the original CSV file after conversion
    
    Returns:
        Path to the created Parquet file
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if output_path is None:
        output_path = csv_path.with_suffix(".parquet")
    
    print(f"Converting {csv_path.name} to Parquet format...")
    print(f"  Reading CSV...", end=" ", flush=True)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"✓ ({len(df):,} rows, {len(df.columns)} columns, {csv_size_mb:.1f} MB)")
    
    # Write Parquet
    print(f"  Writing Parquet...", end=" ", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression='snappy')
    parquet_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ ({parquet_size_mb:.1f} MB)")
    
    # Calculate compression ratio
    compression_ratio = (1 - parquet_size_mb / csv_size_mb) * 100
    size_reduction = csv_size_mb - parquet_size_mb
    
    print(f"\n  Compression: {compression_ratio:.1f}% smaller ({size_reduction:.1f} MB saved)")
    
    # Optionally delete CSV
    if delete_csv:
        print(f"  Deleting original CSV...", end=" ", flush=True)
        csv_path.unlink()
        print(f"✓")
    
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CSV files to Parquet format for better compression."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to CSV file to convert"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet file path (default: same as CSV with .parquet extension)"
    )
    parser.add_argument(
        "--delete-csv",
        action="store_true",
        help="Delete original CSV file after successful conversion"
    )
    
    args = parser.parse_args()
    
    try:
        parquet_path = convert_csv_to_parquet(
            csv_path=args.csv_file,
            output_path=args.output,
            delete_csv=args.delete_csv
        )
        print(f"\n✓ Successfully converted to: {parquet_path}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

