# Code Directory

Contains all data processing and modeling code for the Player Value Prediction project.

## Directory Structure
```
Code/
├── Data/               # Raw input CSVs (players.csv, etc.)
├── Data_Processed/     # Processed outputs (see below)
├── Implementations/    # Key processing notebooks
├── Notebooks/          # Analysis and modeling
└── ...
```

## Key Processing Workflows

Run these notebooks to generate the datasets:

### 1. Core Features
**Notebook:** `Implementations/generate_player_core_features.ipynb`  
**Output:** `Data_Processed/player_core_features.csv`  
**What it does:**
- Extracts basic player stats and game events.
- Computes cumulative totals and lag (10-game) features.
- Saves a clean CSV for analysis.

### 2. Extended Features (Nationality + Parquet)
**Notebook:** `Implementations/generate_player_extended_features.ipynb`  
**Output:** `Data_Processed/player_extended_features.parquet`  
**What it does:**
- Includes all core features.
- Adds one-hot encoded nationality data (184 countries).
- Saves directly to **Parquet** format for efficiency (snappy compression).

## Helper Scripts
- **`convert_csv_to_parquet.py`**: Manual CSV-to-Parquet converter utility.
- **`prepare_data.py`**: Shared utilities for data split/scaling.

## Usage
1. Place raw data in `Data/`.
2. Run the desired notebook in `Implementations/` to generate `Data_Processed/` files.
3. Run analysis notebooks in `Notebooks/` using the generated data.

