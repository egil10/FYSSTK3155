# PROJECT 3 Code Directory

This directory contains all code for the Player Value Prediction project, including data processing scripts, model implementations, and analysis notebooks.

## Directory Structure

```
Code/
├── Data/                    # Raw input data (see Data/README.md)
├── Data_Processed/          # Processed datasets (output)
├── Implementations/         # Data processing and model scripts
├── Notebooks/               # Analysis and modeling notebooks
├── Plots/                   # Generated visualizations
├── Tables/                  # Result tables
└── README.md                # This file
```

## Data Processing Scripts (`Implementations/`)

### Data Aggregation Notebooks

1. **`data_aggregating.ipynb`**
   - Basic feature extraction
   - Creates: `nn_tabular_dataset.csv`, `rnn_dataset.npz`, `meta.csv`
   - Features: static player info, per-game events, aggregated statistics

2. **`data_agg_cumlag.ipynb`**
   - Adds cumulative and lag features
   - Creates: `cumlag_nn_tabular_dataset.csv`, `cumlag_rnn_dataset.npz`, `cumlag_meta.csv`
   - Features: all basic features + cumulative stats + lag_10 features

3. **`data_agg_nationality.ipynb`**
   - Adds nationality one-hot encoding (184 countries)
   - Creates: `nat_nn_tabular_dataset.csv`, `nat_rnn_dataset.npz`, `nat_meta.csv`
   - Features: all cumulative/lag features + nationality (1/0 encoding)
   - Uses 1/0 encoding for all categorical features (foot, position, nationality)

### Utility Scripts

- **`convert_csv_to_parquet.py`**: Converts large CSV files to Parquet format for efficient storage
- **`prepare_data.py`**: Data preparation utilities (train/test splitting, standardization)
- **`rnn.ipynb`**: RNN model implementation for sequential player data

### R Scripts

- **`descriptive_stats.R`**: Statistical analysis and visualizations
- **`descriptive_nationality.R`**: Nationality-specific analysis

## Analysis Notebooks (`Notebooks/`)

### Regression Models

- **`ridge_analysis.ipynb`**: Ridge regression baseline model
  - Hyperparameter tuning
  - Performance evaluation
  - Feature importance

### Neural Network Models

- **`NN_analysis.ipynb`**: Feed-forward neural network (MLP)
  - Basic feature set
  - Architecture exploration
  - Comparison with Ridge regression

- **`NN_analysis_nationality.ipynb`**: MLP with nationality features
  - Extended feature set (212 features)
  - Impact of nationality on predictions
  - Performance comparison

- **`test_NN.ipynb`**: Neural network testing and validation

## Data Flow

```
Raw Data (Data/)
    ↓
[Data Processing Scripts]
    ↓
Processed Data (Data_Processed/)
    ↓
[Analysis Notebooks]
    ↓
Results (Plots/, Tables/)
```

## Running the Code

### 1. Data Processing

Run the data aggregation notebooks in order:

```bash
# Start with basic features
jupyter notebook Implementations/data_aggregating.ipynb

# Then add cumulative/lag features
jupyter notebook Implementations/data_agg_cumlag.ipynb

# Finally add nationality features
jupyter notebook Implementations/data_agg_nationality.ipynb
```

### 2. Convert to Parquet (if needed)

For large datasets:

```bash
python Implementations/convert_csv_to_parquet.py
```

### 3. Run Analysis

```bash
# Baseline model
jupyter notebook Notebooks/ridge_analysis.ipynb

# Neural networks
jupyter notebook Notebooks/NN_analysis.ipynb
jupyter notebook Notebooks/NN_analysis_nationality.ipynb
```

## Output Files

All processed datasets are saved to `Data_Processed/`:

**File Naming Convention:**
- `{prefix}nn_tabular_dataset.csv` - Tabular data for ML models
- `{prefix}rnn_dataset.npz` - RNN sequence data
- `{prefix}meta.csv` - Metadata (player_id, valuation_date)

**Prefixes:**
- No prefix: Basic features
- `cumlag_`: Cumulative + lag features
- `nat_`: Nationality-enhanced features

**File Formats:**
- CSV: Human-readable, large file size
- Parquet: Compressed, efficient (97% size reduction)
- NPZ: NumPy compressed format for arrays

## Important Notes

1. **Data Location**: All scripts expect data in `Data/` and output to `Data_Processed/`
2. **Path Handling**: Scripts use `Path.cwd().parent` to work from any directory
3. **Memory**: Large datasets may require significant RAM (8GB+ recommended)
4. **Processing Time**: Data aggregation can take 10-20 minutes
5. **Git**: Large CSV files are gitignored; use Parquet for version control

## Dependencies

See `requirements.txt` for full list. Key packages:
- `pandas`, `numpy` - Data processing
- `torch` - Neural network models
- `scikit-learn` - Traditional ML models
- `matplotlib`, `seaborn` - Visualization
- `pyarrow` - Parquet file support

## Troubleshooting

**Issue**: "File not found" errors
- **Solution**: Ensure you're running from the correct directory or check paths in scripts

**Issue**: Out of memory errors
- **Solution**: Process datasets one at a time, or use Parquet format to reduce memory usage

**Issue**: Slow processing
- **Solution**: Data aggregation is computationally intensive; be patient or optimize scripts

---

For data file documentation, see [Data/README.md](Data/README.md).

