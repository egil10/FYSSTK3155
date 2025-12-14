# PROJECT 3: Player Value Prediction

This project focuses on predicting football player market values using machine learning models applied to Transfermarkt data. The project involves comprehensive data processing, feature engineering, and model comparison across multiple approaches including Ridge regression, feed-forward neural networks (MLP), and recurrent neural networks (RNN).

## Project Overview

The goal is to predict player market values (in euros) using historical performance data, player characteristics, and temporal patterns. The project explores different feature engineering strategies and model architectures to understand which factors most influence player valuations.

## Repository Structure

```
PROJECT 3/
├── Code/
│   ├── Data/                    # Raw Transfermarkt CSV files
│   │   ├── players.csv         # Player biographical data
│   │   ├── player_valuations.csv  # Historical market values
│   │   ├── game_events.csv     # Match events (goals, cards, etc.)
│   │   └── README.md           # Data documentation
│   │
│   ├── Data_Processed/          # Processed datasets (output)
│   │   ├── nn_tabular_dataset.csv      # Basic feature set
│   │   ├── cumlag_nn_tabular_dataset.csv  # Cumulative + lag features
│   │   ├── nat_nn_tabular_dataset.parquet  # Nationality-enhanced (Parquet)
│   │   ├── rnn_dataset.npz     # RNN sequences (basic)
│   │   ├── cumlag_rnn_dataset.npz  # RNN sequences (cumulative)
│   │   └── nat_rnn_dataset.npz     # RNN sequences (nationality)
│   │
│   ├── Implementations/        # Data processing scripts
│   │   ├── data_aggregating.ipynb      # Basic feature extraction
│   │   ├── data_agg_cumlag.ipynb       # Cumulative + lag features
│   │   ├── data_agg_nationality.ipynb  # Nationality features
│   │   ├── rnn.ipynb           # RNN model implementation
│   │   ├── prepare_data.py     # Data preparation utilities
│   │   └── convert_csv_to_parquet.py   # CSV to Parquet converter
│   │
│   ├── Notebooks/              # Analysis and modeling notebooks
│   │   ├── ridge_analysis.ipynb       # Ridge regression baseline
│   │   ├── NN_analysis.ipynb          # MLP neural network
│   │   ├── NN_analysis_nationality.ipynb  # MLP with nationality
│   │   └── test_NN.ipynb       # Neural network testing
│   │
│   ├── Plots/                  # Generated visualizations
│   ├── Tables/                 # Result tables
│   └── README.md               # Code documentation
│
└── README.md                    # This file
```

## Data Sources

The project uses the **Football Data from Transfermarkt** dataset available on Kaggle:

🔗 **https://www.kaggle.com/datasets/davidcariboo/player-scores**

### Required Data Files

Place the following CSV files in `Code/Data/`:

1. **players.csv** - Player biographical information
   - `player_id`, `date_of_birth`, `height_in_cm`, `foot`, `position`, `country_of_citizenship`

2. **player_valuations.csv** - Historical market values
   - `player_id`, `date`, `market_value_in_eur`, `player_club_domestic_competition_id`

3. **game_events.csv** - Match events
   - `player_id`, `game_id`, `date`, `type`, `description`, `minute`

## Feature Engineering

The project implements three progressively more complex feature sets:

### 1. Basic Features (`data_aggregating.ipynb`)
- Static features: height, age, position (one-hot), foot preference (one-hot), Big-5 league flag
- Per-game features: goals, assists, yellow cards, red cards, substitutions
- Aggregated features: mean and sum over last 20 games

### 2. Cumulative + Lag Features (`data_agg_cumlag.ipynb`)
- All basic features
- Cumulative statistics: total goals, assists, cards, substitutions up to valuation date
- Lag features: sum of events over last 10 games before valuation

### 3. Nationality-Enhanced Features (`data_agg_nationality.ipynb`)
- All cumulative + lag features
- Nationality one-hot encoding: 184 country features (1/0 encoding)
- All categorical features use 1/0 encoding instead of boolean

## Data Processing Workflow

### Step 1: Run Data Aggregation Scripts

Choose the feature set you want:

```bash
# Basic features
jupyter notebook Code/Implementations/data_aggregating.ipynb

# Cumulative + lag features
jupyter notebook Code/Implementations/data_agg_cumlag.ipynb

# Nationality-enhanced features
jupyter notebook Code/Implementations/data_agg_nationality.ipynb
```

These scripts generate:
- Tabular datasets (CSV) for traditional ML models
- RNN sequence datasets (NPZ) for recurrent models
- Metadata files (CSV) with player IDs and dates

### Step 2: Convert Large CSVs to Parquet (Optional)

For datasets exceeding 100MB, convert to Parquet format for Git:

```bash
python Code/Implementations/convert_csv_to_parquet.py
```

This creates compressed Parquet files that are much smaller (typically 95%+ reduction).

### Step 3: Run Analysis Notebooks

```bash
# Ridge regression baseline
jupyter notebook Code/Notebooks/ridge_analysis.ipynb

# Neural network analysis
jupyter notebook Code/Notebooks/NN_analysis.ipynb

# Neural network with nationality features
jupyter notebook Code/Notebooks/NN_analysis_nationality.ipynb
```

## Output Files

All processed datasets are saved to `Code/Data_Processed/`:

**Tabular Datasets:**
- `nn_tabular_dataset.csv` - Basic features (28 columns)
- `cumlag_nn_tabular_dataset.csv` - Cumulative + lag (28 columns)
- `nat_nn_tabular_dataset.csv` - With nationality (212 columns)
- `nat_nn_tabular_dataset.parquet` - Compressed version (3.3 MB vs 136 MB)

**RNN Datasets:**
- `rnn_dataset.npz` - Sequences for basic features
- `cumlag_rnn_dataset.npz` - Sequences for cumulative features
- `nat_rnn_dataset.npz` - Sequences with nationality features

**Metadata:**
- `meta.csv`, `cumlag_meta.csv`, `nat_meta.csv` - Player IDs and valuation dates

## Model Architectures

### Ridge Regression
- Baseline linear model with L2 regularization
- Hyperparameter tuning for alpha
- Group-based train/test split (by player_id) to prevent data leakage

### Feed-Forward Neural Network (MLP)
- Multi-layer perceptron with dropout
- Architecture: Input → 128 → 64 → 1
- Activation: ReLU with dropout (0.1)
- Optimizer: Adam with learning rate 5e-4

### Recurrent Neural Network (RNN)
- GRU-based architecture for sequential data
- Sequence length: 20 games
- Static features concatenated with hidden state
- Handles variable-length sequences with padding

## Key Features

- **Temporal Feature Engineering**: Cumulative statistics and lag features capture player performance trends
- **Nationality Analysis**: 184 country features explore geographic market value patterns
- **Data Leakage Prevention**: Group-based splitting ensures no player appears in both train and test sets
- **Multiple Model Comparison**: Ridge, MLP, and RNN models compared on same datasets
- **Efficient Storage**: Parquet format for large datasets (97% size reduction)

## Dependencies

```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn jupyter pyarrow
```

## Notes

- Large CSV files (>100MB) are gitignored; use Parquet format for version control
- All scripts use fixed random seeds for reproducibility
- Data processing can take 10-20 minutes depending on dataset size
- RNN models require more memory and computation time than tabular models

## Results and Analysis

See the analysis notebooks in `Code/Notebooks/` for:
- Model performance comparisons
- Feature importance analysis
- Hyperparameter optimization results
- Visualization of predictions and residuals

---

For detailed code documentation, see [Code/README.md](Code/README.md).
