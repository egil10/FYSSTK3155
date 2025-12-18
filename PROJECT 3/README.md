# PROJECT 3: Player Value Prediction

Predicting football player market values using machine learning models on Transfermarkt data.

## Project Overview
This project processes historical player data (biographical, events, performance) to predict valuations. It explores Ridge Regression and Neural Networks (MLP/RNN) with progressively complex feature sets.

## Structure
```
PROJECT 3/
├── Code/
│   ├── Data/                # Raw CSVs (see Code/Data/README.md)
│   ├── Data_Processed/      # Output datasets
│   ├── Implementations/     # Processing scripts
│   └── Notebooks/           # Analysis & Models
└── README.md
```

## Workflows & Datasets

### 1. Core Feature Set
- **Source**: `Code/Implementations/data_agg_cumlag.ipynb`
- **Output**: `player_core_features.csv`
- **Content**: Player stats, cumulative performance, and 10-game lag features.

### 2. Extended Feature Set (Nationality)
- **Source**: `Code/Implementations/data_agg_nationality.ipynb`
- **Output**: `player_extended_features.parquet`
- **Content**: Includes all core features plus 184 one-hot encoded nationality features. Saved as Parquet for performance.

## Getting Started

1. **Setup Data**:
   - Download `players.csv`, `player_valuations.csv`, `game_events.csv` from [Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores).
   - Place them in `Code/Data/`.

2. **Generate Data**:
   - Run the notebooks in `Code/Implementations/` to build the datasets.

3. **Run Analysis**:
   - Explore models in `Code/Notebooks/`.

## Dependencies
`pandas`, `numpy`, `torch`, `scikit-learn`, `matplotlib`, `pyarrow`, `jupyter`
