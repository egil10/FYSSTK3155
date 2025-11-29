# Transfermarkt Data Guide

This folder stores both the raw Transfermarkt exports and the derived feature tables. Raw CSVs are large and remain ignored by git, while the aggregated feature file (`game_features.parquet`) is whitelisted so the project can ship a ready-made dataset snapshot.

**Note:** The main dataset (`game_features.parquet`) is saved in Parquet format for optimal compression (~74% smaller than CSV) while preserving all data types.

---

## Raw CSV Bundle _(ignored by git)_

Drop the official Transfermarkt CSVs here with the original filenames:

```
appearances.csv        club_games.csv      competitions.csv
clubs.csv              games.csv           game_events.csv
game_lineups.csv       players.csv         player_valuations.csv
transfers.csv
```

These files can total several hundred megabytes, so keep them local.

You can also enrich the dataset with public player-rating data such as the [Player Performance Scores dataset on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores/), which provides per-match performance metrics that can be joined on player IDs for modeling work.

---

## Building the Feature Dataset

### Game-level features (one row per game) - **MAIN DATASET**

Build comprehensive lag-based features for all games with strict temporal filtering and optimized lag windows:

```bash
python Code/Implementations/Transfermarkt_data.py
```

This produces `game_features.parquet` (default) with:
- **One row per game** with features for both home and away teams
- **Optimized lag windows** for better feature engineering efficiency
- **Strict temporal filtering** (no data leakage)
- **Column structure**: Metadata (ID_*) → RESULT (W/D/L) → Predictor features

**Key command-line options:**
- `--data-dir` – alternate location of the raw CSVs (defaults to this folder)
- `--output` – output path (default: `Code/Data/game_features.parquet`)
- `--also-save-csv` – also save CSV version alongside Parquet
- `--start-season` – first season to include (optional, defaults to all seasons)
- `--end-season` – last season to include (optional, defaults to all seasons)
- `--competition-id` – competition code to filter (optional, defaults to all competitions)

**Example: Filter to Premier League only**
```bash
python Code/Implementations/Transfermarkt_data.py --competition-id GB1
```

---

## Dataset Structure

### Column Organization

1. **Metadata columns** (ID_* prefix, ~22 columns)
   - Never used in model training
   - Examples: `ID_GAME`, `ID_DATE`, `ID_HOME_TEAM`, `ID_REFIREE`, `ID_HOME_GOALS`, `ID_AWAY_GOALS`
   - Include game identifiers, dates, teams, managers, stadiums, attendance, etc.

2. **Target variable** (`RESULT` - 1 column)
   - Values: "W" (home win), "D" (draw), "L" (home loss)
   - Computed from `ID_HOME_GOALS` and `ID_AWAY_GOALS`

3. **Predictor features** (~220 columns, all lagged, no leakage)

### Feature Naming Convention

All features follow the pattern:
```
{team}_{feature}_{statistic}_L{window}
```

Where:
- `team`: `home` or `away`
- `feature`: e.g., `points`, `goals`, `squad_value_total`, `goals_scored`
- `statistic`: `mean`, `sum`, `max`, `min`
- `window`: Lag window size (see optimized windows below)

**Example features:**
- `home_points_sum_L3` - Sum of points from last 3 matches (home team)
- `away_goals_mean_L10` - Average goals scored in last 10 matches (away team)
- `home_squad_value_total_mean_L20` - Average squad value over last 20 matches

---

## Optimized Lag Windows

The dataset uses **optimized lag windows** to reduce dimensionality while maintaining predictive power:

### Fast-Changing Features (L3, L10, L20)
These features change frequently and benefit from multiple windows:
- **Club Performance**: `points`, `goal_difference`, `goals_scored`, `goals_conceded`
- **Appearance Stats**: `goals` (mean/max/min), `assists`, `minutes_played`, `yellow_cards`, `red_cards`
- **Event Features**: `goals_event`

### Slow-Changing Features (L5, L20)
These structural features change slowly and only need longer windows:
- **Lineup/Squad Composition**: `n_starters`, `avg_height`, `min_height`, `max_height`, `height_spread`, `defenders`, `midfielders`, `forwards`, `avg_age`, `median_age`, `squad_value_total`, `avg_market_value_starting_xi`, `continuity_index`, `new_signings_played`, `starters_percentage`

### Interaction Features (diff_*)
Relative strength metrics (home - away):
- **Fast windows** (L3, L10, L20): `diff_points`, `diff_goal_difference`, `diff_goals_scored`
- **Slow windows** (L5, L20): `diff_squad_value_total`, `diff_avg_age`, `diff_avg_height`

---

## Feature Groups

### 1. Club Performance Features
**Lag windows:** L3, L10, L20
- `home/away_points_sum_L{N}` - Points accumulated
- `home/away_goal_difference_mean_L{N}` - Average goal difference
- `home/away_goals_scored_mean_L{N}` - Average goals scored
- `home/away_goals_conceded_mean_L{N}` - Average goals conceded

### 2. Appearance-Based Performance Features
**Lag windows:** L3, L10, L20
- `home/away_goals_{mean|max|min}_L{N}` - Goal scoring distribution
- `home/away_assists_mean_L{N}` - Average assists
- `home/away_minutes_played_mean_L{N}` - Average minutes played
- `home/away_yellow_cards_sum_L{N}` - Yellow cards accumulated
- `home/away_red_cards_sum_L{N}` - Red cards accumulated

### 3. Lineup/Squad Composition Features
**Lag windows:** L5, L20
- `home/away_n_starters_mean_L{N}` - Average number of starters
- `home/away_avg_height_mean_L{N}` - Average squad height
- `home/away_{min|max}_height_mean_L{N}` - Height range
- `home/away_height_spread_mean_L{N}` - Height diversity
- `home/away_defenders_mean_L{N}` - Number of defenders
- `home/away_midfielders_mean_L{N}` - Number of midfielders
- `home/away_forwards_mean_L{N}` - Number of forwards
- `home/away_avg_age_mean_L{N}` - Average squad age
- `home/away_median_age_mean_L{N}` - Median squad age
- `home/away_squad_value_total_mean_L{N}` - Total squad market value
- `home/away_avg_market_value_starting_xi_mean_L{N}` - Average starting XI value
- `home/away_continuity_index_mean_L{N}` - Lineup continuity (0-1)
- `home/away_new_signings_played_mean_L{N}` - New signings in lineup
- `home/away_starters_percentage_mean_L{N}` - Percentage of squad that starts

### 4. Event Features
**Lag windows:** L3, L10, L20
- `home/away_goals_event_sum_L{N}` - Goals from event data

### 5. Interaction Features (Relative Strength)
**Lag windows:** Varies by feature type
- `diff_points_L{3|10|20}` - Point difference (home - away)
- `diff_goal_difference_L{3|10|20}` - Goal difference gap
- `diff_goals_scored_L{3|10|20}` - Goals scored difference
- `diff_squad_value_total_L{5|20}` - Squad value difference
- `diff_avg_age_L{5|20}` - Age difference
- `diff_avg_height_L{5|20}` - Height difference

---

## Removed/Useless Features

The following features were removed during optimization:

### Completely Missing (100% NA)
- `home/away_missing_key_players_mean_L{1|3|5|10|20}` - Feature was leaky and always missing

### Constant Values (Always Zero)
- `home/away_{shots|fouls|passes|touches|possession_proxy_events}_mean_L{1|3|5|10|20}` - All event features were constant at 0
- `diff_{shots|possession_proxy_events}_L{1|3|5|10|20}` - Derived features from constant values

**Total removed:** ~70 useless columns (18% reduction)

---

## Feature Engineering Details

### Strict Temporal Filtering (No Data Leakage)
- All features use `shift(1)` to ensure only past matches are included
- Rolling windows use data strictly before the current match timestamp
- Player valuations use latest valuation ≤ match date (strict temporal alignment)

### Data Quality
- Missing values handled with median imputation for numeric features
- All feature columns converted to numeric types before aggregation
- Features with >50% missing values filtered out

### Optimizations
- Reduced lag windows: Fast features (3 windows) vs Slow features (2 windows)
- Removed redundant/collinear features
- Removed features with zero variance or constant values
- Result: ~220 predictive features (down from ~385, 43% reduction)

---

## Using the Dataset

### Reading Parquet Files

```python
import pandas as pd

# Read the main dataset
df = pd.read_parquet("Code/Data/game_features.parquet")

print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# Separate metadata, target, and predictors
metadata_cols = [col for col in df.columns if col.startswith('ID_')]
target_col = 'RESULT'
predictor_cols = [col for col in df.columns 
                  if col not in metadata_cols and col != target_col]

print(f"Metadata columns: {len(metadata_cols)}")
print(f"Target: {target_col}")
print(f"Predictor features: {len(predictor_cols)}")
```

### Time-Series Cross-Validation

The dataset is structured for strict temporal validation:
1. Sort by `ID_DATE` before splitting
2. Use only data before the test period for training
3. All features are pre-lagged, so no leakage risk

**Example temporal split:**
```python
# Sort by date
df = df.sort_values('ID_DATE')

# Split at 80th percentile of dates
split_date = df['ID_DATE'].quantile(0.8)
train = df[df['ID_DATE'] < split_date]
test = df[df['ID_DATE'] >= split_date]
```

### Predictive Features List

A list of all predictive feature columns is automatically generated and saved to:
```
Code/Data/predictive_features.txt
```

This file is updated each time you rebuild the dataset.

---

## File Information

**File sizes (approximate):**
- `game_features.parquet`: ~60-70 MB (compressed)
- `game_features.csv`: ~240-250 MB (uncompressed)
- Compression ratio: ~74% smaller as Parquet

**Dataset statistics:**
- Rows: ~74,000 games (varies by data coverage)
- Columns: ~220-240 (metadata + target + predictors)
- Predictive features: ~220
- Time span: All seasons in raw data (typically 2012-2025)

---

## Converting CSV to Parquet

If you have an existing CSV file, convert it to save space:

```bash
python Code/Implementations/convert_csv_to_parquet.py Code/Data/game_features.csv
```

Optional flags:
- `--output` - Specify output path (default: replaces .csv with .parquet)
- `--delete-csv` - Delete original CSV after conversion

---

## Regenerating the Dataset

Rerun the build script whenever you:
- Update the raw CSV files
- Want to refresh the feature calculations
- Need to filter by competition or season

The script automatically:
- Processes all raw CSV files
- Computes all lag-based features
- Filters out useless features
- Saves to Parquet format
- Updates `predictive_features.txt`

**Note:** The build process takes ~5-10 minutes depending on your system. Progress is displayed in the terminal.
