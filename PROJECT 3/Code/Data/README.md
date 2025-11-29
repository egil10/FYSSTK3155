## Transfermarkt Data Guide

This folder stores both the raw Transfermarkt exports and the derived
feature tables. Raw CSVs are large and remain ignored by git,
while the aggregated feature files (`game_features.parquet` and `pl_team_features.*`) 
are whitelisted so the project can ship ready-made dataset snapshots.

**Note:** The main dataset (`game_features.parquet`) is saved in Parquet format for 
optimal compression (~74% smaller than CSV) while preserving all data types.

### Raw CSV bundle _(ignored by git)_
Drop the official Transfermarkt CSVs here with the original filenames:

```
appearances.csv        club_games.csv      competitions.csv
clubs.csv              games.csv           game_events.csv
game_lineups.csv       players.csv         player_valuations.csv
transfers.csv
```

These files can total several hundred megabytes, so keep them local.

You can also enrich the dataset with public player-rating data such as the
[Player Performance Scores dataset on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores/),
which provides per-match performance metrics that can be joined on player IDs
for modelling work.

### Build feature tables

#### Game-level features (one row per game) - **RECOMMENDED**

Build comprehensive lag-based features for all games with strict temporal filtering:

```bash
python Code/Implementations/Transfermarkt_data.py
```

This produces `game_features.parquet` (default) with:
- **One row per game** with features for both home and away teams
- **Strict lag-based features** following naming convention: `{team}_{feature}_{stat}_L{N}`
- **Multi-window lags**: L1, L3, L5, L10, L20 for all features
- **No data leakage**: All features use only data from matches before the current game
- **Interaction features**: `diff_*` features for relative strength metrics
- **Column structure**: Metadata (ID_*) → RESULT (W/D/L) → Predictor features

**Column naming convention:**
- Metadata: `ID_GAME`, `ID_COMPETITION`, `ID_DATE`, etc. (never used in training)
- Target: `RESULT` (W/D/L based on home/away goals)
- Features: `home_points_sum_L3`, `away_goals_mean_L10`, `diff_squad_value_total_L5`, etc.

**File format:**
- Default output is **Parquet** (73% smaller than CSV, ~64 MB vs ~243 MB)
- Use `--also-save-csv` if you need CSV format
- Use `--output` to specify custom path/format

#### Premier League team features (legacy, club-level)
For the original Premier League club-level dataset:

```bash
python Code/Implementations/Transfermarkt_data.py \
  --start-season 2021 \
  --end-season 2025 \
  --competition-id GB1 \
  --output Code/Data/pl_team_features.csv
```

**Key command-line options:**
- `--data-dir` – alternate location of the raw CSVs (defaults to this folder)
- `--output` – output path (default: `Code/Data/game_features.parquet`)
- `--also-save-csv` – also save CSV version alongside Parquet
- `--start-season` – first season to include (optional, defaults to all seasons)
- `--end-season` – last season to include (optional, defaults to all seasons)
- `--competition-id` – competition code to filter (optional, defaults to all competitions)

**Feature engineering:**
- All features are **strictly lagged** (no data leakage)
- Multi-window rolling statistics: L1, L3, L5, L10, L20
- Feature groups: club performance, appearances, lineups, events
- Interaction features: home-away differences for key metrics
- A list of predictive columns is written to `Code/Data/predictive_features.txt`

**Converting CSV to Parquet:**
If you have an existing CSV file, convert it to save space:
```bash
python Code/Implementations/convert_csv_to_parquet.py Code/Data/game_features.csv
```

### What's inside the feature tables

#### `game_features.parquet` (one row per game) - **MAIN DATASET**

**Structure:**
1. **Metadata columns** (ID_* prefix): Game identifiers, dates, teams, managers, etc.
   - Never used in model training
   - Examples: `ID_GAME`, `ID_DATE`, `ID_HOME_TEAM`, `ID_REFIREE`

2. **Target variable** (`RESULT`): Match outcome
   - Values: "W" (home win), "D" (draw), "L" (home loss)
   - Computed from `ID_HOME_GOALS` and `ID_AWAY_GOALS`

3. **Predictor features** (lagged, no leakage):
   - **Club performance**: `home_points_sum_L3`, `away_goal_difference_mean_L10`, etc.
   - **Appearances**: `home_goals_mean_L5`, `away_assists_mean_L1`, etc.
   - **Lineup/squad**: `home_squad_value_total_mean_L20`, `away_avg_age_mean_L5`, etc.
   - **Events**: `home_shots_mean_L3`, `away_possession_proxy_events_mean_L10`, etc.
   - **Interactions**: `diff_points_L5`, `diff_squad_value_total_L10`, etc.

**Feature naming pattern:**
```
{team}_{feature}_{statistic}_L{window}
```
- `team`: `home` or `away`
- `feature`: e.g., `points`, `goals`, `squad_value_total`, `shots`
- `statistic`: `mean`, `sum`, `max`, `min`
- `window`: `1`, `3`, `5`, `10`, `20` (number of previous matches)

**Example features:**
- `home_points_sum_L3` - Sum of points from last 3 matches (home team)
- `away_goals_mean_L10` - Average goals scored in last 10 matches (away team)
- `diff_squad_value_total_L5` - Difference in squad value (home - away) over last 5 matches

#### `pl_team_features.*` (one row per club-game)
Each row is a club-game entry designed to mirror the tidyverse pipeline we used
in R. Below is a full column reference so you know exactly what is persisted and
why it exists.

#### Match metadata & momentum
| Column | Description / motivation |
| ------ | ----------------------- |
| `game_id`, `competition_id`, `competition_type` | Canonical identifiers for merges and for filtering across competitions. |
| `season`, `round`, `round_number`, `date` | Temporal context for slicing seasons or evaluating trends (round number is parsed to an integer so matchdays can be sorted reliably). |
| `home_*`, `away_*` info (`*_club_id`, `*_club_name`, `*_goals`, `*_position`, `*_manager_name`, `*_formation`) | Everything needed to reconstruct the match narrative and target variables. |
| `stadium`, `attendance`, `referee`, `url` | Venue + reference metadata. |
| `club_id`, `opponent_id`, `own_goals`, `opponent_goals`, `hosting`, `is_home`, `is_win` | Viewpoint-normalised fields for per-club modelling. |
| `goal_difference`, `points` | Base metrics for result-based tasks (3/1/0 points plus margin). |
| `prev_points`, `prev_goal_difference`, `prev_goals_scored`, `prev_goals_conceded` | Previous-match form indicators (shifted lag-1 values for each club; `NaN` if no prior match).

#### Appearance aggregates
| Column | Description / motivation |
| ------ | ----------------------- |
| `goals_mean`, `goals_max`, `goals_min` | Distribution of individual scoring output within the squad that match. |
| `assists_mean`, `assists_max`, `assists_min` | Same idea for creativity. |
| `minutes_mean`, `minutes_max`, `minutes_min` | Captures rotation level and workload spread. |
| `yellow_cards_sum`, `red_cards_sum` | Discipline summary from appearance logs.

#### Lineup structure & squad economics
| Column | Description / motivation |
| ------ | ----------------------- |
| `n_players`, `n_starters`, `starters_percentage`, `n_captains` | Squad size, starting XI count, rotation intensity, leadership continuity. |
| `avg_height`, `min_height`, `max_height`, `height_spread` | Physical profile and diversity for aerial/fitness analyses. |
| `defenders`, `midfielders`, `forwards`, `others` | Positional mix for tactical shape inference. |
| `avg_age`, `median_age` | Experience vs youth balance. |
| `squad_value_mean`, `squad_value_max`, `squad_value_min`, `squad_value_total` | Financial strength using player valuations as of each match date (latest known valuation before kickoff). |
| `avg_market_value_starting_xi` | Quality of the eleven actually deployed. |
| `new_signings_played` | Count of players signed that season who appeared—proxy for integration and churn. |
| `continuity_index` | Fraction of starters retained from the previous match (intersection / 11). |
| `missing_key_players` | Number of top-5 market-value players absent from the starting XI (injuries, rotation).

#### Event counts & tactical proxies
| Column | Description / motivation |
| ------ | ----------------------- |
| `n_events` | Total play-by-play entries for involvement. |
| `shots`, `goals_event` | Direct attacking volume and conversions. |
| `fouls` | Defensive/aggression indicator. |
| `subs`, `n_subs_used` | Substitution opportunities vs actual usage (fitness/strategy). |
| `passes`, `touches` | Possession-oriented actions (when present in the feed). |
| `possession_proxy_events` | `(shots + passes + touches) / n_events` as a crude possession stand-in.

### Using the dataset

**Reading Parquet files:**
```python
import pandas as pd

# Read the main dataset
df = pd.read_parquet("Code/Data/game_features.parquet")

# All features are ready for time-series cross-validation
# Metadata columns (ID_*) should be excluded from training
# RESULT is the target variable
# All other columns are lagged predictors
```

**File sizes:**
- `game_features.parquet`: ~64 MB (compressed from 243 MB CSV)
- Compression ratio: ~74% smaller
- Preserves all data types and is faster to read/write

**Time-series cross-validation:**
The dataset is structured for strict temporal validation:
- Sort by `ID_DATE` before splitting
- Use only data before the test period for training
- All features are pre-lagged, so no leakage risk

Rerun the script whenever you refresh the raw data and commit the new snapshot if needed.
