# Data Directory

This directory contains the required CSV files for the player value prediction project.

## Dataset Source

The data comes from the **Football Data from Transfermarkt** dataset on Kaggle:

🔗 **https://www.kaggle.com/datasets/davidcariboo/player-scores**

## Required Files

This directory must contain the following CSV files:

### 1. `players.csv`
Contains player biographical and physical information.

**Key columns used:**
- `player_id` - Unique identifier for each player
- `date_of_birth` - Player's date of birth (used to calculate age)
- `height_in_cm` - Player's height in centimeters
- `foot` - Preferred foot (left/right/both)
- `position` - Player's position (standardized to GK/DEF/MID/ATT)

### 2. `player_valuations.csv`
Contains historical market valuations for players over time.

**Key columns used:**
- `player_id` - Unique identifier for each player
- `date` - Date of the valuation
- `market_value_in_eur` - Player's market value in euros (target variable)
- `player_club_domestic_competition_id` - League/competition ID (used to identify Big-5 leagues: GB1, ES1, IT1, DE1, FR1)

### 3. `game_events.csv`
Contains match events (goals, assists, cards, substitutions) from football games.

**Key columns used:**
- `player_id` - Player who performed the event
- `player_assist_id` - Player who provided the assist (for goals)
- `player_in_id` - Player substituted in
- `game_id` - Unique identifier for the match
- `date` - Date of the match
- `type` - Type of event (Goals, Cards, Substitutions)
- `description` - Event description (used to distinguish yellow/red cards)
- `minute` - Minute when the event occurred

**Event types extracted:**
- Goals (for goals scored)
- Assists (from goal events)
- Yellow cards
- Red cards
- Substitutions (both in and out)

## Setup Instructions

1. Download the dataset from the Kaggle link above
2. Extract the ZIP file
3. Copy the three required CSV files (`players.csv`, `player_valuations.csv`, `game_events.csv`) into this directory
4. Ensure all files are in CSV format and properly formatted

## Data Processing

The notebook `data_aggregating.ipynb` uses this data to:
- Extract static player features (height, position, foot preference, age)
- Build per-game event features (goals, assists, cards, substitutions)
- Create time-series sequences for RNN models
- Generate aggregated features for tabular/neural network models
- Prepare training datasets for player value prediction

## Notes

- All date columns should be parseable as datetime
- Missing values in key columns are handled during processing
- The code expects numeric IDs for players and games
- Market values should be numeric values in euros

