## Transfermarkt Data Guide

This folder stores both the raw Transfermarkt exports and the derived
Premier-League feature table. Raw CSVs are large and remain ignored by git,
while the aggregated `pl_team_features.*` files are whitelisted so the project
can ship a ready-made dataset snapshot.

### Raw CSV bundle _(ignored by git)_
Drop the official Transfermarkt CSVs here with the original filenames:

```
appearances.csv        club_games.csv      competitions.csv
clubs.csv              games.csv           game_events.csv
game_lineups.csv       players.csv         player_valuations.csv
transfers.csv
```

These files can total several hundred megabytes, so keep them local.

### Build the Premier League feature table
From `PROJECT 3` run:

```bash
python Code/Implementations/Transfermarkt_data.py \
  --start-season 2021 \
  --end-season 2025 \
  --output Code/Data/pl_team_features.csv
```

Key command-line options:
- `--data-dir` – alternate location of the raw CSVs (defaults to this folder).
- `--output` – `.csv` or `.parquet` path for the aggregated dataset.
- `--competition-id` – Transfermarkt competition code (default `GB1` for EPL).

### What’s inside `pl_team_features.*`
Each row is a club-game entry designed to mirror the tidyverse pipeline we used
in R. Below is a full column reference so you know exactly what is persisted and
why it exists.

#### Match metadata & momentum
| Column | Description / motivation |
| ------ | ----------------------- |
| `game_id`, `competition_id`, `competition_type` | Canonical identifiers for merges and for filtering across competitions. |
| `season`, `round`, `date` | Temporal context for slicing seasons or evaluating trends. |
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
| `squad_value_mean`, `squad_value_max`, `squad_value_min`, `squad_value_total` | Financial strength snapshots using latest Transfermarkt valuations. |
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

Feel free to inspect the CSV directly or convert to Parquet for smaller file
size and typed columns. The 2021–2025 Premier League slice is ~1.5 MB as CSV and
even smaller as Parquet, so it is checked into git for convenience. Rerun the
script whenever you refresh the raw data and commit the new snapshot if needed.

