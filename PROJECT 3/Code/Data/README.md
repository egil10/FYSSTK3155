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

#### Match metadata
| Column | Description / motivation |
| ------ | ----------------------- |
| `game_id` | Unique Transfermarkt match id; key for merges with other datasets. |
| `competition_id`, `competition_type` | Competition code (`GB1`) and league vs cup indicator; handy if you extend to other leagues later. |
| `season`, `round`, `date` | Temporal context for aggregations, plotting trends, or filtering season windows. |
| `home_club_id`, `away_club_id`, `home_club_name`, `away_club_name` | Club identifiers straight from Transfermarkt; allow lookups, joins, or quick labeling. |
| `home_club_goals`, `away_club_goals`, `aggregate` | Ground-truth scoreline (plus aggregate text) for model targets or sanity checks. |
| `home_club_position`, `away_club_position` | League-table rank at kickoff; helps compare squads with different form. |
| `home_club_manager_name`, `away_club_manager_name` | Manager context for narrative or future manager-based analysis. |
| `stadium`, `attendance`, `referee`, `url` | Venue metadata, crowd size, officiating crew, and canonical Transfermarkt link for reproducibility. |
| `home_club_formation`, `away_club_formation` | Formation strings for quick tactical summaries. |
| `club_id` | The specific club for this row (matches either `home_club_id` or `away_club_id`). |
| `own_goals`, `opponent_goals` | Score from the perspective of `club_id`; simplifies home/away toggling. |
| `own_position`, `opponent_position` | Table rank of club vs opponent in the same format. |
| `own_manager_name`, `opponent_manager_name` | Mirrors manager info but aligned with `club_id`. |
| `opponent_id` | Explicit key to the opposing club (useful for graph/network work). |
| `hosting` | Textual value (“Home”/“Away”) from Transfermarkt for human-readable output. |
| `is_home`, `is_win` | Boolean flags for modelling classification targets (home advantage, match outcome). |

#### Appearance aggregates
| Column | Description / motivation |
| ------ | ----------------------- |
| `goals_mean`, `goals_max`, `goals_min` | Average/best/worst individual scoring contribution across appearances for the club in that match; captures distribution rather than just total goals. |
| `assists_mean`, `assists_max`, `assists_min` | Same idea for assists—useful for creative output metrics. |
| `minutes_mean`, `minutes_max`, `minutes_min` | Captures how minutes were shared across the squad (e.g., rotated XI vs full-strength). |
| `yellow_cards_sum`, `red_cards_sum` | Total disciplinary actions drawn from the appearance table (complements event-level fouls). |

#### Lineup structure
| Column | Description / motivation |
| ------ | ----------------------- |
| `n_players` | Number of player entries in the lineup (should be >=11, includes bench). |
| `n_captains` | Count of captaincy flags—indicates leadership continuity or anomalies. |
| `avg_height`, `min_height`, `max_height` | Squad physical profile for aerial/physical matchup studies. |
| `defenders`, `midfielders`, `forwards`, `others` | Counts of players in each positional bucket based on Transfermarkt positions, revealing tactical leanings (e.g., back three vs back four). |

#### Event counts
| Column | Description / motivation |
| ------ | ----------------------- |
| `n_events` | Total events recorded for the club in play-by-play (proxy for involvement). |
| `shots`, `fouls`, `subs`, `goals_event` | Counts of specific event types to gauge attacking volume, discipline, rotation, and goals. |

#### Squad market value
| Column | Description / motivation |
| ------ | ----------------------- |
| `squad_value_mean`, `squad_value_max`, `squad_value_min` | Aggregate market values (Transfermarkt) of players tied to the club, letting you relate financial muscle to performance. |

Feel free to inspect the CSV directly or convert to Parquet for smaller file
size and typed columns. The 2021–2025 Premier League slice is ~1.5 MB as CSV and
even smaller as Parquet, so it is checked into git for convenience. Rerun the
script whenever you refresh the raw data and commit the new snapshot if needed.

