## Project 3 – Transfermarkt Aggregation

This project turns the raw Transfermarkt exports in `Code/Data` into a tidy,
team-game level dataset for the English Premier League (competition id `GB1`,
seasons 2021–2025 by default). The output can be plugged into notebooks in
`Code/Notebooks` for modelling or visualization.


### Repository Layout
- `Code/Implementations/Transfermarkt_data.py` – main data-build script
- `Code/Data/` – raw CSV exports (ignored by git, keep them locally)
- `Code/Notebooks/`, `Code/Plots/`, `Code/Tables/` – downstream analysis


### Data Requirements
1. Download the Transfermarkt CSV bundle (matches, lineups, events, etc.).
2. Place the files in `Code/Data/` and keep the original filenames:
   `appearances.csv`, `club_games.csv`, `clubs.csv`, `competitions.csv`,
   `games.csv`, `game_events.csv`, `game_lineups.csv`, `players.csv`,
   `player_valuations.csv`, `transfers.csv`.
3. The folder contains a `.gitkeep` so the directory stays in git while the
   large CSVs remain ignored.


### Building the Aggregated Dataset
The script reproduces the tidyverse pipeline from R using pandas. It:
1. Filters Premier League matches between the requested seasons.
2. Aggregates appearance stats (goals, assists, minutes, cards).
3. Summarizes lineup structure (height, captains, positional buckets).
4. Counts event-level actions (shots, fouls, substitutions, goals).
5. Collapses player market values into club-level squad value stats.
6. Produces one row per club per match with home/away information.

Run it from the project root:
```bash
python Code/Implementations/Transfermarkt_data.py \
  --start-season 2021 \
  --end-season 2025 \
  --competition-id GB1 \
  --output Code/Data/pl_team_features.csv
```

Options:
- `--data-dir` path to the folder with the raw CSVs (defaults to `Code/Data`).
- `--output` target file (`.csv` or `.parquet`). Defaults to
  `Code/Data/pl_team_features.csv`.

After it finishes you should see a confirmation similar to:
```
Built 1520 team-game rows.
Saved dataset to Code/Data/pl_team_features.csv
```


### Tips
- Re-run the script whenever the raw data is refreshed; downstream notebooks
  can then point to the regenerated aggregate file.
- Keep the raw CSVs outside git (handled via `.gitignore`) to avoid massive
  commits and to respect dataset licensing.

