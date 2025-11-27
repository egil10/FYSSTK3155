"""Build aggregated Transfermarkt features for Premier League clubs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_COMPETITION_ID = "GB1"
DEF_POSITIONS = {"Centre-Back", "Left-Back", "Right-Back", "Defender", "Sweeper"}
MID_POSITIONS = {
    "Defensive Midfield",
    "Central Midfield",
    "Attacking Midfield",
    "Left Midfield",
    "Right Midfield",
    "midfield",
}
FWD_POSITIONS = {
    "Centre-Forward",
    "Left Winger",
    "Right Winger",
    "Second Striker",
    "Attack",
}


def load_csv(data_dir: Path, name: str) -> pd.DataFrame:
    file_path = data_dir / f"{name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing expected file: {file_path}")
    return pd.read_csv(file_path)


def build_pl_datasets(
    data_dir: Path,
    start_season: int,
    end_season: int,
    competition_id: str = DEFAULT_COMPETITION_ID,
) -> pd.DataFrame:
    """Replicate the R aggregation pipeline using pandas."""
    resources = {
        "appearances": load_csv(data_dir, "appearances"),
        "clubs": load_csv(data_dir, "clubs"),
        "club_games": load_csv(data_dir, "club_games"),
        "competitions": load_csv(data_dir, "competitions"),
        "games": load_csv(data_dir, "games"),
        "game_events": load_csv(data_dir, "game_events"),
        "game_lineups": load_csv(data_dir, "game_lineups"),
        "players": load_csv(data_dir, "players"),
        "player_valuations": load_csv(data_dir, "player_valuations"),
        "transfers": load_csv(data_dir, "transfers"),
    }

    games = resources["games"].copy()
    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    pl_games = games[
        (games["competition_id"] == competition_id)
        & (games["season"] >= start_season)
        & (games["season"] <= end_season)
    ].copy()
    if pl_games.empty:
        raise ValueError(
            "Premier League subset is empty. Check competition id or season limits."
        )
    pl_game_ids = pl_games["game_id"].unique()

    # Club-level rows per match with home/away marker
    pl_club_games = (
        pl_games.merge(resources["club_games"], on="game_id", how="left")
        .assign(is_home=lambda df: df["club_id"] == df["home_club_id"])
        .reset_index(drop=True)
    )

    appearances = resources["appearances"]
    appearance_features = (
        appearances[appearances["game_id"].isin(pl_game_ids)]
        .groupby(["game_id", "player_club_id"])
        .agg(
            goals_mean=("goals", "mean"),
            goals_max=("goals", "max"),
            goals_min=("goals", "min"),
            assists_mean=("assists", "mean"),
            assists_max=("assists", "max"),
            assists_min=("assists", "min"),
            minutes_mean=("minutes_played", "mean"),
            minutes_max=("minutes_played", "max"),
            minutes_min=("minutes_played", "min"),
            yellow_cards_sum=("yellow_cards", "sum"),
            red_cards_sum=("red_cards", "sum"),
        )
        .reset_index()
        .rename(columns={"player_club_id": "club_id"})
    )

    players = resources["players"].copy()
    player_columns = ["player_id"]
    if "height_in_cm" in players.columns:
        player_columns.append("height_in_cm")
    position_source = next(
        (col for col in ["player_position", "position", "pos", "position_group"] if col in players.columns),
        None,
    )
    if position_source:
        player_columns.append(position_source)
    players_subset = players[player_columns].copy()
    if position_source:
        players_subset = players_subset.rename(columns={position_source: "player_position"})
    else:
        players_subset["player_position"] = pd.NA

    lineups_raw = resources["game_lineups"].copy()
    lineups = lineups_raw.merge(players_subset, on="player_id", how="left")
    if "position" in lineups_raw.columns:
        lineups["resolved_position"] = lineups_raw["position"].fillna(lineups["player_position"])
    else:
        lineups["resolved_position"] = lineups["player_position"]
    lineup_features = (
        lineups[lineups["game_id"].isin(pl_game_ids)]
        .assign(
            team_captain=lambda df: pd.to_numeric(
                df["team_captain"], errors="coerce"
            ).fillna(0),
        )
        .groupby(["game_id", "club_id"])
        .agg(
            n_players=("player_id", "count"),
            n_captains=("team_captain", "sum"),
            avg_height=("height_in_cm", "mean"),
            min_height=("height_in_cm", "min"),
            max_height=("height_in_cm", "max"),
            defenders=("resolved_position", lambda s: s.isin(DEF_POSITIONS).sum()),
            midfielders=("resolved_position", lambda s: s.isin(MID_POSITIONS).sum()),
            forwards=("resolved_position", lambda s: s.isin(FWD_POSITIONS).sum()),
        )
        .reset_index()
    )
    lineup_features["others"] = (
        lineup_features["n_players"]
        - lineup_features[["defenders", "midfielders", "forwards"]].sum(axis=1)
    )

    events = resources["game_events"]
    event_features = (
        events[events["game_id"].isin(pl_game_ids)]
        .assign(type_clean=lambda df: df["type"].str.lower())
        .groupby(["game_id", "club_id"])
        .agg(
            n_events=("game_event_id", "count"),
            fouls=("type_clean", lambda s: (s == "foul").sum()),
            shots=("type_clean", lambda s: (s == "shot").sum()),
            subs=("type_clean", lambda s: (s == "substitution").sum()),
            goals_event=("type_clean", lambda s: (s == "goal").sum()),
        )
        .reset_index()
    )

    valuations = resources["player_valuations"].copy()
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuation_features = (
        valuations[valuations["date"] >= pd.Timestamp(f"{start_season}-01-01")]
        .groupby("player_id")
        .agg(
            market_value_mean=("market_value_in_eur", "mean"),
            market_value_max=("market_value_in_eur", "max"),
            market_value_min=("market_value_in_eur", "min"),
        )
        .reset_index()
        .merge(
            resources["players"][["player_id", "current_club_id"]],
            on="player_id",
            how="left",
        )
        .groupby("current_club_id")
        .agg(
            squad_value_mean=("market_value_mean", "mean"),
            squad_value_max=("market_value_max", "max"),
            squad_value_min=("market_value_min", "min"),
        )
        .reset_index()
        .rename(columns={"current_club_id": "club_id"})
    )

    pl_features = (
        pl_club_games.merge(
            appearance_features, on=["game_id", "club_id"], how="left"
        )
        .merge(lineup_features, on=["game_id", "club_id"], how="left")
        .merge(event_features, on=["game_id", "club_id"], how="left")
        .merge(valuation_features, on="club_id", how="left")
        .sort_values(["season", "game_id", "club_id"])
        .reset_index(drop=True)
    )
    return pl_features


def save_features(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Premier League Transfermarkt aggregates."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing raw CSVs (default: Code/Data relative to this file).",
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2021,
        help="First season to include (default: 2021).",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2025,
        help="Last season to include (default: 2025).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the aggregated dataset (default: Code/Data/pl_team_features.csv).",
    )
    parser.add_argument(
        "--competition-id",
        type=str,
        default=DEFAULT_COMPETITION_ID,
        help="Transfermarkt competition identifier (default: GB1 for Premier League).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = (
        args.data_dir
        if args.data_dir is not None
        else Path(__file__).resolve().parents[1] / "Data"
    )
    output_path = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parents[1] / "Data" / "pl_team_features.csv"
    )
    features = build_pl_datasets(
        data_dir=data_dir,
        start_season=args.start_season,
        end_season=args.end_season,
        competition_id=args.competition_id,
    )
    saved_path = save_features(features, output_path)
    print(f"Built {len(features)} team-game rows.")
    print(f"Saved dataset to {saved_path}")


if __name__ == "__main__":
    main()
