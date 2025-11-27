"""Build aggregated Transfermarkt features for Premier League clubs."""

from __future__ import annotations

import argparse
from pathlib import Path

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
def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace({0: pd.NA})
    return numerator / denom


def _season_to_year(value: str | int | float) -> int | None:
    if pd.isna(value):
        return None
    try:
        n = int(str(value).split("/")[0])
    except ValueError:
        return None
    if n >= 1900:
        return n
    if n >= 50:
        return 1900 + n
    return 2000 + n


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
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
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
    pl_club_games["goal_difference"] = (
        pl_club_games["own_goals"] - pl_club_games["opponent_goals"]
    )
    pl_club_games["points"] = pd.NA
    win_mask = pl_club_games["own_goals"] > pl_club_games["opponent_goals"]
    draw_mask = pl_club_games["own_goals"] == pl_club_games["opponent_goals"]
    pl_club_games.loc[win_mask, "points"] = 3
    pl_club_games.loc[draw_mask, "points"] = 1
    pl_club_games["points"] = pd.to_numeric(pl_club_games["points"], errors="coerce")
    pl_club_games = pl_club_games.sort_values(["club_id", "date", "game_id"])
    group = pl_club_games.groupby("club_id", group_keys=False)
    pl_club_games["prev_points"] = group["points"].shift(1)
    pl_club_games["prev_goal_difference"] = group["goal_difference"].shift(1)
    pl_club_games["prev_goals_scored"] = group["own_goals"].shift(1)
    pl_club_games["prev_goals_conceded"] = group["opponent_goals"].shift(1)

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

    latest_valuations = (
        resources["player_valuations"]
        .sort_values("date")
        .groupby("player_id")
        .last()
        .reset_index()
        .rename(columns={"market_value_in_eur": "market_value_latest"})
    )

    players = resources["players"].copy()
    players["date_of_birth"] = pd.to_datetime(
        players.get("date_of_birth"), errors="coerce"
    )
    player_columns = ["player_id"]
    for col in ["height_in_cm", "date_of_birth", "current_club_id", "market_value_in_eur"]:
        if col in players.columns and col not in player_columns:
            player_columns.append(col)
    position_source = next(
        (
            col
            for col in ["player_position", "position", "pos", "position_group"]
            if col in players.columns
        ),
        None,
    )
    if position_source:
        player_columns.append(position_source)
    players_subset = players[player_columns].copy()
    if position_source:
        players_subset = players_subset.rename(
            columns={position_source: "player_position"}
        )
    else:
        players_subset["player_position"] = pd.NA

    lineups_raw = resources["game_lineups"].copy()
    if "date" in lineups_raw.columns:
        lineups_raw = lineups_raw.rename(columns={"date": "lineup_date"})
    else:
        lineups_raw["lineup_date"] = pd.NaT
    lineups = (
        lineups_raw.merge(players_subset, on="player_id", how="left")
        .merge(
            latest_valuations[["player_id", "market_value_latest"]],
            on="player_id",
            how="left",
        )
        .merge(
            pl_games[["game_id", "date", "season"]].rename(columns={"date": "match_date"}),
            on="game_id",
            how="left",
        )
    )
    lineups["match_date"] = pd.to_datetime(lineups["match_date"], errors="coerce")
    lineups["date_of_birth"] = pd.to_datetime(lineups["date_of_birth"], errors="coerce")
    lineups["age"] = (
        (lineups["match_date"] - lineups["date_of_birth"]).dt.days / 365.25
    )
    lineups["team_captain"] = pd.to_numeric(
        lineups.get("team_captain"), errors="coerce"
    ).fillna(0)
    lineups["height_in_cm"] = pd.to_numeric(lineups["height_in_cm"], errors="coerce")
    lineups["player_market_value"] = lineups["market_value_latest"]
    lineups["is_starter"] = lineups["type"].str.lower() == "starting_lineup"
    lineups["starter_market_value"] = lineups["player_market_value"].where(
        lineups["is_starter"]
    )
    if "position" in lineups_raw.columns:
        lineups["resolved_position"] = lineups_raw["position"].fillna(
            lineups["player_position"]
        )
    else:
        lineups["resolved_position"] = lineups["player_position"]
    lineups_filtered = lineups[lineups["game_id"].isin(pl_game_ids)].copy()

    transfers = resources["transfers"].copy()
    transfers["transfer_date"] = pd.to_datetime(
        transfers.get("transfer_date"), errors="coerce"
    )
    transfers["transfer_season_start"] = transfers["transfer_season"].apply(
        _season_to_year
    )
    recent_transfers = (
        transfers[
            transfers["transfer_season_start"].between(
                start_season, end_season, inclusive="both"
            )
        ]
        .dropna(subset=["to_club_id"])
        .sort_values("transfer_date")
        .groupby(["player_id", "to_club_id"], as_index=False)
        .first()
    )
    lineups_filtered = lineups_filtered.merge(
        recent_transfers[
            [
                "player_id",
                "to_club_id",
                "transfer_date",
                "transfer_season_start",
            ]
        ],
        on="player_id",
        how="left",
    )
    lineups_filtered["is_new_signing"] = (
        (lineups_filtered["club_id"] == lineups_filtered["to_club_id"])
        & (lineups_filtered["season"] == lineups_filtered["transfer_season_start"])
        & lineups_filtered["transfer_date"].notna()
        & lineups_filtered["match_date"].notna()
        & (lineups_filtered["match_date"] >= lineups_filtered["transfer_date"])
    )

    starter_sets = (
        lineups_filtered[lineups_filtered["is_starter"]]
        .groupby(["game_id", "club_id"])["player_id"]
        .agg(lambda ids: frozenset(ids))
        .reset_index()
        .rename(columns={"player_id": "starter_set"})
        .merge(pl_games[["game_id", "date"]], on="game_id", how="left")
        .sort_values(["club_id", "date", "game_id"])
    )
    starter_sets["prev_set"] = starter_sets.groupby("club_id")["starter_set"].shift(1)
    starter_sets["continuity_index"] = starter_sets.apply(
        lambda row: (
            len(row["starter_set"].intersection(row["prev_set"])) / 11
            if isinstance(row["starter_set"], frozenset)
            and isinstance(row["prev_set"], frozenset)
            and len(row["prev_set"]) > 0
            else pd.NA
        ),
        axis=1,
    )
    continuity_features = starter_sets[["game_id", "club_id", "continuity_index"]]

    club_top_players = latest_valuations.merge(
        players[["player_id", "current_club_id"]],
        on="player_id",
        how="left",
    )
    if "current_club_id" in club_top_players.columns:
        club_top_players = (
            club_top_players.dropna(subset=["current_club_id"])
            .sort_values(
                ["current_club_id", "market_value_latest"], ascending=[True, False]
            )
            .groupby("current_club_id")
            .head(5)
            .groupby("current_club_id")["player_id"]
            .agg(lambda ids: frozenset(ids))
            .reset_index()
            .rename(
                columns={"current_club_id": "club_id", "player_id": "top_players"}
            )
        )
    else:
        club_top_players = pd.DataFrame(
            {"club_id": [], "top_players": []}
        )
    missing_key = starter_sets.merge(club_top_players, on="club_id", how="left")
    missing_key["missing_key_players"] = missing_key.apply(
        lambda row: (
            len(row["top_players"] - row["starter_set"])
            if isinstance(row["top_players"], frozenset)
            and isinstance(row["starter_set"], frozenset)
            else pd.NA
        ),
        axis=1,
    )
    missing_key = missing_key[["game_id", "club_id", "missing_key_players"]]

    new_signings = (
        lineups_filtered.groupby(["game_id", "club_id"])["is_new_signing"]
        .sum()
        .reset_index()
        .rename(columns={"is_new_signing": "new_signings_played"})
    )

    lineup_features = (
        lineups_filtered.groupby(["game_id", "club_id"])
        .agg(
            n_players=("player_id", "count"),
            n_starters=("is_starter", "sum"),
            n_captains=("team_captain", "sum"),
            avg_height=("height_in_cm", "mean"),
            min_height=("height_in_cm", "min"),
            max_height=("height_in_cm", "max"),
            height_spread=("height_in_cm", lambda s: s.std(ddof=0)),
            defenders=("resolved_position", lambda s: s.isin(DEF_POSITIONS).sum()),
            midfielders=("resolved_position", lambda s: s.isin(MID_POSITIONS).sum()),
            forwards=("resolved_position", lambda s: s.isin(FWD_POSITIONS).sum()),
            avg_age=("age", "mean"),
            median_age=("age", "median"),
            squad_value_mean=("player_market_value", "mean"),
            squad_value_max=("player_market_value", "max"),
            squad_value_min=("player_market_value", "min"),
            squad_value_total=("player_market_value", "sum"),
            starter_market_value_sum=("starter_market_value", "sum"),
        )
        .reset_index()
    )
    lineup_features["others"] = (
        lineup_features["n_players"]
        - lineup_features[["defenders", "midfielders", "forwards"]].sum(axis=1)
    )
    lineup_features["height_spread"] = lineup_features["height_spread"].fillna(0)
    lineup_features["starters_percentage"] = _safe_ratio(
        lineup_features["n_starters"], lineup_features["n_players"]
    )
    lineup_features["avg_market_value_starting_xi"] = _safe_ratio(
        lineup_features["starter_market_value_sum"], lineup_features["n_starters"]
    )
    lineup_features = (
        lineup_features.drop(columns=["starter_market_value_sum"])
        .merge(continuity_features, on=["game_id", "club_id"], how="left")
        .merge(missing_key, on=["game_id", "club_id"], how="left")
        .merge(new_signings, on=["game_id", "club_id"], how="left")
    )
    lineup_features["continuity_index"] = lineup_features["continuity_index"].fillna(0)
    lineup_features["new_signings_played"] = lineup_features[
        "new_signings_played"
    ].fillna(0)

    events = resources["game_events"]
    event_features = (
        events[events["game_id"].isin(pl_game_ids)]
        .assign(type_clean=lambda df: df["type"].str.lower())
        .groupby(["game_id", "club_id"])
        .agg(
            n_events=("game_event_id", "count"),
            fouls=("type_clean", lambda s: s.eq("fouls").sum() + s.eq("foul").sum()),
            shots=("type_clean", lambda s: s.eq("shots").sum() + s.eq("shot").sum()),
            subs=(
                "type_clean",
                lambda s: s.eq("substitutions").sum() + s.eq("substitution").sum(),
            ),
            goals_event=(
                "type_clean",
                lambda s: s.eq("goals").sum() + s.eq("goal").sum(),
            ),
            passes=("type_clean", lambda s: s.eq("passes").sum()),
            touches=("type_clean", lambda s: s.eq("touches").sum()),
        )
        .reset_index()
    )
    event_features["n_subs_used"] = event_features["subs"]
    event_features["possession_proxy_events"] = (
        (event_features["shots"] + event_features["passes"] + event_features["touches"])
        / event_features["n_events"].replace(0, pd.NA)
    )

    pl_features = (
        pl_club_games.merge(
            appearance_features, on=["game_id", "club_id"], how="left"
        )
        .merge(lineup_features, on=["game_id", "club_id"], how="left")
        .merge(event_features, on=["game_id", "club_id"], how="left")
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
