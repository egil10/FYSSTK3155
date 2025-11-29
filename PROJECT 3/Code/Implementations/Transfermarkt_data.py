"""Build aggregated Transfermarkt features for all games (one row per game)."""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from functools import partial
from typing import Dict, Any

import pandas as pd
import numpy as np

try:
    from .feature_engineering import (
        PREDICTIVE_METADATA_PATH,
        prepare_features,
    )
except ImportError:
    from feature_engineering import (
        PREDICTIVE_METADATA_PATH,
        prepare_features,
    )

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
    """Load CSV with optimized settings for large files."""
    file_path = data_dir / f"{name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing expected file: {file_path}")
    # Use low_memory=False for better type inference on large files
    # Use engine='c' for faster parsing
    return pd.read_csv(file_path, low_memory=False, engine='c')


def build_game_datasets(
    data_dir: Path,
    start_season: int = None,
    end_season: int = None,
    competition_id: str = None,
) -> pd.DataFrame:
    """Build aggregated features for all games using pandas.
    
    Returns one row per game with features for both home and away teams.
    """
    print("=" * 80)
    print("BUILDING GAME-LEVEL TRANSFERMARKT DATASET")
    print("=" * 80)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading data files...")
    
    resources = {}
    file_names = [
        "appearances", "clubs", "club_games", "competitions", "games",
        "game_events", "game_lineups", "players", "player_valuations", "transfers"
    ]
    
    for name in file_names:
        print(f"  Loading {name}.csv...", end=" ", flush=True)
        resources[name] = load_csv(data_dir, name)
        print(f"[OK] ({len(resources[name]):,} rows)")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing games...")

    # Start with all games, optionally filter by competition
    games = resources["games"].copy()
    
    # Apply competition filter if provided
    if competition_id is not None:
        games = games[games["competition_id"] == competition_id].copy()
        print(f"  Filtered to competition {competition_id}: {len(games):,} games")
    else:
        print(f"  Processing all games: {len(games):,} games...")
    
    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games["round_number"] = (
        games["round"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )
    
    # CRITICAL: Apply season cutoff (2014+) for data quality
    # Data before 2014 has catastrophic NAs in lineup/valuation features
    SEASON_CUTOFF = 2014
    games_before = len(games)
    games = games[games["season"] >= SEASON_CUTOFF].copy()
    games_filtered = len(games)
    print(f"  Applied season cutoff (>= {SEASON_CUTOFF}): {games_before:,} → {games_filtered:,} games")
    
    # Get all game IDs as set for faster membership testing
    all_game_ids_set = set(games["game_id"].unique())
    all_game_ids = games["game_id"].unique()
    print(f"  Unique game IDs: {len(all_game_ids):,}")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Building club-level features...")

    # We'll augment games.csv directly (1 row per game)
    # Create club_games view for calculating club-level features
    club_games = (
        games.merge(resources["club_games"], on="game_id", how="left")
        .assign(is_home=lambda df: df["club_id"] == df["home_club_id"])
        .reset_index(drop=True)
    )
    # Optimize points calculation using np.select
    club_games["goal_difference"] = (
        club_games["own_goals"] - club_games["opponent_goals"]
    )
    club_games["points"] = np.select(
        [club_games["goal_difference"] > 0, club_games["goal_difference"] == 0],
        [3, 1],
        default=0
    ).astype("float64")
    club_games = club_games.sort_values(["club_id", "date", "game_id"])
    group = club_games.groupby("club_id", group_keys=False)
    club_games["prev_points"] = group["points"].shift(1)
    club_games["prev_goal_difference"] = group["goal_difference"].shift(1)
    club_games["prev_goals_scored"] = group["own_goals"].shift(1)
    club_games["prev_goals_conceded"] = group["opponent_goals"].shift(1)
    
    # Extract club-level features for merging back to games
    # Include date for lag calculations
    # Note: We keep own_goals/opponent_goals for current match (not prev_*)
    # The lagging happens inside compute_comprehensive_lagged_features via shift(1)
    club_history = club_games[["game_id", "club_id", "is_home", "date", 
                                "points", "goal_difference", "own_goals", "opponent_goals"]].copy()

    print(f"  Computing appearance features...", end=" ", flush=True)
    appearances = resources["appearances"]
    # Filter first for efficiency
    appearances_filtered = appearances[appearances["game_id"].isin(all_game_ids_set)]
    appearance_features = (
        appearances_filtered
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
    print(f"[OK] ({len(appearance_features):,} club-game combinations)")

    print(f"  Processing player valuations...", end=" ", flush=True)
    player_valuations = resources["player_valuations"].copy()
    player_valuations["date"] = pd.to_datetime(
        player_valuations.get("date"), errors="coerce"
    )
    player_valuations["player_id"] = pd.to_numeric(
        player_valuations["player_id"], errors="coerce"
    )
    player_valuations = player_valuations.dropna(subset=["player_id"])
    latest_valuations = (
        player_valuations.sort_values("date")
        .groupby("player_id")
        .last()
        .reset_index()
        .rename(columns={"market_value_in_eur": "market_value_latest"})
    )
    print(f"[OK] ({len(latest_valuations):,} players)")

    print(f"  Processing player data...", end=" ", flush=True)
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
    print(f"[OK] ({len(players_subset):,} players)")

    print(f"  Processing lineups...", end=" ", flush=True)
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
            games[["game_id", "date", "season"]].rename(columns={"date": "match_date"}),
            on="game_id",
            how="left",
        )
    )
    lineups["match_date"] = pd.to_datetime(lineups["match_date"], errors="coerce")
    lineups["player_id"] = pd.to_numeric(lineups["player_id"], errors="coerce")
    # Note: valuations already merged above (latest_valuations), will be lagged by lag feature engineering
    lineups = lineups.sort_values(["player_id", "match_date"])
    lineups["date_of_birth"] = pd.to_datetime(lineups["date_of_birth"], errors="coerce")
    lineups["age"] = (
        (lineups["match_date"] - lineups["date_of_birth"]).dt.days / 365.25
    )
    # Use latest valuation directly - lag feature engineering will create lagged versions
    lineups["player_market_value"] = lineups["market_value_latest"]
    lineups["team_captain"] = pd.to_numeric(
        lineups.get("team_captain"), errors="coerce"
    ).fillna(0)
    lineups["height_in_cm"] = pd.to_numeric(lineups["height_in_cm"], errors="coerce")
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
    lineups_filtered = lineups[lineups["game_id"].isin(all_game_ids_set)].copy()
    print(f"[OK] ({len(lineups_filtered):,} lineup entries)")

    print(f"  Processing transfers...", end=" ", flush=True)
    transfers = resources["transfers"].copy()
    transfers["transfer_date"] = pd.to_datetime(
        transfers.get("transfer_date"), errors="coerce"
    )
    transfers["transfer_season_start"] = transfers["transfer_season"].apply(
        _season_to_year
    )
    # Use all transfers - no season filtering
    recent_transfers = (
        transfers
        .dropna(subset=["to_club_id"])
        .sort_values("transfer_date")
        .groupby(["player_id", "to_club_id"], as_index=False)
        .first()
    )
    print(f"[OK] ({len(recent_transfers):,} transfers)")
    
    print(f"  Merging transfer data with lineups...", end=" ", flush=True)
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
    print(f"[OK]")

    print(f"  Computing lineup continuity features...", end=" ", flush=True)
    starter_sets = (
        lineups_filtered[lineups_filtered["is_starter"]]
        .groupby(["game_id", "club_id"])["player_id"]
        .agg(lambda ids: frozenset(ids))
        .reset_index()
        .rename(columns={"player_id": "starter_set"})
        .merge(games[["game_id", "date"]], on="game_id", how="left")
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
    print(f"[OK] ({len(continuity_features):,} entries)")

    # REMOVED: missing_key_players calculation
    # This feature is leaky (uses current_club_id and latest valuations from future)
    # and was 100% NA in the dataset. Removing entirely.

    print(f"  Computing new signings...", end=" ", flush=True)
    new_signings = (
        lineups_filtered.groupby(["game_id", "club_id"])["is_new_signing"]
        .sum()
        .reset_index()
        .rename(columns={"is_new_signing": "new_signings_played"})
    )
    print(f"[OK]")

    print(f"  Computing lineup features...", end=" ", flush=True)
    # Note: n_players kept because it's needed for starters_percentage and is a useful structural feature
    lineup_features = (
        lineups_filtered.groupby(["game_id", "club_id"])
        .agg(
            n_players=("player_id", "count"),
            n_starters=("is_starter", "sum"),
            avg_height=("height_in_cm", "mean"),
            min_height=("height_in_cm", "min"),
            max_height=("height_in_cm", "max"),
            height_spread=("height_in_cm", lambda s: s.std(ddof=0)),
            defenders=("resolved_position", lambda s: s.isin(DEF_POSITIONS).sum()),
            midfielders=("resolved_position", lambda s: s.isin(MID_POSITIONS).sum()),
            forwards=("resolved_position", lambda s: s.isin(FWD_POSITIONS).sum()),
            avg_age=("age", "mean"),
            median_age=("age", "median"),
            squad_value_total=("player_market_value", "sum"),
            starter_market_value_sum=("starter_market_value", "sum"),
        )
        .reset_index()
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
        .merge(new_signings, on=["game_id", "club_id"], how="left")
    )
    # Fix FutureWarning about downcasting - handle pd.NA values properly
    if "continuity_index" in lineup_features.columns:
        # Use pd.to_numeric with errors='coerce' to handle pd.NA, then fillna
        continuity_series = lineup_features["continuity_index"].copy()
        # Convert to numeric (this converts pd.NA to NaN)
        continuity_series = pd.to_numeric(continuity_series, errors="coerce")
        # Fill NaN with 0.0 and ensure float64 type
        lineup_features["continuity_index"] = continuity_series.fillna(0.0).astype("float64")
    lineup_features["new_signings_played"] = lineup_features[
        "new_signings_played"
    ].fillna(0)
    print(f"[OK] ({len(lineup_features):,} club-game combinations)")

    print(f"  Computing event features...", end=" ", flush=True)
    events = resources["game_events"]
    # Filter and clean type in one pass
    events_filtered = events[events["game_id"].isin(all_game_ids_set)].copy()
    events_filtered["type_clean"] = events_filtered["type"].str.lower()
    
    # REMOVED: shots, fouls, passes, touches, possession_proxy_events (all constant at 0)
    # KEEP: goals_event (has actual values)
    event_features = (
        events_filtered
        .groupby(["game_id", "club_id"])
        .agg(
            goals_event=(
                "type_clean",
                lambda s: s.eq("goals").sum() + s.eq("goal").sum(),
            ),
        )
        .reset_index()
    )
    print(f"[OK] ({len(event_features):,} club-game combinations)")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Combining features and computing lagged windows...")
    # Combine all club-level features (these are per-match, not lagged yet)
    # club_history already includes date and current match stats
    club_features = (
        club_history
        .merge(appearance_features, on=["game_id", "club_id"], how="left")
        .merge(lineup_features, on=["game_id", "club_id"], how="left")
        .merge(event_features, on=["game_id", "club_id"], how="left")
    )
    
    # Ensure we have goals_scored and goals_conceded from own_goals/opponent_goals
    # These are CURRENT match values (not lagged) - lagging happens inside compute_comprehensive_lagged_features
    if "own_goals" in club_features.columns and "goals_scored" not in club_features.columns:
        club_features["goals_scored"] = club_features["own_goals"]
    if "opponent_goals" in club_features.columns and "goals_conceded" not in club_features.columns:
        club_features["goals_conceded"] = club_features["opponent_goals"]
    
    # Now compute comprehensive lagged features using the optimized window system
    try:
        from .lag_feature_engineering import compute_comprehensive_lagged_features
    except ImportError:
        from lag_feature_engineering import compute_comprehensive_lagged_features
    
    # Extended lag windows for comprehensive feature engineering:
    # - Fast features (goals, points, events): L1, L2, L3, L5, L10, L20
    # - Slow features (height, age, squad_value, etc.): L5, L10, L20
    lag_windows_fast = [1, 2, 3, 5, 10, 20]
    lag_windows_slow = [5, 10, 20]
    games_features = compute_comprehensive_lagged_features(
        club_features=club_features,
        games_df=games,
        lag_windows_fast=lag_windows_fast,
        lag_windows_slow=lag_windows_slow
    )
    
    print(f"  Finalizing dataset...", end=" ", flush=True)
    games_features = (
        games_features
        .sort_values(["season", "game_id"])
        .reset_index(drop=True)
    )
    print(f"[OK]")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Complete! Final dataset: {len(games_features):,} games")
    print(f"  Columns: {len(games_features.columns)}")
    return games_features


def reorder_columns_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns: metadata (ID_ prefix) → RESULT → predictor features.
    
    Returns DataFrame with columns in this order:
    1. Metadata columns (renamed with ID_ prefix)
    2. RESULT column (W/D/L)
    3. All predictor features (home_* and away_* features)
    """
    df = df.copy()
    
    # Define metadata column mappings (old_name -> new_name)
    metadata_mappings = {
        "game_id": "ID_GAME",
        "competition_id": "ID_COMPETITION",
        "season": "ID_SEASON",
        "round": "ID_ROUND",
        "date": "ID_DATE",
        "home_club_id": "ID_HOME_CLUB",
        "away_club_id": "ID_AWAY_CLUB",
        "home_club_name": "ID_HOME_TEAM",
        "away_club_name": "ID_AWAY_TEAM",
        "home_club_manager_name": "ID_HOME_MANAGER",
        "away_club_manager_name": "ID_AWAY_MANAGER",
        "stadium": "ID_STADIUM",
        "attendance": "ID_ATTENDANCE",
        "referee": "ID_REFIREE",
        "url": "ID_URL",
        "round_number": "ID_ROUND_NUMBER",
        "home_club_goals": "ID_HOME_GOALS",
        "away_club_goals": "ID_AWAY_GOALS",
        "aggregate": "ID_AGGREGATE",
        "competition_type": "ID_COMPETITION_TYPE",
        "home_club_formation": "ID_HOME_FORMATION",
        "away_club_formation": "ID_AWAY_FORMATION",
    }
    
    # Define metadata column order (in the exact order specified by user)
    metadata_order = [
        "ID_GAME", "ID_COMPETITION", "ID_SEASON", "ID_ROUND", "ID_DATE",
        "ID_HOME_CLUB", "ID_AWAY_CLUB", "ID_HOME_TEAM", "ID_AWAY_TEAM",
        "ID_HOME_MANAGER", "ID_AWAY_MANAGER", "ID_STADIUM", "ID_ATTENDANCE",
        "ID_REFIREE", "ID_URL", "ID_ROUND_NUMBER", "ID_HOME_GOALS", "ID_AWAY_GOALS",
        "ID_AGGREGATE", "ID_COMPETITION_TYPE", "ID_HOME_FORMATION", "ID_AWAY_FORMATION",
    ]
    
    # Step 1: Rename metadata columns
    rename_dict = {}
    for old_name, new_name in metadata_mappings.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name
    df = df.rename(columns=rename_dict)
    
    # Step 2: Remove target_result if it exists (we're creating RESULT instead)
    if "target_result" in df.columns:
        df = df.drop(columns=["target_result"])
    
    # Step 3: Create RESULT column from ID_HOME_GOALS and ID_AWAY_GOALS
    # After renaming, these should be ID_HOME_GOALS and ID_AWAY_GOALS
    if "ID_HOME_GOALS" in df.columns and "ID_AWAY_GOALS" in df.columns:
        # Create RESULT column as string type
        df["RESULT"] = "L"  # Default to loss
        win_mask = df["ID_HOME_GOALS"] > df["ID_AWAY_GOALS"]
        draw_mask = df["ID_HOME_GOALS"] == df["ID_AWAY_GOALS"]
        df.loc[win_mask, "RESULT"] = "W"
        df.loc[draw_mask, "RESULT"] = "D"
        # Ensure it's string type
        df["RESULT"] = df["RESULT"].astype("string")
    elif "home_club_goals" in df.columns and "away_club_goals" in df.columns:
        # Fallback if renaming didn't happen for some reason
        df["RESULT"] = "L"
        win_mask = df["home_club_goals"] > df["away_club_goals"]
        draw_mask = df["home_club_goals"] == df["away_club_goals"]
        df.loc[win_mask, "RESULT"] = "W"
        df.loc[draw_mask, "RESULT"] = "D"
        df["RESULT"] = df["RESULT"].astype("string")
    
    # Step 4: Identify all columns after renaming
    metadata_cols_present = [col for col in metadata_order if col in df.columns]
    result_col = ["RESULT"] if "RESULT" in df.columns else []
    
    # All other columns are predictor features - preserve their current relative order
    all_cols = list(df.columns)
    predictor_cols = [
        col for col in all_cols 
        if col not in metadata_cols_present and col != "RESULT"
    ]
    
    # Step 5: Reorder columns: metadata → RESULT → predictors (preserve predictor order)
    final_column_order = metadata_cols_present + result_col + predictor_cols
    
    # Ensure we have all columns (should be the same, but double-check)
    missing_cols = [col for col in df.columns if col not in final_column_order]
    if missing_cols:
        final_column_order = final_column_order + missing_cols
    
    return df[final_column_order]


def save_features(df: pd.DataFrame, output_path: Path) -> Path:
    """Save features to file, using appropriate format and compression."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        # Use snappy compression for good balance of speed and compression
        df.to_parquet(output_path, index=False, compression='snappy')
    else:
        df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Transfermarkt game aggregates with features for all games."
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
        default=None,
        help="First season to include (optional, defaults to all seasons).",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help="Last season to include (optional, defaults to all seasons).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the aggregated dataset (default: Code/Data/game_features.parquet).",
    )
    parser.add_argument(
        "--also-save-csv",
        action="store_true",
        help="Also save CSV version alongside Parquet (for compatibility).",
    )
    parser.add_argument(
        "--competition-id",
        type=str,
        default=None,
        help="Transfermarkt competition identifier (optional, defaults to all competitions).",
    )
    return parser.parse_args()


def main() -> None:
    start_time = datetime.now()
    args = parse_args()
    data_dir = (
        args.data_dir
        if args.data_dir is not None
        else Path(__file__).resolve().parents[1] / "Data"
    )
    output_path = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parents[1] / "Data" / "game_features.parquet"
    )
    
    print(f"\nData directory: {data_dir}")
    print(f"Output path: {output_path}\n")
    
    raw_features = build_game_datasets(
        data_dir=data_dir,
        start_season=args.start_season,
        end_season=args.end_season,
        competition_id=args.competition_id,
    )
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running feature engineering...")
    processed_features, predictive_cols = prepare_features(raw_features)
    print(f"  [OK] Feature engineering complete")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Reordering columns for modeling...")
    processed_features = reorder_columns_for_modeling(processed_features)
    print(f"  [OK] Column reordering complete")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving results...")
    saved_path = save_features(processed_features, output_path)
    
    # Get file size
    file_size_mb = saved_path.stat().st_size / (1024 * 1024)
    
    # Optionally save CSV version
    if args.also_save_csv or output_path.suffix.lower() == ".csv":
        csv_path = output_path.with_suffix(".csv")
        print(f"  Also saving CSV version...", end=" ", flush=True)
        processed_features.to_csv(csv_path, index=False)
        csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"[OK] ({csv_size_mb:.1f} MB)")
    else:
        csv_size_mb = None
    
    PREDICTIVE_METADATA_PATH.write_text("\n".join(predictive_cols))
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Built {len(processed_features):,} game rows")
    print(f"Total columns: {len(processed_features.columns)}")
    print(f"Predictive feature count: {len(predictive_cols)}")
    print(f"Saved dataset to: {saved_path}")
    print(f"  File size: {file_size_mb:.1f} MB ({saved_path.suffix.upper()})")
    if csv_size_mb:
        compression_ratio = (1 - file_size_mb / csv_size_mb) * 100
        print(f"  CSV size: {csv_size_mb:.1f} MB")
        print(f"  Compression: {compression_ratio:.1f}% smaller")
    print(f"Total processing time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()