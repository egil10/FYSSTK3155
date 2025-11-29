"""Comprehensive lag-based feature engineering with strict temporal filtering.

This module implements a complete lag-based feature engineering pipeline that:
- Computes multi-window lagged features (L1, L3, L5, L10, L20)
- Applies strict temporal filtering (no data leakage)
- Follows naming convention: {team}_{feature}_{stat}_L{N}
- Adds interaction features (diff_*)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def safe_rolling_agg(
    series: pd.Series,
    window: int,
    agg_func: str,
    min_periods: int = 1
) -> pd.Series:
    """Safely compute rolling aggregation, handling NA values and type conversion.
    
    Args:
        series: Series to aggregate
        window: Rolling window size
        agg_func: Aggregation function ('mean', 'sum', 'max', 'min', 'std')
        min_periods: Minimum periods for rolling window
    
    Returns:
        Series with rolling aggregation results
    """
    # Convert to numeric, replacing pd.NA with NaN
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Compute rolling aggregation
    rolled = numeric_series.rolling(window=window, min_periods=min_periods)
    
    if agg_func == "mean":
        result = rolled.mean()
    elif agg_func == "sum":
        result = rolled.sum()
    elif agg_func == "max":
        result = rolled.max()
    elif agg_func == "min":
        result = rolled.min()
    elif agg_func == "std":
        result = rolled.std()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")
    
    return result


def compute_comprehensive_lagged_features(
    club_features: pd.DataFrame,
    games_df: pd.DataFrame,
    lag_windows: List[int] = [1, 3, 5, 10, 20],
) -> pd.DataFrame:
    """Compute comprehensive lagged features with strict temporal filtering.
    
    This function replaces the current simple feature merging with a comprehensive
    lag-based approach that computes rolling statistics across multiple windows.
    
    Args:
        club_features: DataFrame with one row per club-game, must include:
            - game_id, club_id, is_home, date
            - All computed features (points, goals, appearances, lineups, events, etc.)
        games_df: DataFrame with game metadata
        lag_windows: List of lag window sizes [1, 3, 5, 10, 20]
    
    Returns:
        DataFrame with one row per game, with lagged features following naming convention
    """
    from datetime import datetime
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Computing comprehensive lagged features...")
    print(f"  Windows: {lag_windows}")
    
    # Ensure proper sorting for lag calculations
    club_features = club_features.copy()
    club_features = club_features.sort_values(["club_id", "date", "game_id"]).reset_index(drop=True)
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(club_features["date"]):
        club_features["date"] = pd.to_datetime(club_features["date"], errors="coerce")
    
    # Convert all feature columns to numeric to handle pd.NA values before rolling
    # Skip identifier columns
    id_cols = {"game_id", "club_id", "is_home", "date"}
    feature_cols = [col for col in club_features.columns if col not in id_cols]
    
    for col in feature_cols:
        if col in club_features.columns:
            # Convert to numeric, replacing pd.NA/NaN with NaN
            club_features[col] = pd.to_numeric(club_features[col], errors='coerce')
    
    # Group by club for rolling calculations
    grouped = club_features.groupby("club_id", group_keys=False)
    
    # Prepare base features - ensure we have goals_scored and goals_conceded
    if "own_goals" in club_features.columns and "goals_scored" not in club_features.columns:
        club_features["goals_scored"] = club_features["own_goals"]
    if "opponent_goals" in club_features.columns and "goals_conceded" not in club_features.columns:
        club_features["goals_conceded"] = club_features["opponent_goals"]
    
    # Compute all lagged features
    lagged_cols = {}
    
    # 4A. Club Previous-Match Performance
    performance_features = {
        "points": "sum",
        "goal_difference": "mean", 
        "goals_scored": "mean",
        "goals_conceded": "mean",
    }
    
    # 4B. Appearance-Based Performance  
    appearance_base_cols = {
        "goals": ["mean", "max", "min"],
        "assists": ["mean"],
        "minutes_played": ["mean"],
        "yellow_cards": ["sum"],
        "red_cards": ["sum"],
    }
    
    # Map actual column names
    appearance_col_map = {
        "goals": ["goals_mean", "goals"],
        "assists": ["assists_mean", "assists"],
        "minutes_played": ["minutes_mean", "minutes_played"],
        "yellow_cards": ["yellow_cards_sum", "yellow_cards"],
        "red_cards": ["red_cards_sum", "red_cards"],
    }
    
    # 4C. Lineup/Squad Composition
    lineup_features = {
        "n_players": "mean",
        "n_starters": "mean", 
        "n_captains": "mean",
        "avg_height": "mean",
        "min_height": "mean",
        "max_height": "mean",
        "height_spread": "mean",
        "defenders": "mean",
        "midfielders": "mean",
        "forwards": "mean",
        "others": "mean",
        "avg_age": "mean",
        "median_age": "mean",
        "squad_value_total": "mean",
        "avg_market_value_starting_xi": "mean",
        "continuity_index": "mean",
        "missing_key_players": "mean",
        "new_signings_played": "mean",
        "starters_percentage": "mean",
    }
    
    # 4D. Game Event Features
    event_features = {
        "shots": "mean",
        "fouls": "mean",
        "goals_event": "sum",
        "passes": "mean",
        "touches": "mean",
        "possession_proxy_events": "mean",
    }
    
    # Compute lagged features for each window
    for window in lag_windows:
        print(f"    Computing L{window} features...", end=" ", flush=True)
        
        # Club performance features
        for feat_name, stat in performance_features.items():
            if feat_name in club_features.columns:
                shifted = grouped[feat_name].shift(1)
                if stat == "sum":
                    rolled = shifted.rolling(window=window, min_periods=1).sum()
                elif stat == "mean":
                    rolled = shifted.rolling(window=window, min_periods=1).mean()
                lagged_cols[f"{feat_name}_{stat}_L{window}"] = rolled.reset_index(level=0, drop=True)
        
        # Appearance features (handle different column names)
        for feat_name, stats in appearance_base_cols.items():
            # Find actual column name
            actual_cols = appearance_col_map.get(feat_name, [feat_name])
            actual_col = None
            for col in actual_cols:
                if col in club_features.columns:
                    actual_col = col
                    break
            
            if actual_col:
                shifted = grouped[actual_col].shift(1)
                for stat in stats:
                    if stat == "mean":
                        rolled = shifted.rolling(window=window, min_periods=1).mean()
                    elif stat == "max":
                        rolled = shifted.rolling(window=window, min_periods=1).max()
                    elif stat == "min":
                        rolled = shifted.rolling(window=window, min_periods=1).min()
                    elif stat == "sum":
                        rolled = shifted.rolling(window=window, min_periods=1).sum()
                    lagged_cols[f"{feat_name}_{stat}_L{window}"] = rolled.reset_index(level=0, drop=True)
        
        # Lineup features
        for feat_name, stat in lineup_features.items():
            if feat_name in club_features.columns:
                shifted = grouped[feat_name].shift(1)
                if stat == "mean":
                    rolled = shifted.rolling(window=window, min_periods=1).mean()
                lagged_cols[f"{feat_name}_{stat}_L{window}"] = rolled.reset_index(level=0, drop=True)
        
        # Event features
        for feat_name, stat in event_features.items():
            if feat_name in club_features.columns:
                shifted = grouped[feat_name].shift(1)
                if stat == "mean":
                    rolled = shifted.rolling(window=window, min_periods=1).mean()
                elif stat == "sum":
                    rolled = shifted.rolling(window=window, min_periods=1).sum()
                lagged_cols[f"{feat_name}_{stat}_L{window}"] = rolled.reset_index(level=0, drop=True)
        
        print(f"✓")
    
    # Create DataFrame with lagged features
    lagged_df = pd.DataFrame(lagged_cols)
    
    # Combine with identifiers
    result_df = pd.concat([
        club_features[["game_id", "club_id", "is_home", "date"]],
        lagged_df
    ], axis=1)
    
    # Split into home and away
    home_df = result_df[result_df["is_home"] == True].copy()
    away_df = result_df[result_df["is_home"] == False].copy()
    
    # Rename columns with home_/away_ prefix
    home_rename = {col: f"home_{col}" for col in lagged_df.columns}
    home_rename["game_id"] = "game_id"
    home_df = home_df.rename(columns=home_rename).drop(columns=["club_id", "is_home", "date"])
    
    away_rename = {col: f"away_{col}" for col in lagged_df.columns}
    away_rename["game_id"] = "game_id"
    away_df = away_df.rename(columns=away_rename).drop(columns=["club_id", "is_home", "date"])
    
    # Merge with games
    games_with_features = (
        games_df
        .merge(home_df, on="game_id", how="left")
        .merge(away_df, on="game_id", how="left")
    )
    
    # Add interaction features (diff_*)
    print(f"  Computing interaction features (diff_*)...", end=" ", flush=True)
    interaction_features = add_interaction_features(games_with_features, lag_windows)
    print(f"✓")
    
    return interaction_features


def add_interaction_features(df: pd.DataFrame, lag_windows: List[int]) -> pd.DataFrame:
    """Add derived interaction features (diff_*).
    
    Computes relative strength features: home - away for various metrics.
    """
    df = df.copy()
    
    # Features to compute differences for
    diff_features = [
        ("points", "sum"),
        ("goal_difference", "mean"),
        ("goals_scored", "mean"),
        ("squad_value_total", "mean"),
        ("avg_age", "mean"),
        ("avg_height", "mean"),
        ("shots", "mean"),
        ("possession_proxy_events", "mean"),
    ]
    
    for window in lag_windows:
        for feat_name, stat in diff_features:
            home_col = f"home_{feat_name}_{stat}_L{window}"
            away_col = f"away_{feat_name}_{stat}_L{window}"
            
            if home_col in df.columns and away_col in df.columns:
                diff_col = f"diff_{feat_name}_L{window}"
                df[diff_col] = df[home_col] - df[away_col]
    
    return df

