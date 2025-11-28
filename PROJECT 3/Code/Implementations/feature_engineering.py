"""Leak-free feature engineering for Premier League aggregates."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

RESULT_MAP = {"win": 3, "draw": 1, "loss": 0}

LEAKAGE_COLUMNS = [
    "home_club_goals",
    "away_club_goals",
    "own_goals",
    "opponent_goals",
    "goal_difference",
    "points",
    "is_win",
    "goals_mean",
    "goals_max",
    "goals_min",
    "assists_mean",
    "assists_max",
    "assists_min",
    "minutes_mean",
    "minutes_max",
    "minutes_min",
    "yellow_cards_sum",
    "red_cards_sum",
    "n_events",
    "fouls",
    "shots",
    "subs",
    "goals_event",
    "passes",
    "touches",
    "n_subs_used",
    "possession_proxy_events",
]

TRANSFORM_COLUMNS = [
    "points",
    "goal_difference",
    "own_goals",
    "opponent_goals",
    "squad_value_total",
    "avg_market_value_starting_xi",
    "avg_age",
    "avg_height",
    "defenders",
    "midfielders",
    "forwards",
    "continuity_index",
    "missing_key_players",
    "new_signings_played",
    "shots",
    "fouls",
    "passes",
    "touches",
]

ROLLING_COLS = {"points", "own_goals", "opponent_goals", "squad_value_total", "shots"}

PREDICTIVE_METADATA_PATH = (
    Path(__file__).resolve().parents[1] / "Data" / "predictive_features.txt"
)

def prepare_features(pl_features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = pl_features.sort_values(["club_id", "date", "game_id"]).reset_index(drop=True)
    grouped = df.groupby("club_id", group_keys=False)

    draw_mask = df["goal_difference"] == 0
    df["target_result"] = np.select(
        [df["is_win"] == 1, draw_mask],
        [RESULT_MAP["win"], RESULT_MAP["draw"]],
        default=RESULT_MAP["loss"],
    )

    for col in TRANSFORM_COLUMNS:
        if col not in df.columns:
            continue
        df[f"lag_1_{col}"] = grouped[col].shift(1)
        if col in ROLLING_COLS:
            rolling = (
                grouped[col]
                .rolling(window=5, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
                .shift(1)
            )
            df[f"roll_5_avg_{col}"] = rolling

    base_cols = [
        c
        for c in df.columns
        if c not in LEAKAGE_COLUMNS
        and c not in {f"lag_1_{col}" for col in TRANSFORM_COLUMNS}
        and not c.startswith("roll_5_avg_")
        and c != "target_result"
    ]
    lag_cols = [c for c in df.columns if c.startswith("lag_1_")]
    roll_cols = [c for c in df.columns if c.startswith("roll_5_avg_")]
    predictive_cols = base_cols + lag_cols + roll_cols

    return df, predictive_cols

