import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

def prepare_data(
    df: pd.DataFrame,
    *,
    target_log: str = "y_log",
    target_raw: str = "y_raw",
    group_col: str = "player_id",
    test_size: float = 0.2,
    seed: int = 6114,
    numeric_only: bool = True,
    drop_na: bool = True,
    standardize: bool = True,
):
    """
    Prepare tabular player valuation data for Ridge and NN.

    Input:
        df : pandas DataFrame (already loaded in notebook)

    Returns:
        dict with:
            - X_full, ylog_full, yraw_full, groups_full
            - X_train, X_test
            - ylog_train, ylog_test
            - yraw_train, yraw_test
            - groups_train, groups_test
            - train_idx, test_idx
            - feature_cols
            - scaler
            - summary
    """


    ### Sanity checks ###

    for col in [target_log, target_raw, group_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")

    n_rows_total = len(df)

    ### Ensure boolean features are treated as numeric (0/1) ###
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df = df.copy()
        df[bool_cols] = df[bool_cols].astype(np.float32)

    ### Feature selection ###
    if numeric_only:
        X_df = df.select_dtypes(include=[np.number]).copy()
    else:
        X_df = df.copy()

    ### Drop targets and group column from features ###
    drop_cols = [c for c in [target_log, target_raw, group_col] if c in X_df.columns]
    X_df = X_df.drop(columns=drop_cols, errors="ignore")

    if X_df.shape[1] == 0:
        raise ValueError("No features left after filtering.")

    feature_cols = X_df.columns.tolist()


    ### Targets and groups ###

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    y_log = df[target_log].replace([np.inf, -np.inf], np.nan)
    y_raw = df[target_raw].replace([np.inf, -np.inf], np.nan)
    groups = df[group_col]

    ### Drop NaN / inf rows ###
    if drop_na:
        mask = (
            ~X_df.isna().any(axis=1)
            & ~y_log.isna()
            & ~y_raw.isna()
            & ~groups.isna()
        )
        X_df = X_df.loc[mask]
        y_log = y_log.loc[mask]
        y_raw = y_raw.loc[mask]
        groups = groups.loc[mask]

    n_rows_used = len(X_df)


    ### Full (cleaned) arrays ###

    X_full = X_df.to_numpy(dtype=np.float32)
    ylog_full = y_log.to_numpy(dtype=np.float32).reshape(-1, 1)
    yraw_full = y_raw.to_numpy(dtype=np.float32).reshape(-1, 1)
    groups_full = groups.to_numpy()

    ### Train/test split (leakage safe, groupwise) ###

    idx = np.arange(len(X_full))
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_idx, test_idx = next(
        splitter.split(idx, ylog_full, groups=groups_full)
    )

    X_train_raw = X_full[train_idx]
    X_test_raw = X_full[test_idx]

    ylog_train = ylog_full[train_idx]
    ylog_test = ylog_full[test_idx]
    yraw_train = yraw_full[train_idx]
    yraw_test = yraw_full[test_idx]

    groups_train = groups_full[train_idx]
    groups_test = groups_full[test_idx]

    ### Safety check: No player_id in both test and train set ###
    overlap = len(set(groups_train).intersection(set(groups_test)))
    if overlap != 0:
        raise RuntimeError(
            f"Group leakage detected: {overlap} players appear in both train and test."
        )

    ### Scaling ###

    scaler = StandardScaler()
    if standardize:
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test = scaler.transform(X_test_raw).astype(np.float32)
    else:
        X_train = X_train_raw.astype(np.float32)
        X_test = X_test_raw.astype(np.float32)


    ### Make summary dictionary ###

    summary = {
        "n_rows_total": int(n_rows_total),
        "n_rows_used": int(n_rows_used),
        "n_features": int(len(feature_cols)),
        "test_size": float(test_size),
        "seed": int(seed),
        "numeric_only": bool(numeric_only),
        "drop_na": bool(drop_na),
        "standardize": bool(standardize),
        "player_overlap_train_test": int(overlap),
    }

    return {
        # Full data (for plotting / analysis)
        "X_full": X_full,
        "ylog_full": ylog_full,
        "yraw_full": yraw_full,
        "groups_full": groups_full,

        # Train / test split
        "X_train": X_train,
        "X_test": X_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,

        "ylog_train": ylog_train,
        "ylog_test": ylog_test,
        "yraw_train": yraw_train,
        "yraw_test": yraw_test,

        "groups_train": groups_train,
        "groups_test": groups_test,
        "train_idx": train_idx,
        "test_idx": test_idx,

        # Metadata
        "feature_cols": feature_cols,
        "scaler": scaler,
        "summary": summary,
    }
