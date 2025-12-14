# data_agg_nationality.py
# Optimized version with nationality features, 1/0 encoding, and parallel processing with progress tracking

import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

# ----------------------------
# Project paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = PROJECT_ROOT / "processed_player_value"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLAYERS_CSV = DATA_DIR / "players.csv"
VALUATIONS_CSV = DATA_DIR / "player_valuations.csv"
EVENTS_CSV = DATA_DIR / "game_events.csv"

# ----------------------------
# Config
# ----------------------------
SEQ_LEN_T = 20
LAG_MATCHES = 10  # Number of matches for lag features
MIN_PRIOR_GAMES = 3
MAX_SAMPLES = None  # set None to keep all
USE_LOG_TARGET = True
N_JOBS = max(1, cpu_count() - 1)  # Use all but one CPU core

# ----------------------------
# Helpers
# ----------------------------
def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce", utc=False)

def compute_age_years(dob, ref_date):
    """Vectorized age computation"""
    if pd.isna(dob) or pd.isna(ref_date):
        return np.nan
    return (ref_date - dob).days / 365.25

def standardize_position(pos):
    if pd.isna(pos):
        return "UNK"
    p = str(pos).upper()
    if "GOAL" in p or p == "GK":
        return "GK"
    if "DEF" in p:
        return "DEF"
    if "MID" in p:
        return "MID"
    if "ATT" in p or "FORW" in p or "WING" in p or "STRIK" in p:
        return "ATT"
    return p[:10]

def standardize_foot(foot):
    if pd.isna(foot):
        return "UNK"
    f = str(foot).lower()
    if f.startswith("right"):
        return "R"
    if f.startswith("left"):
        return "L"
    if "both" in f:
        return "B"
    return "UNK"

def make_big5_flag(val_df):
    """
    Big-5 leagues flag (England, Spain, Italy, Germany, France).
    Uses player_valuations.csv column: player_club_domestic_competition_id
    """
    BIG5_IDS = {"GB1", "ES1", "IT1", "DE1", "FR1"}
    comp = val_df["player_club_domestic_competition_id"].fillna("").astype(str).str.upper()
    val_df["is_big5_league"] = comp.isin(BIG5_IDS).astype(np.float32)
    return val_df

# ----------------------------
# Load data
# ----------------------------
print("Loading CSVs...")
script_start_time = time.time()
start_time = time.time()
players = pd.read_csv(PLAYERS_CSV)
valuations = pd.read_csv(VALUATIONS_CSV)
events = pd.read_csv(EVENTS_CSV, low_memory=False)
print(f"  Loaded in {time.time() - start_time:.2f}s")

players["date_of_birth"] = safe_to_datetime(players["date_of_birth"])
valuations["date"] = safe_to_datetime(valuations["date"])
events["date"] = safe_to_datetime(events["date"])

valuations = valuations.dropna(subset=["player_id", "date", "market_value_in_eur"])
valuations["market_value_in_eur"] = pd.to_numeric(
    valuations["market_value_in_eur"], errors="coerce"
)
valuations = valuations.dropna(subset=["market_value_in_eur"])
valuations = valuations.sort_values(["player_id", "date"]).reset_index(drop=True)

# ----------------------------
# Static player features with nationality
# ----------------------------
print("\nProcessing static player features...")
start_time = time.time()

# Get nationality column
if "country_of_citizenship" in players.columns:
    nat_col = "country_of_citizenship"
elif "nationality" in players.columns:
    nat_col = "nationality"
else:
    raise ValueError("No nationality column found in players.csv")

players_static = players[
    ["player_id", "height_in_cm", "foot", "position", nat_col]
].copy()

players_static["height_in_cm"] = pd.to_numeric(
    players_static["height_in_cm"], errors="coerce"
)
players_static["foot"] = players_static["foot"].apply(standardize_foot)
players_static["pos_group"] = players_static["position"].apply(standardize_position)

# Standardize nationality (fill missing with UNK)
players_static[nat_col] = players_static[nat_col].fillna("UNK").astype(str).str.strip()

players_dob = players[["player_id", "date_of_birth"]]

# Get all unique countries (should be 184)
all_countries = sorted(players_static[nat_col].unique())
print(f"  Found {len(all_countries)} unique countries")

# Create one-hot encoding for foot, position, and nationality
# Use dtype=int to get 1/0 instead of True/False
static_ohe_foot_pos = pd.get_dummies(
    players_static[["foot", "pos_group"]].fillna("UNK"),
    prefix=["foot", "pos"],
    dtype=np.int8  # Use int8 for 1/0 encoding
)

# Nationality one-hot encoding (184 countries)
static_ohe_nat = pd.get_dummies(
    players_static[[nat_col]],
    prefix="nat",
    dtype=np.int8  # Use int8 for 1/0 encoding
)

# Combine all static features
players_static_num = pd.concat(
    [
        players_static[["player_id", "height_in_cm"]].reset_index(drop=True),
        static_ohe_foot_pos.reset_index(drop=True),
        static_ohe_nat.reset_index(drop=True),
    ],
    axis=1,
).drop_duplicates("player_id")

print(f"  Static features shape: {players_static_num.shape}")
print(f"    - Foot/Pos features: {len(static_ohe_foot_pos.columns)}")
print(f"    - Nationality features: {len(static_ohe_nat.columns)}")
print(f"  Completed in {time.time() - start_time:.2f}s")

# ----------------------------
# Event-based per-game features (optimized)
# ----------------------------
print("\nBuilding per-game event features...")
start_time = time.time()

ev = events.dropna(subset=["date", "game_id"]).copy()
ev["game_id"] = pd.to_numeric(ev["game_id"], errors="coerce").astype("Int64")
ev["minute"] = pd.to_numeric(ev["minute"], errors="coerce")

desc = ev["description"].fillna("")
is_goal = ev["type"] == "Goals"
is_yellow = (ev["type"] == "Cards") & desc.str.contains("Yellow card", case=False)
is_red = (ev["type"] == "Cards") & desc.str.contains("Red card", case=False)
is_sub = ev["type"] == "Substitutions"

def count_events(df, col="player_id", name="count"):
    return (
        df[[col, "game_id"]]
        .dropna()
        .groupby([col, "game_id"])
        .size()
        .rename(name)
        .reset_index()
        .rename(columns={col: "player_id"})
    )

goals = count_events(ev[is_goal], "player_id", "goals")
assists = count_events(ev[is_goal], "player_assist_id", "assists")
yellow = count_events(ev[is_yellow], "player_id", "yellow_cards")
red = count_events(ev[is_red], "player_id", "red_cards")
sub_in = count_events(ev[is_sub], "player_in_id", "sub_in")
sub_out = count_events(ev[is_sub], "player_id", "sub_out")

# Get game dates (optimized: group once)
game_dates = ev.groupby("game_id")["date"].min().reset_index(name="game_date")

# Build per_game efficiently using merge
pairs = pd.concat(
    [goals, assists, yellow, red, sub_in, sub_out], axis=0
)[["player_id", "game_id"]].drop_duplicates()

per_game = pairs.merge(game_dates, on="game_id", how="left")

# Merge all event counts at once (more efficient)
event_dfs = [goals, assists, yellow, red, sub_in, sub_out]
for df in event_dfs:
    per_game = per_game.merge(df, on=["player_id", "game_id"], how="left")

per_game = per_game.fillna(0)
per_game = per_game.sort_values(["player_id", "game_date"]).reset_index(drop=True)

GAME_FEATURES = [
    "goals", "assists", "yellow_cards", "red_cards", "sub_in", "sub_out"
]

print(f"  Completed in {time.time() - start_time:.2f}s")

# ----------------------------
# Pre-compute cumulative sums for each player (OPTIMIZATION)
# ----------------------------
print("\nPre-computing cumulative statistics...")
start_time = time.time()
per_game_cum = per_game.copy()
for feat in GAME_FEATURES:
    per_game_cum[f"cumulative_{feat}"] = per_game.groupby("player_id")[feat].cumsum().astype(np.float32)
print(f"  Completed in {time.time() - start_time:.2f}s")

# ----------------------------
# Build datasets with cumulative and lag features
# ----------------------------
print("\nBuilding datasets with cumulative and lag features...")
start_time = time.time()

# Prepare valuations with static features
val = valuations.merge(players_dob, on="player_id", how="left")
# Vectorized age computation
val["age_years"] = val.apply(
    lambda r: compute_age_years(r["date_of_birth"], r["date"]), axis=1
)
val = val.merge(players_static_num, on="player_id", how="left")
val = make_big5_flag(val)

val["y_raw"] = pd.to_numeric(val["market_value_in_eur"], errors="coerce").astype(np.float32)
val["y_log"] = np.log1p(val["y_raw"])

static_cols = ["height_in_cm", "age_years", "is_big5_league"] + [
    c for c in val.columns if c.startswith("foot_") or c.startswith("pos_") or c.startswith("nat_")
]

# Pre-group for efficiency
pgroups = {pid: g for pid, g in per_game_cum.groupby("player_id")}
vgroups = {pid: g for pid, g in val.groupby("player_id")}

# ----------------------------
# Parallel processing function
# ----------------------------
def process_player(args):
    """Process a single player's data - all config passed via args"""
    (pid, vg, gg, game_features, seq_len_t, min_prior_games, 
     lag_matches, use_log_target, static_cols_list) = args
    
    g_dates = gg["game_date"].to_numpy()
    g_feats = gg[game_features].to_numpy(dtype=np.float32)
    cum_feats = gg[[f"cumulative_{f}" for f in game_features]].to_numpy(dtype=np.float32)
    
    val_dates = vg["date"].to_numpy()
    idxs = np.searchsorted(g_dates, val_dates, side="left")
    
    results = []
    
    for i, n_before in enumerate(idxs):
        if n_before < min_prior_games:
            continue
        
        # RNN sequence (last seq_len_t games)
        seq = g_feats[max(0, n_before - seq_len_t):n_before]
        if seq.shape[0] < seq_len_t:
            seq = np.vstack([np.zeros((seq_len_t - seq.shape[0], seq.shape[1]), dtype=np.float32), seq])
        
        # Target values
        target_raw = float(vg.iloc[i]["y_raw"])
        target_log = float(vg.iloc[i]["y_log"])
        
        # Static features (convert to numpy array)
        static_vals = vg.iloc[i][static_cols_list].to_numpy(dtype=np.float32)
        
        if use_log_target:
            y_val = target_log
        else:
            y_val = target_raw
        
        # Cumulative features
        if n_before > 0:
            cum_values = cum_feats[n_before - 1].copy()
        else:
            cum_values = np.zeros(len(game_features), dtype=np.float32)
        
        # Lag features: last lag_matches games before valuation
        lag_start = max(0, n_before - lag_matches)
        lag_window = g_feats[lag_start:n_before]
        
        # Build row dictionary
        row_dict = {
            "player_id": pid,
            "valuation_date": vg.iloc[i]["date"],
            "y_raw": target_raw,
            "y_log": target_log,
        }
        
        # Add static features
        for j, col in enumerate(static_cols_list):
            row_dict[col] = float(static_vals[j])
        
        # Add cumulative features
        for j, feat in enumerate(game_features):
            row_dict[f"cumulative_{feat}"] = float(cum_values[j])
        
        # Add lag features (sum over last lag_matches games)
        if len(lag_window) > 0:
            lag_sums = lag_window.sum(axis=0)
            for j, feat in enumerate(game_features):
                row_dict[f"lag_{lag_matches}_{feat}"] = float(lag_sums[j])
        else:
            for feat in game_features:
                row_dict[f"lag_{lag_matches}_{feat}"] = 0.0
        
        results.append((
            seq,
            static_vals,
            y_val,
            (pid, vg.iloc[i]["date"]),
            row_dict
        ))
    
    return results

# Prepare arguments for parallel processing
player_args = []
for pid, vg in vgroups.items():
    if pid not in pgroups:
        continue
    player_args.append((
        pid, vg, pgroups[pid], GAME_FEATURES, SEQ_LEN_T, 
        MIN_PRIOR_GAMES, LAG_MATCHES, USE_LOG_TARGET, static_cols
    ))

print(f"\nProcessing {len(player_args)} players using {N_JOBS} cores...")
process_start = time.time()

# Process players with progress tracking
X_seq, X_static, y_out, meta_rows, nn_rows = [], [], [], [], []

# Use parallel processing if N_JOBS > 1, otherwise sequential with progress bar
if N_JOBS > 1:
    try:
        print("  Using parallel processing...")
        with Pool(processes=N_JOBS) as pool:
            # Use imap for progress tracking
            all_results = list(tqdm(
                pool.imap(process_player, player_args),
                total=len(player_args),
                desc="  Processing players",
                unit="player"
            ))
    except Exception as e:
        print(f"  Parallel processing failed, falling back to sequential: {e}")
        all_results = [process_player(args) for args in tqdm(player_args, desc="  Processing players", unit="player")]
else:
    all_results = [process_player(args) for args in tqdm(player_args, desc="  Processing players", unit="player")]

print(f"  Processing completed in {time.time() - process_start:.2f}s")

# Flatten results
print("\nFlattening results...")
flatten_start = time.time()
for results in tqdm(all_results, desc="  Flattening", unit="player"):
    for seq, static, y_val, meta_row, row_dict in results:
        X_seq.append(seq)
        X_static.append(static)
        y_out.append(y_val)
        meta_rows.append(meta_row)
        nn_rows.append(row_dict)

X_seq = np.asarray(X_seq, dtype=np.float32)
X_static = np.asarray(X_static, dtype=np.float32)
y_out = np.asarray(y_out, dtype=np.float32)

meta = pd.DataFrame(meta_rows, columns=["player_id", "valuation_date"])
tabular_df = pd.DataFrame(nn_rows)

print(f"  Flattening completed in {time.time() - flatten_start:.2f}s")
print(f"  Total samples: {len(tabular_df)}")

# Optional downsampling
if MAX_SAMPLES and len(tabular_df) > MAX_SAMPLES:
    idx = np.random.default_rng(0).choice(len(tabular_df), MAX_SAMPLES, replace=False)
    tabular_df = tabular_df.iloc[idx]
    X_seq = X_seq[idx]
    X_static = X_static[idx]
    y_out = y_out[idx]
    meta = meta.iloc[idx]

# ----------------------------
# Save outputs with prefix
# ----------------------------
PREFIX = "nat_"

print(f"\nSaving datasets...")
save_start = time.time()

tabular_df.to_csv(OUT_DIR / f"{PREFIX}nn_tabular_dataset.csv", index=False)
meta.to_csv(OUT_DIR / f"{PREFIX}meta.csv", index=False)

np.savez_compressed(
    OUT_DIR / f"{PREFIX}rnn_dataset.npz",
    X_seq=X_seq,
    X_static=X_static,
    y=y_out,
)

print(f"  Saved in {time.time() - save_start:.2f}s")

print(f"\n{'='*60}")
print(f"Saved datasets to: {OUT_DIR}")
print(f" - {OUT_DIR / f'{PREFIX}nn_tabular_dataset.csv'}")
print(f" - {OUT_DIR / f'{PREFIX}meta.csv'}")
print(f" - {OUT_DIR / f'{PREFIX}rnn_dataset.npz'}")
print(f"\nDataset shapes:")
print(f"  Tabular: {tabular_df.shape}")
print(f"  RNN X_seq: {X_seq.shape} X_static: {X_static.shape}")
print(f"\nColumns in tabular dataset:")
print(f"  - Static features: {len(static_cols)}")
print(f"  - Cumulative features: {len(GAME_FEATURES)}")
print(f"  - Lag_{LAG_MATCHES} features: {len(GAME_FEATURES)}")
print(f"  - Total columns: {len(tabular_df.columns)}")
total_time = time.time() - script_start_time
print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
print(f"{'='*60}")

