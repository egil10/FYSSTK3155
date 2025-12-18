# Nationality data aggregation script version
_Exported on 12/18/2025 at 13:00:23 GMT+1 from Cursor (2.2.36)_

---

**User**

Good day!

Now I want to make another version of the a data aggregation script in @PROJECT 3/Code/Implementations/ !

So basically it should take inspiration from @PROJECT 3/Code/Implementations/data_aggregating.ipynb and @PROJECT 3/Code/Implementations/data_agg_lag_cumulative.ipynb !

This time here are a few difference though!

And also it should output new data to processed data!

This time use the prefix nat_, so final output should be three files:

nat_meta.csv
nat_nn_tabular_dataset.csv
nat_rnn_dataset.npz

This time lets do these improvements!

Lets make the foot and pos features use 1/0 instead of TRUE/FALSE!

Also lets add nationality for each player lets do country of citizenship as one-hot-encoding where we have 184 distinct countries so aka 184 new columns all using 1/0!

Lets goo!

Also lets optimize as much as possible, make sure to not ruin anything else currently there, and also use parallell processing etc and optimizing to make the script run as fast as possible!

Lets goo!

Rows: 278,558
Columns: 28
$ player_id               <dbl> 10, 10, 10, 10, 10, 10, 10, 26, 26, 26…
$ valuation_date          <date> 2013-01-14, 2013-06-19, 2014-01-07, 2…
$ y_raw                   <dbl> 4000000, 2000000, 1000000, 1000000, 10…
$ y_log                   <dbl> 15.20181, 14.50866, 13.81551, 13.81551…
$ height_in_cm            <dbl> 184, 184, 184, 184, 184, 184, 184, 190…
$ age_years               <dbl> 34.60096, 35.02806, 35.58111, 36.07666…
$ is_big5_league          <dbl> 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,…
$ foot_B                  <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FAL…
$ foot_L                  <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FAL…
$ foot_R                  <lgl> TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TR…
$ foot_UNK                <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FAL…
$ pos_ATT                 <lgl> TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TR…
$ pos_DEF                 <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FAL…
$ pos_GK                  <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FAL…
$ pos_MID                 <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FAL…
$ pos_MISSING             <lgl> FALSE, FALSE, FALSE, FALSE, FALSE, FAL…
$ cumulative_goals        <dbl> 11, 16, 21, 24, 28, 40, 40, 0, 0, 0, 0…
$ cumulative_assists      <dbl> 2, 3, 5, 8, 11, 17, 19, 0, 0, 0, 0, 0,…
$ cumulative_yellow_cards <dbl> 6, 8, 9, 10, 12, 16, 18, 2, 3, 3, 3, 3…
$ cumulative_red_cards    <dbl> 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2,…
$ cumulative_sub_in       <dbl> 3, 6, 9, 11, 21, 25, 28, 0, 0, 0, 0, 0…
$ cumulative_sub_out      <dbl> 8, 13, 16, 21, 25, 38, 44, 0, 0, 0, 0,…
$ lag_10_goals            <dbl> 4, 5, 5, 3, 3, 5, 0, 0, 0, 0, 0, 0, 0,…
$ lag_10_assists          <dbl> 1, 1, 3, 2, 1, 2, 2, 0, 0, 0, 0, 0, 0,…
$ lag_10_yellow_cards     <dbl> 1, 2, 2, 0, 2, 3, 2, 2, 3, 3, 3, 3, 3,…
$ lag_10_red_cards        <dbl> 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2,…
$ lag_10_sub_in           <dbl> 2, 3, 3, 2, 7, 1, 3, 0, 0, 0, 0, 0, 0,…
$ lag_10_sub_out          <dbl> 5, 5, 3, 5, 3, 8, 6, 0, 0, 0, 0, 0, 0,…

---

**Cursor**

Reviewing the existing scripts and checking available nationality data.


Checking the players.csv structure to confirm the nationality column name:


Creating an optimized script with nationality features, 1/0 encoding for foot/pos, and parallel processing:


```json
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "nat_aggregation",
   "metadata": {},
   "outputs": [],
   "source": [
    "# data_agg_nationality.py\n",
    "# Optimized version with nationality features, 1/0 encoding, and parallel processing\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "from multiprocessing import Pool, cpu_count\n",
    "from functools import partial\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# ----------------------------\n",
    "# Project paths (portable, notebook-safe)\n",
    "# ----------------------------\n",
    "try:\n",
    "    PROJECT_ROOT = Path(__file__).resolve().parent.parent\n",
    "except NameError:\n",
    "    # Running in Jupyter / interactive\n",
    "    PROJECT_ROOT = Path.cwd().parent\n",
    "\n",
    "DATA_DIR = PROJECT_ROOT / \"Data\"\n",
    "OUT_DIR = PROJECT_ROOT / \"processed_player_value\"\n",
    "OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "PLAYERS_CSV = DATA_DIR / \"players.csv\"\n",
    "VALUATIONS_CSV = DATA_DIR / \"player_valuations.csv\"\n",
    "EVENTS_CSV = DATA_DIR / \"game_events.csv\"\n",
    "\n",
    "# ----------------------------\n",
    "# Config\n",
    "# ----------------------------\n",
    "SEQ_LEN_T = 20\n",
    "LAG_MATCHES = 10  # Number of matches for lag features\n",
    "MIN_PRIOR_GAMES = 3\n",
    "MAX_SAMPLES = None  # set None to keep all\n",
    "USE_LOG_TARGET = True\n",
    "N_JOBS = max(1, cpu_count() - 1)  # Use all but one CPU core\n",
    "\n",
    "# ----------------------------\n",
    "# Helpers\n",
    "# ----------------------------\n",
    "def safe_to_datetime(s):\n",
    "    return pd.to_datetime(s, errors=\"coerce\", utc=False)\n",
    "\n",
    "def compute_age_years(dob, ref_date):\n",
    "    \"\"\"Vectorized age computation\"\"\"\n",
    "    if pd.isna(dob) or pd.isna(ref_date):\n",
    "        return np.nan\n",
    "    return (ref_date - dob).days / 365.25\n",
    "\n",
    "def standardize_position(pos):\n",
    "    if pd.isna(pos):\n",
    "        return \"UNK\"\n",
    "    p = str(pos).upper()\n",
    "    if \"GOAL\" in p or p == \"GK\":\n",
    "        return \"GK\"\n",
    "    if \"DEF\" in p:\n",
    "        return \"DEF\"\n",
    "    if \"MID\" in p:\n",
    "        return \"MID\"\n",
    "    if \"ATT\" in p or \"FORW\" in p or \"WING\" in p or \"STRIK\" in p:\n",
    "        return \"ATT\"\n",
    "    return p[:10]\n",
    "\n",
    "def standardize_foot(foot):\n",
    "    if pd.isna(foot):\n",
    "        return \"UNK\"\n",
    "    f = str(foot).lower()\n",
    "    if f.startswith(\"right\"):\n",
    "        return \"R\"\n",
    "    if f.startswith(\"left\"):\n",
    "        return \"L\"\n",
    "    if \"both\" in f:\n",
    "        return \"B\"\n",
    "    return \"UNK\"\n",
    "\n",
    "def make_big5_flag(val_df):\n",
    "    \"\"\"\n",
    "    Big-5 leagues flag (England, Spain, Italy, Germany, France).\n",
    "    Uses player_valuations.csv column: player_club_domestic_competition_id\n",
    "    \"\"\"\n",
    "    BIG5_IDS = {\"GB1\", \"ES1\", \"IT1\", \"DE1\", \"FR1\"}\n",
    "    comp = val_df[\"player_club_domestic_competition_id\"].fillna(\"\").astype(str).str.upper()\n",
    "    val_df[\"is_big5_league\"] = comp.isin(BIG5_IDS).astype(np.float32)\n",
    "    return val_df\n",
    "\n",
    "# ----------------------------\n",
    "# Load data\n",
    "# ----------------------------\n",
    "print(\"Loading CSVs...\")\n",
    "players = pd.read_csv(PLAYERS_CSV)\n",
    "valuations = pd.read_csv(VALUATIONS_CSV)\n",
    "events = pd.read_csv(EVENTS_CSV, low_memory=False)\n",
    "\n",
    "players[\"date_of_birth\"] = safe_to_datetime(players[\"date_of_birth\"])\n",
    "valuations[\"date\"] = safe_to_datetime(valuations[\"date\"])\n",
    "events[\"date\"] = safe_to_datetime(events[\"date\"])\n",
    "\n",
    "valuations = valuations.dropna(subset=[\"player_id\", \"date\", \"market_value_in_eur\"])\n",
    "valuations[\"market_value_in_eur\"] = pd.to_numeric(\n",
    "    valuations[\"market_value_in_eur\"], errors=\"coerce\"\n",
    ")\n",
    "valuations = valuations.dropna(subset=[\"market_value_in_eur\"])\n",
    "valuations = valuations.sort_values([\"player_id\", \"date\"]).reset_index(drop=True)\n",
    "\n",
    "# ----------------------------\n",
    "# Static player features with nationality\n",
    "# ----------------------------\n",
    "print(\"Processing static player features...\")\n",
    "\n",
    "# Get nationality column\n",
    "if \"country_of_citizenship\" in players.columns:\n",
    "    nat_col = \"country_of_citizenship\"\n",
    "elif \"nationality\" in players.columns:\n",
    "    nat_col = \"nationality\"\n",
    "else:\n",
    "    raise ValueError(\"No nationality column found in players.csv\")\n",
    "\n",
    "players_static = players[\n",
    "    [\"player_id\", \"height_in_cm\", \"foot\", \"position\", nat_col]\n",
    "].copy()\n",
    "\n",
    "players_static[\"height_in_cm\"] = pd.to_numeric(\n",
    "    players_static[\"height_in_cm\"], errors=\"coerce\"\n",
    ")\n",
    "players_static[\"foot\"] = players_static[\"foot\"].apply(standardize_foot)\n",
    "players_static[\"pos_group\"] = players_static[\"position\"].apply(standardize_position)\n",
    "\n",
    "# Standardize nationality (fill missing with UNK)\n",
    "players_static[nat_col] = players_static[nat_col].fillna(\"UNK\").astype(str).str.strip()\n",
    "\n",
    "players_dob = players[[\"player_id\", \"date_of_birth\"]]\n",
    "\n",
    "# Get all unique countries (should be 184)\n",
    "all_countries = sorted(players_static[nat_col].unique())\n",
    "print(f\"Found {len(all_countries)} unique countries\")\n",
    "\n",
    "# Create one-hot encoding for foot, position, and nationality\n",
    "# Use dtype=int to get 1/0 instead of True/False\n",
    "static_ohe_foot_pos = pd.get_dummies(\n",
    "    players_static[[\"foot\", \"pos_group\"]].fillna(\"UNK\"),\n",
    "    prefix=[\"foot\", \"pos\"],\n",
    "    dtype=np.int8  # Use int8 for 1/0 encoding\n",
    ")\n",
    "\n",
    "# Nationality one-hot encoding (184 countries)\n",
    "static_ohe_nat = pd.get_dummies(\n",
    "    players_static[[nat_col]],\n",
    "    prefix=\"nat\",\n",
    "    dtype=np.int8  # Use int8 for 1/0 encoding\n",
    ")\n",
    "\n",
    "# Combine all static features\n",
    "players_static_num = pd.concat(\n",
    "    [\n",
    "        players_static[[\"player_id\", \"height_in_cm\"]].reset_index(drop=True),\n",
    "        static_ohe_foot_pos.reset_index(drop=True),\n",
    "        static_ohe_nat.reset_index(drop=True),\n",
    "    ],\n",
    "    axis=1,\n",
    ").drop_duplicates(\"player_id\")\n",
    "\n",
    "print(f\"Static features shape: {players_static_num.shape}\")\n",
    "print(f\"  - Foot/Pos features: {len(static_ohe_foot_pos.columns)}\")\n",
    "print(f\"  - Nationality features: {len(static_ohe_nat.columns)}\")\n",
    "\n",
    "# ----------------------------\n",
    "# Event-based per-game features (optimized)\n",
    "# ----------------------------\n",
    "print(\"Building per-game event features...\")\n",
    "\n",
    "ev = events.dropna(subset=[\"date\", \"game_id\"]).copy()\n",
    "ev[\"game_id\"] = pd.to_numeric(ev[\"game_id\"], errors=\"coerce\").astype(\"Int64\")\n",
    "ev[\"minute\"] = pd.to_numeric(ev[\"minute\"], errors=\"coerce\")\n",
    "\n",
    "desc = ev[\"description\"].fillna(\"\")\n",
    "is_goal = ev[\"type\"] == \"Goals\"\n",
    "is_yellow = (ev[\"type\"] == \"Cards\") & desc.str.contains(\"Yellow card\", case=False)\n",
    "is_red = (ev[\"type\"] == \"Cards\") & desc.str.contains(\"Red card\", case=False)\n",
    "is_sub = ev[\"type\"] == \"Substitutions\"\n",
    "\n",
    "def count_events(df, col=\"player_id\", name=\"count\"):\n",
    "    return (\n",
    "        df[[col, \"game_id\"]]\n",
    "        .dropna()\n",
    "        .groupby([col, \"game_id\"])\n",
    "        .size()\n",
    "        .rename(name)\n",
    "        .reset_index()\n",
    "        .rename(columns={col: \"player_id\"})\n",
    "    )\n",
    "\n",
    "# Parallel event counting (if beneficial)\n",
    "goals = count_events(ev[is_goal], \"player_id\", \"goals\")\n",
    "assists = count_events(ev[is_goal], \"player_assist_id\", \"assists\")\n",
    "yellow = count_events(ev[is_yellow], \"player_id\", \"yellow_cards\")\n",
    "red = count_events(ev[is_red], \"player_id\", \"red_cards\")\n",
    "sub_in = count_events(ev[is_sub], \"player_in_id\", \"sub_in\")\n",
    "sub_out = count_events(ev[is_sub], \"player_id\", \"sub_out\")\n",
    "\n",
    "# Get game dates (optimized: group once)\n",
    "game_dates = ev.groupby(\"game_id\")[\"date\"].min().reset_index(name=\"game_date\")\n",
    "\n",
    "# Build per_game efficiently using merge\n",
    "pairs = pd.concat(\n",
    "    [goals, assists, yellow, red, sub_in, sub_out], axis=0\n",
    ")[[\"player_id\", \"game_id\"]].drop_duplicates()\n",
    "\n",
    "per_game = pairs.merge(game_dates, on=\"game_id\", how=\"left\")\n",
    "\n",
    "# Merge all event counts at once (more efficient)\n",
    "event_dfs = [goals, assists, yellow, red, sub_in, sub_out]\n",
    "for df in event_dfs:\n",
    "    per_game = per_game.merge(df, on=[\"player_id\", \"game_id\"], how=\"left\")\n",
    "\n",
    "per_game = per_game.fillna(0)\n",
    "per_game = per_game.sort_values([\"player_id\", \"game_date\"]).reset_index(drop=True)\n",
    "\n",
    "GAME_FEATURES = [\n",
    "    \"goals\", \"assists\", \"yellow_cards\", \"red_cards\", \"sub_in\", \"sub_out\"\n",
    "]\n",
    "\n",
    "# ----------------------------\n",
    "# Pre-compute cumulative sums for each player (OPTIMIZATION)\n",
    "# ----------------------------\n",
    "print(\"Pre-computing cumulative statistics...\")\n",
    "per_game_cum = per_game.copy()\n",
    "for feat in GAME_FEATURES:\n",
    "    per_game_cum[f\"cumulative_{feat}\"] = per_game.groupby(\"player_id\")[feat].cumsum().astype(np.float32)\n",
    "\n",
    "# ----------------------------\n",
    "# Build datasets with cumulative and lag features\n",
    "# ----------------------------\n",
    "print(\"Building datasets with cumulative and lag features...\")\n",
    "\n",
    "# Prepare valuations with static features\n",
    "val = valuations.merge(players_dob, on=\"player_id\", how=\"left\")\n",
    "# Vectorized age computation\n",
    "val[\"age_years\"] = val.apply(\n",
    "    lambda r: compute_age_years(r[\"date_of_birth\"], r[\"date\"]), axis=1\n",
    ")\n",
    "val = val.merge(players_static_num, on=\"player_id\", how=\"left\")\n",
    "val = make_big5_flag(val)\n",
    "\n",
    "val[\"y_raw\"] = pd.to_numeric(val[\"market_value_in_eur\"], errors=\"coerce\").astype(np.float32)\n",
    "val[\"y_log\"] = np.log1p(val[\"y_raw\"])\n",
    "\n",
    "static_cols = [\"height_in_cm\", \"age_years\", \"is_big5_league\"] + [\n",
    "    c for c in val.columns if c.startswith(\"foot_\") or c.startswith(\"pos_\") or c.startswith(\"nat_\")\n",
    "]\n",
    "\n",
    "# Pre-group for efficiency\n",
    "pgroups = {pid: g for pid, g in per_game_cum.groupby(\"player_id\")}\n",
    "vgroups = {pid: g for pid, g in val.groupby(\"player_id\")}\n",
    "\n",
    "# ----------------------------\n",
    "# Parallel processing function\n",
    "# ----------------------------\n",
    "def process_player(args):\n",
    "    \"\"\"Process a single player's data\"\"\"\n",
    "    pid, vg, gg = args\n",
    "    \n",
    "    g_dates = gg[\"game_date\"].to_numpy()\n",
    "    g_feats = gg[GAME_FEATURES].to_numpy(dtype=np.float32)\n",
    "    cum_feats = gg[[f\"cumulative_{f}\" for f in GAME_FEATURES]].to_numpy(dtype=np.float32)\n",
    "    \n",
    "    val_dates = vg[\"date\"].to_numpy()\n",
    "    idxs = np.searchsorted(g_dates, val_dates, side=\"left\")\n",
    "    \n",
    "    results = []\n",
    "    \n",
    "    for i, n_before in enumerate(idxs):\n",
    "        if n_before < MIN_PRIOR_GAMES:\n",
    "            continue\n",
    "        \n",
    "        # RNN sequence (last SEQ_LEN_T games)\n",
    "        seq = g_feats[max(0, n_before - SEQ_LEN_T):n_before]\n",
    "        if seq.shape[0] < SEQ_LEN_T:\n",
    "            seq = np.vstack([np.zeros((SEQ_LEN_T - seq.shape[0], seq.shape[1]), dtype=np.float32), seq])\n",
    "        \n",
    "        # Target values\n",
    "        target_raw = float(vg.iloc[i][\"y_raw\"])\n",
    "        target_log = float(vg.iloc[i][\"y_log\"])\n",
    "        \n",
    "        # Static features (convert to numpy array)\n",
    "        static_vals = vg.iloc[i][static_cols].to_numpy(dtype=np.float32)\n",
    "        \n",
    "        if USE_LOG_TARGET:\n",
    "            y_val = target_log\n",
    "        else:\n",
    "            y_val = target_raw\n",
    "        \n",
    "        # Cumulative features\n",
    "        if n_before > 0:\n",
    "            cum_values = cum_feats[n_before - 1].copy()\n",
    "        else:\n",
    "            cum_values = np.zeros(len(GAME_FEATURES), dtype=np.float32)\n",
    "        \n",
    "        # Lag features: last LAG_MATCHES games before valuation\n",
    "        lag_start = max(0, n_before - LAG_MATCHES)\n",
    "        lag_window = g_feats[lag_start:n_before]\n",
    "        \n",
    "        # Build row dictionary\n",
    "        row_dict = {\n",
    "            \"player_id\": pid,\n",
    "            \"valuation_date\": vg.iloc[i][\"date\"],\n",
    "            \"y_raw\": target_raw,\n",
    "            \"y_log\": target_log,\n",
    "        }\n",
    "        \n",
    "        # Add static features\n",
    "        for j, col in enumerate(static_cols):\n",
    "            row_dict[col] = float(static_vals[j])\n",
    "        \n",
    "        # Add cumulative features\n",
    "        for j, feat in enumerate(GAME_FEATURES):\n",
    "            row_dict[f\"cumulative_{feat}\"] = float(cum_values[j])\n",
    "        \n",
    "        # Add lag features (sum over last LAG_MATCHES games)\n",
    "        if len(lag_window) > 0:\n",
    "            lag_sums = lag_window.sum(axis=0)\n",
    "            for j, feat in enumerate(GAME_FEATURES):\n",
    "                row_dict[f\"lag_{LAG_MATCHES}_{feat}\"] = float(lag_sums[j])\n",
    "        else:\n",
    "            for feat in GAME_FEATURES:\n",
    "                row_dict[f\"lag_{LAG_MATCHES}_{feat}\"] = 0.0\n",
    "        \n",
    "        results.append((\n",
    "            seq,\n",
    "            static_vals,\n",
    "            y_val,\n",
    "            (pid, vg.iloc[i][\"date\"]),\n",
    "            row_dict\n",
    "        ))\n",
    "    \n",
    "    return results\n",
    "\n",
    "# Prepare arguments for parallel processing\n",
    "player_args = []\n",
    "for pid, vg in vgroups.items():\n",
    "    if pid not in pgroups:\n",
    "        continue\n",
    "    player_args.append((pid, vg, pgroups[pid]))\n",
    "\n",
    "print(f\"Processing {len(player_args)} players using {N_JOBS} cores...\")\n",
    "\n",
    "# Process players in parallel\n",
    "X_seq, X_static, y_out, meta_rows, nn_rows = [], [], [], [], []\n",
    "\n",
    "with Pool(processes=N_JOBS) as pool:\n",
    "    all_results = pool.map(process_player, player_args)\n",
    "\n",
    "# Flatten results\n",
    "for results in all_results:\n",
    "    for seq, static, y_val, meta_row, row_dict in results:\n",
    "        X_seq.append(seq)\n",
    "        X_static.append(static)\n",
    "        y_out.append(y_val)\n",
    "        meta_rows.append(meta_row)\n",
    "        nn_rows.append(row_dict)\n",
    "\n",
    "X_seq = np.asarray(X_seq, dtype=np.float32)\n",
    "X_static = np.asarray(X_static, dtype=np.float32)\n",
    "y_out = np.asarray(y_out, dtype=np.float32)\n",
    "\n",
    "meta = pd.DataFrame(meta_rows, columns=[\"player_id\", \"valuation_date\"])\n",
    "tabular_df = pd.DataFrame(nn_rows)\n",
    "\n",
    "# Optional downsampling\n",
    "if MAX_SAMPLES and len(tabular_df) > MAX_SAMPLES:\n",
    "    idx = np.random.default_rng(0).choice(len(tabular_df), MAX_SAMPLES, replace=False)\n",
    "    tabular_df = tabular_df.iloc[idx]\n",
    "    X_seq = X_seq[idx]\n",
    "    X_static = X_static[idx]\n",
    "    y_out = y_out[idx]\n",
    "    meta = meta.iloc[idx]\n",
    "\n",
    "# ----------------------------\n",
    "# Save outputs with prefix\n",
    "# ----------------------------\n",
    "PREFIX = \"nat_\"\n",
    "\n",
    "tabular_df.to_csv(OUT_DIR / f\"{PREFIX}nn_tabular_dataset.csv\", index=False)\n",
    "meta.to_csv(OUT_DIR / f\"{PREFIX}meta.csv\", index=False)\n",
    "\n",
    "np.savez_compressed(\n",
    "    OUT_DIR / f\"{PREFIX}rnn_dataset.npz\",\n",
    "    X_seq=X_seq,\n",
    "    X_static=X_static,\n",
    "    y=y_out,\n",
    ")\n",
    "\n",
    "print(f\"\\nSaved datasets to: {OUT_DIR}\")\n",
    "print(f\" - {OUT_DIR / f'{PREFIX}nn_tabular_dataset.csv'}\")\n",
    "print(f\" - {OUT_DIR / f'{PREFIX}meta.csv'}\")\n",
    "print(f\" - {OUT_DIR / f'{PREFIX}rnn_dataset.npz'}\")\n",
    "print(f\"Tabular shape: {tabular_df.shape}\")\n",
    "print(f\"RNN X_seq: {X_seq.shape} X_static: {X_static.shape}\")\n",
    "print(f\"\\nColumns in tabular dataset:\")\n",
    "print(f\"  - Static features: {len(static_cols)}\")\n",
    "print(f\"  - Cumulative features: {len(GAME_FEATURES)}\")\n",
    "print(f\"  - Lag_{LAG_MATCHES} features: {len(GAME_FEATURES)}\")\n",
    "print(f\"  - Total columns: {len(tabular_df.columns)}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "versi
```

Creating the notebook using the correct tool:


```plaintext


->

# data_agg_nationality.py
# Optimized version with nationality features, 1/0 encoding, and parallel processing

import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Project paths (portable, notebook-safe)
# ----------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Running in Jupyter / interactive
    PROJECT_ROOT = Path.cwd().parent

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
players = pd.read_csv(PLAYERS_CSV)
valuations = pd.read_csv(VALUATIONS_CSV)
events = pd.read_csv(EVENTS_CSV, low_memory=False)

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
print("Processing static player features...")

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
print(f"Found {len(all_countries)} unique countries")

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

print(f"Static features shape: {players_static_num.shape}")
print(f"  - Foot/Pos features: {len(static_ohe_foot_pos.columns)}")
print(f"  - Nationality features: {len(static_ohe_nat.columns)}")

# ----------------------------
# Event-based per-game features (optimized)
# ----------------------------
print("Building per-game event features...")

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

# Parallel event counting (if beneficial)
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

# ----------------------------
# Pre-compute cumulative sums for each player (OPTIMIZATION)
# ----------------------------
print("Pre-computing cumulative statistics...")
per_game_cum = per_game.copy()
for feat in GAME_FEATURES:
    per_game_cum[f"cumulative_{feat}"] = per_game.groupby("player_id")[feat].cumsum().astype(np.float32)

# ----------------------------
# Build datasets with cumulative and lag features
# ----------------------------
print("Building datasets with cumulative and lag features...")

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
    """Process a single player's data"""
    pid, vg, gg = args
    
    g_dates = gg["game_date"].to_numpy()
    g_feats = gg[GAME_FEATURES].to_numpy(dtype=np.float32)
    cum_feats = gg[[f"cumulative_{f}" for f in GAME_FEATURES]].to_numpy(dtype=np.float32)
    
    val_dates = vg["date"].to_numpy()
    idxs = np.searchsorted(g_dates, val_dates, side="left")
    
    results = []
    
    for i, n_before in enumerate(idxs):
        if n_before < MIN_PRIOR_GAMES:
            continue
        
        # RNN sequence (last SEQ_LEN_T games)
        seq = g_feats[max(0, n_before - SEQ_LEN_T):n_before]
        if seq.shape[0] < SEQ_LEN_T:
            seq = np.vstack([np.zeros((SEQ_LEN_T - seq.shape[0], seq.shape[1]), dtype=np.float32), seq])
        
        # Target values
        target_raw = float(vg.iloc[i]["y_raw"])
        target_log = float(vg.iloc[i]["y_log"])
        
        # Static features (convert to numpy array)
        static_vals = vg.iloc[i][static_cols].to_numpy(dtype=np.float32)
        
        if USE_LOG_TARGET:
            y_val = target_log
        else:
            y_val = target_raw
        
        # Cumulative features
        if n_before > 0:
            cum_values = cum_feats[n_before - 1].copy()
        else:
            cum_values = np.zeros(len(GAME_FEATURES), dtype=np.float32)
        
        # Lag features: last LAG_MATCHES games before valuation
        lag_start = max(0, n_before - LAG_MATCHES)
        lag_window = g_feats[lag_start:n_before]
        
        # Build row dictionary
        row_dict = {
            "player_id": pid,
            "valuation_date": vg.iloc[i]["date"],
            "y_raw": target_raw,
            "y_log": target_log,
        }
        
        # Add static features
        for j, col in enumerate(static_cols):
            row_dict[col] = float(static_vals[j])
        
        # Add cumulative features
        for j, feat in enumerate(GAME_FEATURES):
            row_dict[f"cumulative_{feat}"] = float(cum_values[j])
        
        # Add lag features (sum over last LAG_MATCHES games)
        if len(lag_window) > 0:
            lag_sums = lag_window.sum(axis=0)
            for j, feat in enumerate(GAME_FEATURES):
                row_dict[f"lag_{LAG_MATCHES}_{feat}"] = float(lag_sums[j])
        else:
            for feat in GAME_FEATURES:
                row_dict[f"lag_{LAG_MATCHES}_{feat}"] = 0.0
        
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
    player_args.append((pid, vg, pgroups[pid]))

print(f"Processing {len(player_args)} players using {N_JOBS} cores...")

# Process players in parallel
X_seq, X_static, y_out, meta_rows, nn_rows = [], [], [], [], []

with Pool(processes=N_JOBS) as pool:
    all_results = pool.map(process_player, player_args)

# Flatten results
for results in all_results:
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

tabular_df.to_csv(OUT_DIR / f"{PREFIX}nn_tabular_dataset.csv", index=False)
meta.to_csv(OUT_DIR / f"{PREFIX}meta.csv", index=False)

np.savez_compressed(
    OUT_DIR / f"{PREFIX}rnn_dataset.npz",
    X_seq=X_seq,
    X_static=X_static,
    y=y_out,
)

print(f"\nSaved datasets to: {OUT_DIR}")
print(f" - {OUT_DIR / f'{PREFIX}nn_tabular_dataset.csv'}")
print(f" - {OUT_DIR / f'{PREFIX}meta.csv'}")
print(f" - {OUT_DIR / f'{PREFIX}rnn_dataset.npz'}")
print(f"Tabular shape: {tabular_df.shape}")
print(f"RNN X_seq: {X_seq.shape} X_static: {X_static.shape}")
print(f"\nColumns in tabular dataset:")
print(f"  - Static features: {len(static_cols)}")
print(f"  - Cumulative features: {len(GAME_FEATURES)}")
print(f"  - Lag_{LAG_MATCHES} features: {len(GAME_FEATURES)}")
print(f"  - Total columns: {len(tabular_df.columns)}")
```

Fixing multiprocessing to work in Jupyter notebooks by making the parallel function self-contained and passing all needed parameters:


Updating the parallel processing function to pass all required parameters explicitly:


```json
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

print(f"Processing {len(player_args)} players using {N_JOBS} cores...")

# Process players in parallel
X_seq, X_static, y_out, meta_rows, nn_rows = [], [], [], [], []

# Use parallel processing if N_JOBS > 1, otherwise sequential (for debugging)
if N_JOBS > 1:
    try:
        with Pool(processes=N_JOBS) as pool:
            all_results = pool.map(process_player, player_args)
    except Exception as e:
        print(f"Parallel processing failed, falling back to sequential: {e}")
        all_results = [process_player(args) for args in player_args]
else:
   
```

Reading the relevant section and updating it:


Updating the function to pass all parameters explicitly:


```plaintext
# ----------------------------
# Parallel processing function
# ----------------------------
def process_player(args):
    """Process a single player's data"""
    pid, vg, gg = args
    
    g_dates = gg["game_date"].to_numpy()
    g_feats = gg[GAME_FEATURES].to_numpy(dtype=np.float32)
    cum_feats = gg[[f"cumulative_{f}" for f in GAME_FEATURES]].to_numpy(dtype=np.float32)
    
    val_dates = vg["date"].to_numpy()
    idxs = np.searchsorted(g_dates, val_dates, side="left")
    
    results = []
    
    for i, n_before in enumerate(idxs):
        if n_before < MIN_PRIOR_GAMES:
            continue
        
        # RNN sequence (last SEQ_LEN_T games)
        seq = g_feats[max(0, n_before - SEQ_LEN_T):n_before]
        if seq.shape[0] < SEQ_LEN_T:
            seq = np.vstack([np.zeros((SEQ_LEN_T - seq.shape[0], seq.shape[1]), dtype=np.float32), seq])
        
        # Target values
        target_raw = float(vg.iloc[i]["y_raw"])
        target_log = float(vg.iloc[i]["y_log"])
        
        # Static features (convert to numpy array)
        static_vals = vg.iloc[i][static_cols].to_numpy(dtype=np.float32)
        
        if USE_LOG_TARGET:
            y_val = target_log
        else:
            y_val = target_raw
        
        # Cumulative features
        if n_before > 0:
            cum_values = cum_feats[n_before - 1].copy()
        else:
            cum_values = np.zeros(len(GAME_FEATURES), dtype=np.float32)
        
        # Lag features: last LAG_MATCHES games before valuation
        lag_start = max(0, n_before - LAG_MATCHES)
        lag_window = g_feats[lag_start:n_before]
        
        # Build row dictionary
        row_dict = {
            "player_id": pid,
            "valuation_date": vg.iloc[i]["date"],
            "y_raw": target_raw,
            "y_log": target_log,
        }
        
        # Add static features
        for j, col in enumerate(static_cols):
            row_dict[col] = float(static_vals[j])
        
        # Add cumulative features
        for j, feat in enumerate(GAME_FEATURES):
            row_dict[f"cumulative_{feat}"] = float(cum_values[j])
        
        # Add lag features (sum over last LAG_MATCHES games)
        if len(lag_window) > 0:
            lag_sums = lag_window.sum(axis=0)
            for j, feat in enumerate(GAME_FEATURES):
                row_dict[f"lag_{LAG_MATCHES}_{feat}"] = float(lag_sums[j])
        else:
            for feat in GAME_FEATURES:
                row_dict[f"lag_{LAG_MATCHES}_{feat}"] = 0.0
        
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
    player_args.append((pid, vg, pgroups[pid]))

print(f"Processing {len(player_args)} players using {N_JOBS} cores...")

# Process players in parallel
X_seq, X_static, y_out, meta_rows, nn_rows = [], [], [], [], []

with Pool(processes=N_JOBS) as pool:
    all_results = pool.map(process_player, player_args)

->

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

print(f"Processing {len(player_args)} players using {N_JOBS} cores...")

# Process players in parallel
X_seq, X_static, y_out, meta_rows, nn_rows = [], [], [], [], []

# Use parallel processing if N_JOBS > 1, otherwise sequential (for debugging)
if N_JOBS > 1:
    try:
        with Pool(processes=N_JOBS) as pool:
            all_results = pool.map(process_player, player_args)
    except Exception as e:
        print(f"Parallel processing failed, falling back to sequential: {e}")
        all_results = [process_player(args) for args in player_args]
else:
    all_results = [process_player(args) for args in player_args]
```

Removing the unused import:


```plaintext
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

->

import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')
```

Created the optimized data aggregation script with nationality features. Summary:

## Features

1. **Prefix `nat_`**: Outputs `nat_meta.csv`, `nat_nn_tabular_dataset.csv`, and `nat_rnn_dataset.npz`

2. **1/0 encoding**: Foot and position features use `dtype=np.int8` (1/0) instead of boolean

3. **Nationality features**: One-hot encoding for all countries (184) with prefix `nat_`, using 1/0 encoding

4. **Cumulative and lag features**: Includes cumulative statistics and lag_10 features like the second script

5. **Optimizations**:
   - Parallel processing using `multiprocessing.Pool` (uses all but one CPU core)
   - Pre-computed cumulative sums
   - Efficient grouping and merging
   - Fallback to sequential processing if parallel fails (useful for Windows/Jupyter)

6. **All existing features preserved**: Static features, game features, RNN sequences, etc.

The script processes players in parallel, which should speed up execution. It automatically detects the nationality column (`country_of_citizenship` or `nationality`) and creates one-hot encoded features for all countries found in the dataset.

You can run the notebook to generate the three output files with the `nat_` prefix. The script will print progress information including the number of countries found and the final dataset shapes.

---

**User**

Bro it has run for 20 minutes ... are there better ways to make it run faster?????? 

Also I have no way of knowing how far along its gone since there is no feedback in the thing so I have no clue if I should stop it or not 

---

**Cursor**

Adding progress tracking and optimizing the code. Multiprocessing in Jupyter on Windows can be slow. Adding progress bars and improving performance:


Replacing multiprocessing (problematic in Jupyter on Windows) with a sequential approach and adding progress tracking. Optimizing the code:


---

**User**

Can we run it as a script instead of jupyter is that faster`?

---

**Cursor**

Converting to a Python script with progress tracking and optimizations. Scripts are faster and more reliable for multiprocessing.


Creating an optimized Python script with progress tracking and fixing the output directory:


```python
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
print(f"\nTotal time: {time.time() - start_time:.2f}s")
print(f"{'='*60}")

```

Fixing the total time calculation:


```python
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
total_time = time.time() - start_time
print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
print(f"{'='*60}")


```

```python
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
total_time = time.time() - start_time
print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
print(f"{'='*60}")


```

```python
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


```

Checking for linting errors and verifying the script:


Created an optimized Python script with progress tracking. Changes:

## Improvements:

1. **Progress bars** — Uses `tqdm` to show progress in real time
2. **Timing info** — Shows how long each step takes
3. **Fixed output directory** — Now saves to `processed_player_value` (not "Processed")
4. **Better parallel processing** — Uses `imap` for progress tracking in multiprocessing
5. **Faster as a script** — Standalone Python scripts run faster than Jupyter notebooks

## To run it:

```bash
cd "PROJECT 3/Code/Implementations"
python data_agg_nationality.py
```

Or from the project root:
```bash
python "PROJECT 3/Code/Implementations/data_agg_nationality.py"
```

The script will show:
- Progress bars for processing players
- Time taken for each step
- Total time at the end
- Real-time feedback so you can see it's working

If `tqdm` isn't installed, install it:
```bash
pip install tqdm
```

The script should be faster than the notebook, especially with multiprocessing. You'll see progress bars so you know it's working.

```shellscript
cd "PROJECT 3/Code/Implementations"
python data_agg_nationality.py
```

```shellscript
python "PROJECT 3/Code/Implementations/data_agg_nationality.py"
```

```shellscript
pip install tqdm
```

---

**User**

Bro it is still slow. Lets do this. 

Lets take @PROJECT 3/Code/Implementations/data_agg_cumlag.ipynb and copy and paste it and only add one-shot feature columns with nationality so one column per nationality and 1/0 down. Thats it. Pls fix. 

---

**Cursor**

Creating a simpler version based on the cumlag script, adding only nationality one-hot encoding (1/0). No parallel processing complexity.


Creating a simple version that adds only nationality one-hot encoding:


```plaintext
# data_agg_nationality.py
# Optimized version with nationality features, 1/0 encoding, and parallel processing

import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Project paths (portable, notebook-safe)
# ----------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Running in Jupyter / interactive
    PROJECT_ROOT = Path.cwd().parent

DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = PROJECT_ROOT / "Processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

->

# data_agg_nationality.py
# Simple version with nationality features and 1/0 encoding

import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Project paths (portable, notebook-safe)
# ----------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Running in Jupyter / interactive
    PROJECT_ROOT = Path.cwd().parent

DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = PROJECT_ROOT / "processed_player_value"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

Replacing the entire cell with the simple cumlag version plus nationality:


```plaintext
# ----------------------------
# Config
# ----------------------------
SEQ_LEN_T = 20
LAG_MATCHES = 10  # Number of matches for lag features
MIN_PRIOR_GAMES = 3
MAX_SAMPLES = None  # set None to keep all
USE_LOG_TARGET = True
N_JOBS = max(1, cpu_count() - 1)  # Use all but one CPU core

->

# ----------------------------
# Config
# ----------------------------
SEQ_LEN_T = 20
LAG_MATCHES = 10  # Number of matches for lag features
MIN_PRIOR_GAMES = 3
MAX_SAMPLES = None  # set None to keep all
USE_LOG_TARGET = True
```

Updating the static features section to add nationality and use 1/0 encoding:


```plaintext
# ----------------------------
# Static player features with nationality
# ----------------------------
print("Processing static player features...")

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
print(f"Found {len(all_countries)} unique countries")

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

print(f"Static features shape: {players_static_num.shape}")
print(f"  - Foot/Pos features: {len(static_ohe_foot_pos.columns)}")
print(f"  - Nationality features: {len(static_ohe_nat.columns)}")

->

# ----------------------------
# Static player features
# ----------------------------
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

# Create one-hot encoding for foot, position (with 1/0 encoding)
static_ohe_foot_pos = pd.get_dummies(
    players_static[["foot", "pos_group"]].fillna("UNK"),
    prefix=["foot", "pos"],
    dtype=np.int8  # Use int8 for 1/0 encoding
)

# Nationality one-hot encoding (1/0 encoding)
static_ohe_nat = pd.get_dummies(
    players_static[[nat_col]],
    prefix="nat",
    dtype=np.int8  # Use int8 for 1/0 encoding
)

players_static_num = pd.concat(
    [
        players_static[["player_id", "height_in_cm"]].reset_index(drop=True),
        static_ohe_foot_pos.reset_index(drop=True),
        static_ohe_nat.reset_index(drop=True),
    ],
    axis=1,
).drop_duplicates("player_id")
```

Replacing the processing section with the simple cumlag loop and updating static_cols:


```plaintext
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

print(f"Processing {len(player_args)} players using {N_JOBS} cores...")

# Process players in parallel
X_seq, X_static, y_out, meta_rows, nn_rows = [], [], [], [], []

# Use parallel processing if N_JOBS > 1, otherwise sequential (for debugging)
if N_JOBS > 1:
    try:
        with Pool(processes=N_JOBS) as pool:
            all_results = pool.map(process_player, player_args)
    except Exception as e:
        print(f"Parallel processing failed, falling back to sequential: {e}")
        all_results = [process_player(args) for args in player_args]
else:
    all_results = [process_player(args) for args in player_args]

# Flatten results
for results in all_results:
    for seq, static, y_val, meta_row, row_dict in results:
        X_seq.append(seq)
        X_static.append(static)
        y_out.append(y_val)
        meta_rows.append(meta_row)
        nn_rows.append(row_dict)

->

static_cols = ["height_in_cm", "age_years", "is_big5_league"] + [
    c for c in val.columns if c.startswith("foot_") or c.startswith("pos_") or c.startswith("nat_")
]

# Pre-group for efficiency
pgroups = {pid: g for pid, g in per_game_cum.groupby("player_id")}
vgroups = {pid: g for pid, g in val.groupby("player_id")}

X_seq, X_static, y_out, meta_rows, nn_rows = [], [], [], [], []

# Process each player
for pid, vg in vgroups.items():
    if pid not in pgroups:
        continue

    gg = pgroups[pid].copy()
    g_dates = gg["game_date"].to_numpy()
    g_feats = gg[GAME_FEATURES].to_numpy(dtype=np.float32)
    
    # Get cumulative features (exclude current row, so shift by 1)
    cum_feats = gg[[f"cumulative_{f}" for f in GAME_FEATURES]].to_numpy(dtype=np.float32)
    
    # Get valuation dates for this player
    val_dates = vg["date"].to_numpy()
    idxs = np.searchsorted(g_dates, val_dates, side="left")

    for i, n_before in enumerate(idxs):
        if n_before < MIN_PRIOR_GAMES:
            continue

        # RNN sequence (last SEQ_LEN_T games)
        seq = g_feats[max(0, n_before - SEQ_LEN_T):n_before]
        if seq.shape[0] < SEQ_LEN_T:
            seq = np.vstack([np.zeros((SEQ_LEN_T - seq.shape[0], seq.shape[1]), dtype=np.float32), seq])

        # Target values
        target_raw = float(vg.iloc[i]["y_raw"])
        target_log = float(vg.iloc[i]["y_log"])

        X_seq.append(seq)
        X_static.append(vg.iloc[i][static_cols].to_numpy(dtype=np.float32))

        if USE_LOG_TARGET:
            y_out.append(target_log)
        else:
            y_out.append(target_raw)

        meta_rows.append((pid, vg.iloc[i]["date"]))

        # Cumulative features: total up to (and including) the last game before valuation
        # n_before is the number of games before valuation, so index n_before-1 gives cumulative up to last game
        if n_before > 0:
            cum_values = cum_feats[n_before - 1].copy()  # Cumulative sum up to last game before valuation
        else:
            cum_values = np.zeros(len(GAME_FEATURES), dtype=np.float32)

        # Lag features: last LAG_MATCHES games before valuation
        lag_start = max(0, n_before - LAG_MATCHES)
        lag_window = g_feats[lag_start:n_before]
        
        # Build row with cumulative and lag features
        row_dict = {
            "player_id": pid,
            "valuation_date": vg.iloc[i]["date"],
            "y_raw": target_raw,
            "y_log": target_log,
            **vg.iloc[i][static_cols].to_dict(),
        }
        
        # Add cumulative features
        for j, feat in enumerate(GAME_FEATURES):
            row_dict[f"cumulative_{feat}"] = float(cum_values[j])
        
        # Add lag features (sum over last LAG_MATCHES games)
        if len(lag_window) > 0:
            lag_sums = lag_window.sum(axis=0)
            for j, feat in enumerate(GAME_FEATURES):
                row_dict[f"lag_{LAG_MATCHES}_{feat}"] = float(lag_sums[j])
        else:
            for feat in GAME_FEATURES:
                row_dict[f"lag_{LAG_MATCHES}_{feat}"] = 0.0
        
        nn_rows.append(row_dict)
```

Updating the prefix to "nat_":


```plaintext
# ----------------------------
# Save outputs with prefix
# ----------------------------
PREFIX = "nat_"

tabular_df.to_csv(OUT_DIR / f"{PREFIX}nn_tabular_dataset.csv", index=False)
meta.to_csv(OUT_DIR / f"{PREFIX}meta.csv", index=False)

np.savez_compressed(
    OUT_DIR / f"{PREFIX}rnn_dataset.npz",
    X_seq=X_seq,
    X_static=X_static,
    y=y_out,
)

print(f"\nSaved datasets to: {OUT_DIR}")
print(f" - {OUT_DIR / f'{PREFIX}nn_tabular_dataset.csv'}")
print(f" - {OUT_DIR / f'{PREFIX}meta.csv'}")
print(f" - {OUT_DIR / f'{PREFIX}rnn_dataset.npz'}")
print(f"Tabular shape: {tabular_df.shape}")
print(f"RNN X_seq: {X_seq.shape} X_static: {X_static.shape}")
print(f"\nColumns in tabular dataset:")
print(f"  - Static features: {len(static_cols)}")
print(f"  - Cumulative features: {len(GAME_FEATURES)}")
print(f"  - Lag_{LAG_MATCHES} features: {len(GAME_FEATURES)}")
print(f"  - Total columns: {len(tabular_df.columns)}")

->

# ----------------------------
# Save outputs with prefix
# ----------------------------
PREFIX = "nat_"

tabular_df.to_csv(OUT_DIR / f"{PREFIX}nn_tabular_dataset.csv", index=False)
meta.to_csv(OUT_DIR / f"{PREFIX}meta.csv", index=False)

np.savez_compressed(
    OUT_DIR / f"{PREFIX}rnn_dataset.npz",
    X_seq=X_seq,
    X_static=X_static,
    y=y_out,
)

print(f"Saved datasets to: {OUT_DIR}")
print(f" - {OUT_DIR / f'{PREFIX}nn_tabular_dataset.csv'}")
print(f" - {OUT_DIR / f'{PREFIX}meta.csv'}")
print(f" - {OUT_DIR / f'{PREFIX}rnn_dataset.npz'}")
print(f"Tabular shape: {tabular_df.shape}")
print(f"RNN X_seq: {X_seq.shape} X_static: {X_static.shape}")
print(f"\nColumns in tabular dataset:")
print(f"  - Static features: {len(static_cols)}")
print(f"  - Cumulative features: {len(GAME_FEATURES)}")
print(f"  - Lag_{LAG_MATCHES} features: {len(GAME_FEATURES)}")
```

Created a simpler version based on the cumlag script. Changes:

1. Removed parallel processing — uses the same simple loop as cumlag
2. Added nationality one-hot encoding — 1/0 columns (int8) with prefix `nat_`
3. Foot/pos features use 1/0 encoding — changed to `dtype=np.int8` instead of boolean
4. Everything else stays the same — same structure, same speed as cumlag

The notebook should run faster since it uses the same simple processing loop as cumlag, with nationality features added. It will output:
- `nat_meta.csv`
- `nat_nn_tabular_dataset.csv`
- `nat_rnn_dataset.npz`

All nationality, foot, and position features are encoded as 1/0 (int8) instead of True/False.

---

**User**

Nice but bro now the nat_ datasets exceeded 100 MB, too big for github, should we make the csv into json or something else and push this instead?

If so lets keep it locally, gitignore it, and run something to also add a json or something else equivalent for the csv and push this

---

**Cursor**

Switching to Parquet for the tabular data (smaller, faster). Updating the script to save both CSV (local) and Parquet (for GitHub), and updating `.gitignore` to exclude the CSV files.


---

**User**

No bro we already ran it, so lets just do a script to convert the csv to parquet or json?

---

**Cursor**

Creating a script to convert the existing CSV files to Parquet (smaller and faster than JSON):


```python
# convert_csv_to_parquet.py
# Convert large CSV files to Parquet format (much smaller and faster)

import pandas as pd
from pathlib import Path

# ----------------------------
# Project paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "processed_player_value"

# Files to convert (with nat_ prefix)
FILES_TO_CONVERT = [
    "nat_nn_tabular_dataset.csv",
    "nat_meta.csv",
]

print("Converting CSV files to Parquet format...")
print(f"Output directory: {OUT_DIR}\n")

for csv_file in FILES_TO_CONVERT:
    csv_path = OUT_DIR / csv_file
    
    if not csv_path.exists():
        print(f"⚠️  Skipping {csv_file} (not found)")
        continue
    
    print(f"📖 Reading {csv_file}...")
    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape}")
    
    # Get file size
    csv_size = csv_path.stat().st_size / (1024 * 1024)  # MB
    print(f"   CSV size: {csv_size:.2f} MB")
    
    # Convert to parquet
    parquet_file = csv_file.replace(".csv", ".parquet")
    parquet_path = OUT_DIR / parquet_file
    
    print(f"💾 Saving to {parquet_file}...")
    df.to_parquet(parquet_path, compression='snappy', index=False)
    
    # Get parquet file size
    parquet_size = parquet_path.stat().st_size / (1024 * 1024)  # MB
    reduction = (1 - parquet_size / csv_size) * 100
    
    print(f"   Parquet size: {parquet_size:.2f} MB")
    print(f"   Size reduction: {reduction:.1f}%")
    print()

print("✅ Conversion complete!")
print(f"\nParquet files saved to: {OUT_DIR}")
print("You can now add CSV files to .gitignore and commit Parquet files instead.")

```

Removing emojis for Windows compatibility:


```python
# convert_csv_to_parquet.py
# Convert large CSV files to Parquet format (much smaller and faster)

import pandas as pd
from pathlib import Path

# ----------------------------
# Project paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "processed_player_value"

# Files to convert (with nat_ prefix)
FILES_TO_CONVERT = [
    "nat_nn_tabular_dataset.csv",
    "nat_meta.csv",
]

print("Converting CSV files to Parquet format...")
print(f"Output directory: {OUT_DIR}\n")

for csv_file in FILES_TO_CONVERT:
    csv_path = OUT_DIR / csv_file
    
    if not csv_path.exists():
        print(f"[SKIP] Skipping {csv_file} (not found)")
        continue
    
    print(f"[READ] Reading {csv_file}...")
    df = pd.read_csv(csv_path)
    print(f"       Shape: {df.shape}")
    
    # Get file size
    csv_size = csv_path.stat().st_size / (1024 * 1024)  # MB
    print(f"       CSV size: {csv_size:.2f} MB")
    
    # Convert to parquet
    parquet_file = csv_file.replace(".csv", ".parquet")
    parquet_path = OUT_DIR / parquet_file
    
    print(f"[SAVE] Saving to {parquet_file}...")
    df.to_parquet(parquet_path, compression='snappy', index=False)
    
    # Get parquet file size
    parquet_size = parquet_path.stat().st_size / (1024 * 1024)  # MB
    reduction = (1 - parquet_size / csv_size) * 100
    
    print(f"       Parquet size: {parquet_size:.2f} MB")
    print(f"       Size reduction: {reduction:.1f}%")
    print()

print("[DONE] Conversion complete!")
print(f"\nParquet files saved to: {OUT_DIR}")
print("You can now add CSV files to .gitignore and commit Parquet files instead.")


```

Conversion succeeded. Parquet files are ~97% smaller. Updating `.gitignore` to exclude the CSV files:


```ignore
## Transfermarkt raw exports
Code/Data/*
!Code/Data/.gitkeep
!Code/Data/pl_team_features.csv
!Code/Data/pl_team_features.parquet
!Code/Data/game_features.parquet
# CSV is typically much larger, comment out if you need it
# !Code/Data/game_features.csv
!Code/Data/predictive_features.txt
!Code/Data/README.md

## Large processed datasets (use Parquet instead)
Code/processed_player_value/*.csv
!Code/processed_player_value/*.parquet
!Code/processed_player_value/*.npz
!Code/processed_player_value/.gitkeep

## Notebook junk
*.ipynb_checkpoints

## Python cache
__pycache__/


```

Conversion complete.

**Results:**
- `nat_nn_tabular_dataset.csv`: 135.79 MB → 3.33 MB (97.5% reduction)
- `nat_meta.csv`: 4.95 MB → 0.57 MB (88.5% reduction)

**What I did:**
1. Created `convert_csv_to_parquet.py` to convert CSV to Parquet
2. Ran the conversion — Parquet files are now in `processed_player_value/`
3. Updated `.gitignore` to exclude CSV files but keep Parquet files

**Next steps:**
- CSV files stay locally (ignored by git)
- Parquet files are small enough for GitHub
- You can commit the Parquet files

**To use Parquet files in Python:**
```python
df = pd.read_parquet('processed_player_value/nat_nn_tabular_dataset.parquet')
```

Parquet is faster to read/write than CSV and much smaller. The main dataset went from 135 MB to 3.3 MB.

```python
df = pd.read_parquet('processed_player_value/nat_nn_tabular_dataset.parquet')
```

---

**User**

Good day now I wanna change the name of @PROJECT 3/Code/processed_player_value/ to "Data_Processed" and with this pls also update the path references then used in the scripts etc in @PROJECT 3/Code/Implementations/ ! pls fix

---

**Cursor**

Searching for references to `processed_player_value` in the Implementations folder, then renaming the directory and updating all scripts.


Renaming the directory and updating all references:


Updating all file references:


```python
# convert_csv_to_parquet.py
# Convert large CSV files to Parquet format (much smaller and faster)

import pandas as pd
from pathlib import Path

# ----------------------------
# Project paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "Data_Processed"

# Files to convert (with nat_ prefix)
FILES_TO_CONVERT = [
    "nat_nn_tabular_dataset.csv",
    "nat_meta.csv",
]

print("Converting CSV files to Parquet format...")
print(f"Output directory: {OUT_DIR}\n")

for csv_file in FILES_TO_CONVERT:
    csv_path = OUT_DIR / csv_file
    
    if not csv_path.exists():
        print(f"[SKIP] Skipping {csv_file} (not found)")
        continue
    
    print(f"[READ] Reading {csv_file}...")
    df = pd.read_csv(csv_path)
    print(f"       Shape: {df.shape}")
    
    # Get file size
    csv_size = csv_path.stat().st_size / (1024 * 1024)  # MB
    print(f"       CSV size: {csv_size:.2f} MB")
    
    # Convert to parquet
    parquet_file = csv_file.replace(".csv", ".parquet")
    parquet_path = OUT_DIR / parquet_file
    
    print(f"[SAVE] Saving to {parquet_file}...")
    df.to_parquet(parquet_path, compression='snappy', index=False)
    
    # Get parquet file size
    parquet_size = parquet_path.stat().st_size / (1024 * 1024)  # MB
    reduction = (1 - parquet_size / csv_size) * 100
    
    print(f"       Parquet size: {parquet_size:.2f} MB")
    print(f"       Size reduction: {reduction:.1f}%")
    print()

print("[DONE] Conversion complete!")
print(f"\nParquet files saved to: {OUT_DIR}")
print("You can now add CSV files to .gitignore and commit Parquet files instead.")


```

Updating the notebooks:


Updating the notebooks:


```plaintext
DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = PROJECT_ROOT / "processed_player_value"
OUT_DIR.mkdir(parents=True, exist_ok=True)

->

DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = PROJECT_ROOT / "Data_Processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

```plaintext
DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = PROJECT_ROOT / "processed_player_value"
OUT_DIR.mkdir(parents=True, exist_ok=True)

->

DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = PROJECT_ROOT / "Data_Processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

```plaintext
DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = PROJECT_ROOT / "processed_player_value"
OUT_DIR.mkdir(parents=True, exist_ok=True)

->

DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = PROJECT_ROOT / "Data_Processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

Updating the R files and rnn notebook:


```r
############################################################
# Transfermarkt – Nationality Signal (Refined EDA)
#
# Outputs:
#   1) Boxplot: Top 10 vs Bottom 5 countries by median MV
#   2) Facet grid: Median MV lifecycle by nationality
#   3) Facet grid: Max MV lifecycle by nationality
#
# Output size: 12 x 6
############################################################

# ----------------------------
# Libraries
# ----------------------------
library(tidyverse)
library(lubridate)
library(scales)

# ----------------------------
# Paths
# ----------------------------
DATA_DIR <- "../Data"
PLOT_DIR <- "../Plots"

# PLOT_DIR  <- "../Plots"
# DATA_PATH <- "../Data_Processed/cumlag_nn_tabular_dataset.csv"

# ----------------------------
# Load data
# ----------------------------
players <- read_csv(file.path(DATA_DIR, "players.csv"), show_col_types = FALSE)
values  <- read_csv(file.path(DATA_DIR, "player_valuations.csv"), show_col_types = FALSE)

# ----------------------------
# Merge & prepare data
# ----------------------------
df <- values %>%
  inner_join(players, by = "player_id", suffix = c("_val", "_pl")) %>%
  filter(!is.na(position)) %>%
  mutate(
    log_mv = log10(market_value_in_eur_val + 1),
    age = as.numeric(date - as.Date(date_of_birth)) / 365.25
  ) %>%
  filter(
    is.finite(log_mv),
    age >= 16,
    age <= 42
  )

############################################################
# Plot 01: Top 10 vs Bottom 5 countries (Boxplot)
############################################################
country_rank <- df %>%
  group_by(country_of_citizenship) %>%
  summarise(
    median_mv = median(log_mv, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  filter(n >= 300) %>%     # stability
  arrange(desc(median_mv))

top10 <- country_rank %>% slice_head(n = 10)
bot5  <- country_rank %>% slice_tail(n = 5)

countries_01 <- c(
  top10$country_of_citizenship,
  bot5$country_of_citizenship
)

df_01 <- df %>%
  filter(country_of_citizenship %in% countries_01) %>%
  mutate(country = fct_reorder(country_of_citizenship, log_mv, median))

############################################################
# Countries for lifecycle plots (World Cup + peers)
############################################################
countries_grid <- c(
  "Brazil", "Germany", "Italy", "Argentina",
  "France", "Uruguay", "England", "Spain",
  "Netherlands", "Belgium", "Portugal",
  "Croatia", "Denmark", "Turkey", "Ukraine"
)

df_grid <- df %>%
  filter(country_of_citizenship %in% countries_grid) %>%
  mutate(country = fct_reorder(country_of_citizenship, log_mv, median))

# ----------------------------
# Global theme
# ----------------------------
theme_set(
  theme_minimal(base_family = "sans", base_size = 12) +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 11, hjust = 0),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 9),
      strip.text = element_text(size = 9, face = "bold"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      legend.position = "none",
      plot.margin = margin(12, 16, 12, 16)
    )
)

############################################################
# Plot 01 – Boxplot
############################################################
p1 <- ggplot(df_01, aes(country, log_mv, fill = country)) +
  geom_boxplot(outlier.alpha = 0.1, alpha = 0.75) +
  coord_flip() +
  labs(
    title = "Market Value by Country of Citizenship",
    subtitle = "Top 10 vs Bottom 5 countries by median market value",
    x = NULL,
    y = "log10(Market value in EUR + 1)"
  )

ggsave(
  file.path(PLOT_DIR, "01_country_market_value_boxplot.pdf"),
  p1, width = 12, height = 6, device = "pdf"
)

############################################################
# Plot 02 – Median lifecycle
############################################################
p2_data <- df_grid %>%
  mutate(age_int = round(age)) %>%
  group_by(country, age_int) %>%
  summarise(
    value = median(log_mv),
    n_bin = n(),
    .groups = "drop"
  ) %>%
  filter(n_bin >= 25)

p2 <- ggplot(p2_data, aes(age_int, value)) +
  geom_line(linewidth = 0.9) +
  geom_smooth(method = "loess", se = FALSE, span = 0.45,
              linetype = "dashed", linewidth = 0.7) +
  facet_wrap(~country, ncol = 5, scales = "free_y") +
  labs(
    title = "Age–Market Value Lifecycle by Nationality",
    subtitle = "Median log(MV) by age (dashed = loess)",
    x = "Age",
    y = "Median log10(Market value + 1)"
  )

ggsave(
  file.path(PLOT_DIR, "02_country_age_median_grid.pdf"),
  p2, width = 12, height = 6, device = "pdf"
)

############################################################
# Plot 03 – Max lifecycle (Upside envelope)
############################################################
p3_data <- df_grid %>%
  mutate(age_int = round(age)) %>%
  group_by(country, age_int) %>%
  summarise(
    value = max(log_mv),
    n_bin = n(),
    .groups = "drop"
  ) %>%
  filter(n_bin >= 10)   # lower threshold OK for max

p3 <- ggplot(p3_data, aes(age_int, value)) +
  geom_line(linewidth = 0.9, color = "firebrick") +
  geom_smooth(method = "loess", se = FALSE, span = 0.5,
              linetype = "dashed", linewidth = 0.7, color = "grey40") +
  facet_wrap(~country, ncol = 5, scales = "free_y") +
  labs(
    title = "Age–Market Value Upside by Nationality",
    subtitle = "Maximum log(MV) by age (superstar envelope)",
    x = "Age",
    y = "Max log10(Market value + 1)"
  )

ggsave(
  file.path(PLOT_DIR, "03_country_age_max_grid.pdf"),
  p3, width = 12, height = 6, device = "pdf"
)

############################################################
# END
############################################################
cat("Saved 3 nationality-focused plots to:", PLOT_DIR, "\n")

```

```r
############################################################
# Extended Descriptive EDA – Covering All Features (Updated)
# Comprehensive summary statistics, distributions, and relationships
# Adapted for the full cumulative + lagged dataset (278,558 rows)
# Updates: Improved plot 06_lag10_distributions
#   - Included zero values to show full distribution (many events are rare in 10 matches)
#   - Assigned unique colors to each variable for better distinction
#   - Removed overall fill/color; now per variable
#   - Set fixed bandwidth (bw=0.5) for density to reduce waviness from discreteness
#   - Kept histogram with density overlay; scales="free" for varying ranges
#   - Arranged facets in 2 rows for better layout
############################################################

# ----------------------------
# Libraries
# ----------------------------
library(tidyverse)
library(scales)
library(patchwork)
library(corrplot)       # for correlation plot

# ----------------------------
# Paths
# ----------------------------
PLOT_DIR  <- "../Plots"
DATA_PATH <- "../Data_Processed/cumlag_nn_tabular_dataset.csv"

# ----------------------------
# Safety checks
# ----------------------------
dir.create(PLOT_DIR, showWarnings = FALSE, recursive = TRUE)
stopifnot(file.exists(DATA_PATH))

# ----------------------------
# Load & clean data
# ----------------------------
df <- read_csv(DATA_PATH, show_col_types = FALSE) %>%
  filter(
    is.finite(y_log),
    is.finite(age_years),
    is.finite(height_in_cm),
    y_raw > 0
  )

# Add derived columns for plotting
df <- df %>%
  mutate(
    position = case_when(
      pos_ATT ~ "ATT",
      pos_MID ~ "MID",
      pos_DEF ~ "DEF",
      pos_GK ~ "GK",
      TRUE ~ "MISSING"
    ),
    foot = case_when(
      foot_R ~ "Right",
      foot_L ~ "Left",
      foot_B ~ "Both",
      foot_UNK ~ "Unknown",
      TRUE ~ "Unknown"
    ),
    year = year(valuation_date)
  )

# ----------------------------
# Colors
# ----------------------------
COL_BLUE <- "#1f4fd8"
COL_RED <- "#c9332c"
COL_YEL <- "#f2b705"
COL_GRAY <- "grey70"
COL_GREEN <- "#2ca02c"

# ----------------------------
# Global theme
# ----------------------------
theme_set(
  theme_minimal(base_family = "sans", base_size = 12) +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 11, hjust = 0),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 10),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      legend.position = "bottom",
      legend.title = element_blank(),
      plot.margin = margin(12, 16, 12, 16)
    )
)

# ----------------------------
# PDF writer
# ----------------------------
save_pdf <- function(p, name, w = 12, h = 6) {
  ggsave(
    filename = file.path(PLOT_DIR, paste0(name, ".pdf")),
    plot = p,
    width = w,
    height = h,
    device = "pdf"
  )
}

############################################################
# 1. Overall dataset summary
############################################################
# Numerical summary (view in console)
summary(df)

# Unique players & time span
cat("Unique players:", n_distinct(df$player_id), "\n")
cat("Date range:", min(df$valuation_date), "to", max(df$valuation_date), "\n")

############################################################
# 2. Target: Market value distributions (log & raw)
############################################################
p1a <- df %>%
  filter(between(y_log, quantile(y_log, 0.005), quantile(y_log, 0.995))) %>%
  ggplot(aes(y_log)) +
  geom_histogram(aes(y = after_stat(density)), bins = 140, fill = COL_BLUE, alpha = 0.4) +
  geom_density(color = COL_RED, linewidth = 1.2) +
  labs(title = "Distribution of log(Market value)", x = "log(Market value)", y = "Density")

p1b <- df %>%
  filter(y_raw <= quantile(y_raw, 0.99)) %>%
  ggplot(aes(y_raw / 1e6)) +
  geom_histogram(bins = 100, fill = COL_BLUE, alpha = 0.4) +
  scale_x_log10(labels = label_number(prefix = "€", suffix = "M")) +
  labs(title = "Distribution of Market value (raw, log-x)", x = "Market value (€ millions)", y = "Count")

p1 <- p1a / p1b
save_pdf(p1, "01_market_value_distributions", h = 10)

############################################################
# 3. Age & Height
############################################################
p3a <- df %>%
  ggplot(aes(age_years)) +
  geom_density(fill = COL_BLUE, alpha = 0.6, adjust = 1.5) +
  labs(title = "Age distribution", x = "Age (years)", y = "Density")

p3b <- df %>%
  filter(height_in_cm >= 150) %>%
  ggplot(aes(height_in_cm)) +
  geom_density(fill = COL_RED, alpha = 0.6, adjust = 1.5) +
  labs(title = "Height distribution", x = "Height (cm)", y = "Density")

p3 <- p3a + p3b
save_pdf(p3, "02_age_height_dist")

p4 <- df %>%
  mutate(age_bin = cut(age_years, breaks = seq(15, 45, by = 1))) %>%
  group_by(age_bin) %>%
  summarise(age = mean(age_years, na.rm = TRUE),
            y_mean = mean(y_log, na.rm = TRUE),
            h_mean = mean(height_in_cm, na.rm = TRUE),
            .groups = "drop") %>%
  pivot_longer(cols = c(y_mean, h_mean), names_to = "metric") %>%
  ggplot(aes(age, value)) +
  geom_line(color = COL_BLUE, linewidth = 1.1) +
  facet_wrap(~metric, scales = "free_y", 
             labeller = labeller(metric = c(y_mean = "Mean log(MV)", h_mean = "Mean Height (cm)"))) +
  labs(title = "Age trends: Market value & Height")

save_pdf(p4, "03_age_trends")

############################################################
# 4. Categorical features: Position, Foot, Big-5
############################################################
p5_pos <- ggplot(df, aes(position, y_log, fill = position)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.1) +
  scale_fill_manual(values = c("ATT" = COL_RED, "MID" = COL_BLUE, "DEF" = COL_GRAY, "GK" = "black", "MISSING" = "grey85")) +
  labs(title = "Market value by Position", x = NULL, y = "log(Market value)")

p5_foot <- ggplot(df, aes(foot, y_log, fill = foot)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.1) +
  scale_fill_manual(values = c("Right" = COL_BLUE, "Left" = COL_RED, "Both" = COL_GRAY, "Unknown" = "grey85")) +
  labs(title = "Market value by Preferred Foot", x = NULL, y = "log(Market value)")

p5_big5 <- ggplot(df, aes(factor(is_big5_league, labels = c("Non Big-5", "Big-5")), y_log)) +
  geom_boxplot(fill = COL_GREEN, alpha = 0.75) +
  labs(title = "Big-5 League Premium", x = NULL, y = "log(Market value)")

p5 <- (p5_pos | p5_foot) / p5_big5
save_pdf(p5, "04_categorical_mv", h = 10)

############################################################
# 5. Cumulative career stats distributions
############################################################
cum_vars <- c("cumulative_goals", "cumulative_assists", 
              "cumulative_yellow_cards", "cumulative_red_cards",
              "cumulative_sub_in", "cumulative_sub_out")

p_cum <- df %>%
  select(all_of(cum_vars)) %>%
  pivot_longer(everything()) %>%
  filter(value > 0) %>%
  ggplot(aes(log1p(value), fill = name)) +
  geom_density(alpha = 0.4, adjust = 2) +
  facet_wrap(~name, scales = "free_y") +
  labs(title = "Cumulative career stats (log1p scale)", x = "log(1 + count)", y = "Density")

save_pdf(p_cum, "05_cumulative_distributions", w = 14, h = 10)

############################################################
# 6. Lag-10 recent form distributions (Improved)
############################################################
lag_vars <- c("lag_10_goals", "lag_10_assists", 
              "lag_10_yellow_cards", "lag_10_red_cards",
              "lag_10_sub_in", "lag_10_sub_out")

colors <- c(
  "lag_10_goals" = COL_RED,
  "lag_10_assists" = COL_BLUE,
  "lag_10_yellow_cards" = COL_YEL,
  "lag_10_red_cards" = "#8B0000",  # darkred
  "lag_10_sub_in" = COL_GREEN,
  "lag_10_sub_out" = COL_GRAY
)

p_lag <- df %>%
  select(all_of(lag_vars)) %>%
  pivot_longer(everything()) %>%
  # Removed filter(value > 0) to include zeros
  ggplot(aes(value, fill = name, color = name)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 1, alpha = 0.4, position = "identity") +
  geom_density(linewidth = 1.2, bw = 0.5) +  # Fixed bw to reduce waviness
  scale_fill_manual(values = colors) +
  scale_color_manual(values = colors) +
  facet_wrap(~name, scales = "free", nrow = 2) +  # 2 rows for better layout
  labs(title = "Recent form (lag-10 matches, original scale)", x = "Count", y = "Density")

save_pdf(p_lag, "06_lag10_distributions", w = 14, h = 10)

############################################################
# 7. Correlations with log(Market value)
############################################################
cor_df <- df %>%
  select(y_log, age_years, height_in_cm, is_big5_league,
         starts_with("cumulative_"), starts_with("lag_10_")) %>%
  select(where(is.numeric))

cor_mat <- cor(cor_df, use = "pairwise.complete.obs")

pdf(file.path(PLOT_DIR, "07_correlation_matrix.pdf"), width = 14, height = 12)
corrplot(cor_mat, method = "color", type = "upper", 
         tl.cex = 0.8, tl.col = "black",
         order = "hclust", addCoef.col = "black", number.cex = 0.7)
dev.off()

############################################################
# 8. Additional: Market value over time
############################################################
p8 <- df %>%
  group_by(year) %>%
  summarise(mean_y = mean(y_log), .groups = "drop") %>%
  ggplot(aes(year, mean_y)) +
  geom_line(color = COL_BLUE, linewidth = 1.2) +
  geom_point(color = COL_RED) +
  geom_smooth(method = "loess", se = FALSE, color = COL_GRAY, linetype = "dashed") +
  labs(title = "Average log(Market value) over time", x = "Year", y = "Mean log(MV)")

save_pdf(p8, "08_mv_over_time")

############################################################
# END – All major features now covered visually & statistically
############################################################
cat("All plots saved to", PLOT_DIR, "\n")
```

```plaintext
data_dir = Path("../processed_player_value")

->

data_dir = Path("../Data_Processed")
```

```plaintext
data_dir = (nb_dir / ".." / "processed_player_value").resolve()

->

data_dir = (nb_dir / ".." / "Data_Processed").resolve()
```

```plaintext
meta = pd.read_csv("../processed_player_value/meta.csv")

->

meta = pd.read_csv("../Data_Processed/meta.csv")
```

Updating `.gitignore`:


```ignore
## Transfermarkt raw exports
Code/Data/*
!Code/Data/.gitkeep
!Code/Data/pl_team_features.csv
!Code/Data/pl_team_features.parquet
!Code/Data/game_features.parquet
# CSV is typically much larger, comment out if you need it
# !Code/Data/game_features.csv
!Code/Data/predictive_features.txt
!Code/Data/README.md

## Large processed datasets (use Parquet instead)
Code/Data_Processed/*.csv
!Code/Data_Processed/*.parquet
!Code/Data_Processed/*.npz
!Code/Data_Processed/.gitkeep

## Notebook junk
*.ipynb_checkpoints

## Python cache
__pycache__/


```

Verifying all references are updated:


Remaining references are only in notebook output cells (from previous runs), not in code. All code references have been updated.

## Summary of changes:

1. **Renamed directory**: `processed_player_value` → `Data_Processed`
2. **Updated Python scripts**:
   - `convert_csv_to_parquet.py`
3. **Updated Jupyter notebooks**:
   - `data_agg_nationality.ipynb`
   - `data_agg_cumlag.ipynb`
   - `data_aggregating.ipynb`
   - `rnn.ipynb` (3 cells updated)
4. **Updated R scripts**:
   - `descriptive_nationality.R`
   - `descriptive_stats.R`
5. **Updated `.gitignore`**: Changed paths to `Data_Processed`

All code references now point to `Data_Processed`. The remaining matches in the grep results are only in old notebook output cells (historical output), which don't affect functionality.

The directory has been renamed and all scripts should work with the new path.

---

**User**

Also there might be some references in @PROJECT 3/Code/Notebooks/ so lets make sure to update these too! 

---

**Cursor**

Searching for references to `processed_player_value` in the Notebooks folder:


Updating all references in the Notebooks folder:


Searching for the exact context of these references:


Updating all four notebooks:


Updating the notebooks directly:


Updating all four notebooks:


```plaintext
df = pd.read_parquet("../processed_player_value/nat_nn_tabular_dataset.parquet")

->

df = pd.read_parquet("../Data_Processed/nat_nn_tabular_dataset.parquet")
```

```plaintext
df = pd.read_csv("../processed_player_value/nn_tabular_dataset.csv")

->

df = pd.read_csv("../Data_Processed/nn_tabular_dataset.csv")
```

```plaintext
df = pd.read_csv("../processed_player_value/nn_tabular_dataset.csv")

->

df = pd.read_csv("../Data_Processed/nn_tabular_dataset.csv")
```

```plaintext
df = pd.read_csv("../processed_player_value/nn_tabular_dataset.csv")

->

df = pd.read_csv("../Data_Processed/nn_tabular_dataset.csv")
```

Reading the notebooks to find the correct cells:


Using search_replace to update the JSON directly:


Reading the notebook structure to find the correct cell indices:


Updating the notebooks with the correct cell indices:


```plaintext
import pandas as pd

df = pd.read_parquet("../processed_player_value/nat_nn_tabular_dataset.parquet")
print(df.shape)
df.head()

->

import pandas as pd

df = pd.read_parquet("../Data_Processed/nat_nn_tabular_dataset.parquet")
print(df.shape)
df.head()
```

Updating the remaining notebooks:


```plaintext
import pandas as pd

df = pd.read_csv("../processed_player_value/nn_tabular_dataset.csv")

->

import pandas as pd

df = pd.read_csv("../Data_Processed/nn_tabular_dataset.csv")
```

```plaintext
df = pd.read_csv("../processed_player_value/nn_tabular_dataset.csv")

->

df = pd.read_csv("../Data_Processed/nn_tabular_dataset.csv")
```

Fixing the last one in ridge_analysis:


```plaintext
import pandas as pd

df = pd.read_csv("../processed_player_value/nn_tabular_dataset.csv")

data = prepare_data(df, test_size=0.2, seed=42)

->

import pandas as pd

df = pd.read_csv("../Data_Processed/nn_tabular_dataset.csv")

data = prepare_data(df, test_size=0.2, seed=42)
```

Trying with more context:


```plaintext
import pandas as pd

df = pd.read_csv("../processed_player_value/nn_tabular_dataset.csv")

data = prepare_data(df, test_size=0.2, seed=42)
print(data["summary"])

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["ylog_train"].ravel()
y_test  = data["ylog_test"].ravel()

feature_cols = data["feature_cols"]
groups_full = data["groups_full"]

->

import pandas as pd

df = pd.read_csv("../Data_Processed/nn_tabular_dataset.csv")

data = prepare_data(df, test_size=0.2, seed=42)
print(data["summary"])

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["ylog_train"].ravel()
y_test  = data["ylog_test"].ravel()

feature_cols = data["feature_cols"]
groups_full = data["groups_full"]
```

Reading the exact cell content:


Updating just that line:


```plaintext
df = pd.read_csv("../processed_player_value/nn_tabular_dataset.csv")

->

df = pd.read_csv("../Data_Processed/nn_tabular_dataset.csv")
```

Checking the cell structure:


Updating cell index 4:


```plaintext
df = pd.read_csv("../processed_player_value/nn_tabular_dataset.csv")

->

df = pd.read_csv("../Data_Processed/nn_tabular_dataset.csv")
```

All references in the Notebooks folder are updated. Verifying all references across the Code directory:


All code references have been updated. The remaining matches are only in old notebook output cells (historical output from previous runs), which don't affect functionality.

## Summary of updates:

**Updated in Notebooks folder:**
1. `NN_analysis_nationality.ipynb` - Updated parquet path
2. `test_NN.ipynb` - Updated CSV path
3. `ridge_analysis.ipynb` - Updated CSV path
4. `NN_analysis.ipynb` - Updated CSV path

**Previously updated:**
- All scripts in `Implementations/` folder
- All R scripts
- `.gitignore` file

All active code now uses `Data_Processed` instead of `processed_player_value`. The remaining references are only in old output cells from previous notebook executions and don't affect functionality.

---

**User**

Perfect now also lets update the readme in root @README.md and in project 3 @PROJECT 3/README.md to reflect the changes and the theme of the project 3 so only minor updates in root and some bit more comprehensive in project 3 @PROJECT 3/README.md and @PROJECT 3/Code/README.md and also @PROJECT 3/Code/Data/README.md 

---

**Cursor**

Updating the README files to reflect the project focus and recent changes. Reviewing the current files:


Updating the README files:


```markdown
# FYS-STK3155/FYS-STK4155: Applied Data Analysis and Machine Learning

**Course:** Applied Data Analysis and Machine Learning (FYS-STK3155 / FYS-STK4155)  
**Institution:** University of Oslo  
**Academic Year:** 2024-2025

This repository contains implementations, analyses, and reports for the three main projects in the Data Analysis and Machine Learning course. The projects progressively build from fundamental linear regression methods to advanced neural network architectures, providing hands-on experience with both theoretical foundations and practical implementations.

## Repository Overview

This repository is organized into three main projects, each focusing on different aspects of machine learning:

- **PROJECT 1**: Linear Regression and Gradient Descent Methods
- **PROJECT 2**: Feed-Forward Neural Networks
- **PROJECT 3**: Advanced Topics (in progress)

Each project includes complete implementations, comprehensive analyses, Jupyter notebooks, generated visualizations, and detailed reports.

## Project Structure

```
FYSSTK3155/
├── PROJECT 1/              # Linear Regression and Optimization
│   ├── Code/               # Implementation and analysis code
│   ├── README.md           # Project 1 documentation
│   └── FYS_STK3155_PROJECT_1_The_Great_Regression.pdf
│
├── PROJECT 2/              # Neural Networks
│   ├── Code/               # Neural network implementation
│   ├── README.md           # Project 2 documentation
│   └── project2.ipynb      # Project assignment
│
├── PROJECT 3/              # Player Value Prediction
│   ├── Code/               # Data processing and analysis
│   └── README.md           # Project 3 documentation
│
└── README.md               # This file
```

## PROJECT 1: Linear Regression and Gradient Descent

**Focus:** Fundamental regression methods, optimization algorithms, and model evaluation techniques.

### Objectives

- Implement and analyze Ordinary Least Squares (OLS), Ridge, and LASSO regression
- Develop gradient descent optimization algorithms from scratch
- Study bias-variance tradeoff using resampling methods
- Compare closed-form solutions with iterative optimization methods

### Key Components

**Core Implementations:**
- `OLS.py`: Closed-form OLS parameter estimation
- `Ridge.py`: Closed-form Ridge regression with L2 regularization
- `gradient_descent.py`: Standard gradient descent for OLS, Ridge, and LASSO
- `stochastic_gradient_descent.py`: Stochastic variants with advanced optimizers (Momentum, Adagrad, RMSprop, Adam)
- `polynomial_features.py`: Polynomial feature matrix generation
- `prepare_data.py`: Data preparation and Runge function definition

**Analysis Notebooks:**
- `Exa.ipynb`: OLS regression analysis on Runge function
- `Exb.ipynb`: Ridge regression with hyperparameter tuning
- `Exc.ipynb`: Validation of gradient descent implementations
- `Exd.ipynb`: Advanced gradient descent methods comparison
- `Exe.ipynb`: LASSO regression implementation and validation
- `Exf.ipynb`: Stochastic gradient descent methods
- `Exg.ipynb`: Bias-variance analysis using bootstrap resampling
- `Exh.ipynb`: Bias-variance analysis using k-fold cross-validation

### Main Results

- Comprehensive comparison of OLS, Ridge, and LASSO regression methods
- Performance analysis of various gradient descent algorithms
- Systematic study of bias-variance tradeoff through resampling techniques
- Hyperparameter optimization for regularization parameters
- Validation against Scikit-Learn implementations

### Dependencies

- NumPy
- Scikit-Learn
- Matplotlib
- Pandas

For detailed information, see [PROJECT 1/README.md](PROJECT%201/README.md).

## PROJECT 2: Feed-Forward Neural Networks

**Focus:** Building neural networks from scratch, implementing backpropagation, and applying to regression and classification problems.

### Objectives

- Implement a complete feed-forward neural network framework from scratch
- Develop the backpropagation algorithm for gradient computation
- Test multiple activation functions (Sigmoid, ReLU, LeakyReLU)
- Implement various optimization algorithms (SGD, RMSprop, Adam)
- Apply neural networks to regression and classification tasks
- Compare neural network performance with traditional methods

### Key Components

**Core Implementations:**
- `neural_network.py`: Main FFNN class with backpropagation algorithm
- `activations.py`: Activation functions (Sigmoid, ReLU, LeakyReLU, Softmax, Linear) with derivatives
- `losses.py`: Loss functions (MSE, Cross-Entropy, BCE) with gradient computations
- `optimizers.py`: Optimization algorithms (SGD, Momentum, Adagrad, RMSprop, Adam)
- `prepare_data.py`: Data preparation utilities
- `plot_style.py`: Visualization utilities

**Analysis Notebooks:**
- `Exa.ipynb`: Analytical derivation of cost functions and activation derivatives
- `Exb.ipynb`: Basic neural network implementation for regression on Runge function
- `Exb_with_noise.ipynb`: Regression analysis with noise
- `Exc.ipynb`: Validation against Scikit-Learn and automatic differentiation
- `Exd.ipynb`: Activation function and network architecture analysis
- `Exe.ipynb`: L1 and L2 regularization analysis
- `Exf.ipynb`: MNIST classification using neural networks

**Testing:**
- Comprehensive unit tests for all components
- Gradient verification using automatic differentiation
- Validation against standard ML libraries

### Main Results

- Successful implementation of neural networks matching library performance
- Comparison of neural networks with OLS, Ridge, and Lasso regression
- Analysis of activation function impact on training dynamics
- Hyperparameter optimization for network architecture and learning rates
- Classification accuracy on MNIST dataset
- Regularization effects and comparison with traditional methods

### Dependencies

- NumPy
- Matplotlib
- Scikit-Learn

For detailed information, see [PROJECT 2/README.md](PROJECT%202/README.md).

## PROJECT 3: Advanced Topics

**Status:** In progress

This project will explore advanced machine learning topics and applications.

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (for running analysis notebooks)

### Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FYSSTK3155
   ```

2. Install dependencies for a specific project:
   ```bash
   # For PROJECT 1
   cd "PROJECT 1/Code"
   pip install -r requirements.txt

   # For PROJECT 2
   cd "PROJECT 2/Code"
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Navigate to the desired project's `Code/` directory and open the relevant notebook.

### Global Dependencies

If working across multiple projects, install all dependencies:

```bash
pip install numpy scikit-learn matplotlib pandas jupyter
```

## Usage Guidelines

### Running Notebooks

1. Navigate to the project's `Code/` directory
2. Start Jupyter Notebook: `jupyter notebook`
3. Open the desired analysis notebook
4. Ensure all Python modules in the `Code/` directory are accessible (notebooks include path setup)

### Reproducibility

All implementations use fixed random seeds for reproducibility:
- Network weight initialization: seed 6114 (Project 2)
- Data splitting: random_state 42 (standard)
- NumPy random operations: explicitly seeded

To reproduce exact results, ensure you use the same random seeds as specified in the notebooks.

### Code Organization

- **Modular Design**: Each project separates core implementations from analysis notebooks
- **Reusability**: Project 2 reuses optimization code from Project 1
- **Testing**: Project 2 includes comprehensive unit tests
- **Documentation**: Each project has detailed README files

## Learning Outcomes

Through these projects, students gain:

1. **Theoretical Understanding:**
   - Linear regression methods (OLS, Ridge, LASSO)
   - Gradient descent optimization algorithms
   - Neural network architecture and backpropagation
   - Bias-variance tradeoff and model complexity
   - Regularization techniques

2. **Practical Skills:**
   - Implementing machine learning algorithms from scratch
   - Hyperparameter tuning and model selection
   - Data preprocessing and feature engineering
   - Model evaluation and validation techniques
   - Visualization and result interpretation

3. **Technical Competencies:**
   - NumPy for numerical computations
   - Scikit-Learn for comparison and utilities
   - Jupyter Notebooks for interactive analysis
   - Version control and code organization
   - Scientific report writing

## Course Information

**Course Code:** FYS-STK3155 / FYS-STK4155  
**Course Name:** Applied Data Analysis and Machine Learning  
**Institution:** University of Oslo  
**Department:** Physics

**Course Resources:**
- Course website: [Applied Data Analysis and Machine Learning](https://www.uio.no/studier/emner/matnat/fys/FYS-STK3155/index-eng.html)
- Lecture notes: [CompPhysics Machine Learning](https://compphysics.github.io/MachineLearning/)
- Textbook references: See individual project README files

## Contributing

This repository contains coursework submissions. For questions or issues related to the course material, please refer to the course website or contact the course instructors.

## License

This repository contains academic coursework for FYS-STK3155/FYS4155 at the University of Oslo. The code and reports are intended for educational purposes as part of the course requirements.

## Acknowledgments

- Course instructors and teaching assistants at the University of Oslo
- Authors of referenced textbooks and resources (Nielsen, Goodfellow et al., Raschka et al.)
- Scikit-Learn, NumPy, and Matplotlib development teams

## Contact

For questions about this repository or the implementations, please refer to the individual project README files or the course documentation.

---

**Note:** This repository is maintained as part of coursework for FYS-STK3155/FYS4155. All implementations are original work developed for learning and educational purposes.

```

```markdown
# FYS-STK3155/FYS-STK4155: Applied Data Analysis and Machine Learning

**Course:** Applied Data Analysis and Machine Learning (FYS-STK3155 / FYS-STK4155)  
**Institution:** University of Oslo  
**Academic Year:** 2024-2025

This repository contains implementations, analyses, and reports for the three main projects in the Data Analysis and Machine Learning course. The projects progressively build from fundamental linear regression methods to advanced neural network architectures, providing hands-on experience with both theoretical foundations and practical implementations.

## Repository Overview

This repository is organized into three main projects, each focusing on different aspects of machine learning:

- **PROJECT 1**: Linear Regression and Gradient Descent Methods
- **PROJECT 2**: Feed-Forward Neural Networks
- **PROJECT 3**: Advanced Topics (in progress)

Each project includes complete implementations, comprehensive analyses, Jupyter notebooks, generated visualizations, and detailed reports.

## Project Structure

```
FYSSTK3155/
├── PROJECT 1/              # Linear Regression and Optimization
│   ├── Code/               # Implementation and analysis code
│   ├── README.md           # Project 1 documentation
│   └── FYS_STK3155_PROJECT_1_The_Great_Regression.pdf
│
├── PROJECT 2/              # Neural Networks
│   ├── Code/               # Neural network implementation
│   ├── README.md           # Project 2 documentation
│   └── project2.ipynb      # Project assignment
│
├── PROJECT 3/              # Player Value Prediction
│   ├── Code/               # Data processing and analysis
│   └── README.md           # Project 3 documentation
│
└── README.md               # This file
```

## PROJECT 1: Linear Regression and Gradient Descent

**Focus:** Fundamental regression methods, optimization algorithms, and model evaluation techniques.

### Objectives

- Implement and analyze Ordinary Least Squares (OLS), Ridge, and LASSO regression
- Develop gradient descent optimization algorithms from scratch
- Study bias-variance tradeoff using resampling methods
- Compare closed-form solutions with iterative optimization methods

### Key Components

**Core Implementations:**
- `OLS.py`: Closed-form OLS parameter estimation
- `Ridge.py`: Closed-form Ridge regression with L2 regularization
- `gradient_descent.py`: Standard gradient descent for OLS, Ridge, and LASSO
- `stochastic_gradient_descent.py`: Stochastic variants with advanced optimizers (Momentum, Adagrad, RMSprop, Adam)
- `polynomial_features.py`: Polynomial feature matrix generation
- `prepare_data.py`: Data preparation and Runge function definition

**Analysis Notebooks:**
- `Exa.ipynb`: OLS regression analysis on Runge function
- `Exb.ipynb`: Ridge regression with hyperparameter tuning
- `Exc.ipynb`: Validation of gradient descent implementations
- `Exd.ipynb`: Advanced gradient descent methods comparison
- `Exe.ipynb`: LASSO regression implementation and validation
- `Exf.ipynb`: Stochastic gradient descent methods
- `Exg.ipynb`: Bias-variance analysis using bootstrap resampling
- `Exh.ipynb`: Bias-variance analysis using k-fold cross-validation

### Main Results

- Comprehensive comparison of OLS, Ridge, and LASSO regression methods
- Performance analysis of various gradient descent algorithms
- Systematic study of bias-variance tradeoff through resampling techniques
- Hyperparameter optimization for regularization parameters
- Validation against Scikit-Learn implementations

### Dependencies

- NumPy
- Scikit-Learn
- Matplotlib
- Pandas

For detailed information, see [PROJECT 1/README.md](PROJECT%201/README.md).

## PROJECT 2: Feed-Forward Neural Networks

**Focus:** Building neural networks from scratch, implementing backpropagation, and applying to regression and classification problems.

### Objectives

- Implement a complete feed-forward neural network framework from scratch
- Develop the backpropagation algorithm for gradient computation
- Test multiple activation functions (Sigmoid, ReLU, LeakyReLU)
- Implement various optimization algorithms (SGD, RMSprop, Adam)
- Apply neural networks to regression and classification tasks
- Compare neural network performance with traditional methods

### Key Components

**Core Implementations:**
- `neural_network.py`: Main FFNN class with backpropagation algorithm
- `activations.py`: Activation functions (Sigmoid, ReLU, LeakyReLU, Softmax, Linear) with derivatives
- `losses.py`: Loss functions (MSE, Cross-Entropy, BCE) with gradient computations
- `optimizers.py`: Optimization algorithms (SGD, Momentum, Adagrad, RMSprop, Adam)
- `prepare_data.py`: Data preparation utilities
- `plot_style.py`: Visualization utilities

**Analysis Notebooks:**
- `Exa.ipynb`: Analytical derivation of cost functions and activation derivatives
- `Exb.ipynb`: Basic neural network implementation for regression on Runge function
- `Exb_with_noise.ipynb`: Regression analysis with noise
- `Exc.ipynb`: Validation against Scikit-Learn and automatic differentiation
- `Exd.ipynb`: Activation function and network architecture analysis
- `Exe.ipynb`: L1 and L2 regularization analysis
- `Exf.ipynb`: MNIST classification using neural networks

**Testing:**
- Comprehensive unit tests for all components
- Gradient verification using automatic differentiation
- Validation against standard ML libraries

### Main Results

- Successful implementation of neural networks matching library performance
- Comparison of neural networks with OLS, Ridge, and Lasso regression
- Analysis of activation function impact on training dynamics
- Hyperparameter optimization for network architecture and learning rates
- Classification accuracy on MNIST dataset
- Regularization effects and comparison with traditional methods

### Dependencies

- NumPy
- Matplotlib
- Scikit-Learn

For detailed information, see [PROJECT 2/README.md](PROJECT%202/README.md).

## PROJECT 3: Player Value Prediction

**Focus:** Time-series modeling and feature engineering for predicting football player market values using Transfermarkt data.

### Objectives

- Process and aggregate football player data from Transfermarkt
- Engineer temporal features (cumulative statistics, lag features)
- Implement RNN models for sequential player performance data
- Compare neural network architectures (MLP, RNN) with traditional regression methods
- Analyze the impact of player nationality and other static features on market value

### Key Components

**Data Processing:**
- `data_aggregating.ipynb`: Basic feature extraction and dataset creation
- `data_agg_cumlag.ipynb`: Cumulative and lag feature engineering
- `data_agg_nationality.ipynb`: Nationality-enhanced feature set with one-hot encoding
- `convert_csv_to_parquet.py`: Utility to compress large CSV files to Parquet format

**Analysis Notebooks:**
- `ridge_analysis.ipynb`: Ridge regression baseline models
- `NN_analysis.ipynb`: Neural network analysis on basic feature set
- `NN_analysis_nationality.ipynb`: Neural network analysis with nationality features
- `rnn.ipynb`: RNN implementation for sequential player data

**Data Structure:**
- Raw data: `Code/Data/` (players, valuations, game events)
- Processed data: `Code/Data_Processed/` (tabular datasets, RNN sequences, metadata)

### Main Results

- Multiple feature engineering approaches (basic, cumulative, nationality-enhanced)
- Comparison of Ridge regression, MLP, and RNN models
- Analysis of feature importance and model performance
- Temporal feature engineering for player value prediction

### Dependencies

- NumPy, Pandas
- PyTorch (for RNN models)
- Scikit-Learn
- Matplotlib, Seaborn

For detailed information, see [PROJECT 3/README.md](PROJECT%203/README.md).

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (for running analysis notebooks)

### Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FYSSTK3155
   ```

2. Install dependencies for a specific project:
   ```bash
   # For PROJECT 1
   cd "PROJECT 1/Code"
   pip install -r requirements.txt

   # For PROJECT 2
   cd "PROJECT 2/Code"
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Navigate to the desired project's `Code/` directory and open the relevant notebook.

### Global Dependencies

If working across multiple projects, install all dependencies:

```bash
pip install numpy scikit-learn matplotlib pandas jupyter
```

## Usage Guidelines

### Running Notebooks

1. Navigate to the project's `Code/` directory
2. Start Jupyter Notebook: `jupyter notebook`
3. Open the desired analysis notebook
4. Ensure all Python modules in the `Code/` directory are accessible (notebooks include path setup)

### Reproducibility

All implementations use fixed random seeds for reproducibility:
- Network weight initialization: seed 6114 (Project 2)
- Data splitting: random_state 42 (standard)
- NumPy random operations: explicitly seeded

To reproduce exact results, ensure you use the same random seeds as specified in the notebooks.

### Code Organization

- **Modular Design**: Each project separates core implementations from analysis notebooks
- **Reusability**: Project 2 reuses optimization code from Project 1
- **Testing**: Project 2 includes comprehensive unit tests
- **Documentation**: Each project has detailed README files

## Learning Outcomes

Through these projects, students gain:

1. **Theoretical Understanding:**
   - Linear regression methods (OLS, Ridge, LASSO)
   - Gradient descent optimization algorithms
   - Neural network architecture and backpropagation
   - Bias-variance tradeoff and model complexity
   - Regularization techniques

2. **Practical Skills:**
   - Implementing machine learning algorithms from scratch
   - Hyperparameter tuning and model selection
   - Data preprocessing and feature engineering
   - Model evaluation and validation techniques
   - Visualization and result interpretation

3. **Technical Competencies:**
   - NumPy for numerical computations
   - Scikit-Learn for comparison and utilities
   - Jupyter Notebooks for interactive analysis
   - Version control and code organization
   - Scientific report writing

## Course Information

**Course Code:** FYS-STK3155 / FYS-STK4155  
**Course Name:** Applied Data Analysis and Machine Learning  
**Institution:** University of Oslo  
**Department:** Physics

**Course Resources:**
- Course website: [Applied Data Analysis and Machine Learning](https://www.uio.no/studier/emner/matnat/fys/FYS-STK3155/index-eng.html)
- Lecture notes: [CompPhysics Machine Learning](https://compphysics.github.io/MachineLearning/)
- Textbook references: See individual project README files

## Contributing

This repository contains coursework submissions. For questions or issues related to the course material, please refer to the course website or contact the course instructors.

## License

This repository contains academic coursework for FYS-STK3155/FYS4155 at the University of Oslo. The code and reports are intended for educational purposes as part of the course requirements.

## Acknowledgments

- Course instructors and teaching assistants at the University of Oslo
- Authors of referenced textbooks and resources (Nielsen, Goodfellow et al., Raschka et al.)
- Scikit-Learn, NumPy, and Matplotlib development teams

## Contact

For questions about this repository or the implementations, please refer to the individual project README files or the course documentation.

---

**Note:** This repository is maintained as part of coursework for FYS-STK3155/FYS4155. All implementations are original work developed for learning and educational purposes.

```

```markdown
# PROJECT 3: Player Value Prediction

This project focuses on predicting football player market values using machine learning models applied to Transfermarkt data. The project involves comprehensive data processing, feature engineering, and model comparison across multiple approaches including Ridge regression, feed-forward neural networks (MLP), and recurrent neural networks (RNN).

## Project Overview

The goal is to predict player market values (in euros) using historical performance data, player characteristics, and temporal patterns. The project explores different feature engineering strategies and model architectures to understand which factors most influence player valuations.

## Repository Structure

```
PROJECT 3/
├── Code/
│   ├── Data/                    # Raw Transfermarkt CSV files
│   │   ├── players.csv         # Player biographical data
│   │   ├── player_valuations.csv  # Historical market values
│   │   ├── game_events.csv     # Match events (goals, cards, etc.)
│   │   └── README.md           # Data documentation
│   │
│   ├── Data_Processed/          # Processed datasets (output)
│   │   ├── nn_tabular_dataset.csv      # Basic feature set
│   │   ├── cumlag_nn_tabular_dataset.csv  # Cumulative + lag features
│   │   ├── nat_nn_tabular_dataset.parquet  # Nationality-enhanced (Parquet)
│   │   ├── rnn_dataset.npz     # RNN sequences (basic)
│   │   ├── cumlag_rnn_dataset.npz  # RNN sequences (cumulative)
│   │   └── nat_rnn_dataset.npz     # RNN sequences (nationality)
│   │
│   ├── Implementations/        # Data processing scripts
│   │   ├── data_aggregating.ipynb      # Basic feature extraction
│   │   ├── data_agg_cumlag.ipynb       # Cumulative + lag features
│   │   ├── data_agg_nationality.ipynb  # Nationality features
│   │   ├── rnn.ipynb           # RNN model implementation
│   │   ├── prepare_data.py     # Data preparation utilities
│   │   └── convert_csv_to_parquet.py   # CSV to Parquet converter
│   │
│   ├── Notebooks/              # Analysis and modeling notebooks
│   │   ├── ridge_analysis.ipynb       # Ridge regression baseline
│   │   ├── NN_analysis.ipynb          # MLP neural network
│   │   ├── NN_analysis_nationality.ipynb  # MLP with nationality
│   │   └── test_NN.ipynb       # Neural network testing
│   │
│   ├── Plots/                  # Generated visualizations
│   ├── Tables/                 # Result tables
│   └── README.md               # Code documentation
│
└── README.md                    # This file
```

## Data Sources

The project uses the **Football Data from Transfermarkt** dataset available on Kaggle:

🔗 **https://www.kaggle.com/datasets/davidcariboo/player-scores**

### Required Data Files

Place the following CSV files in `Code/Data/`:

1. **players.csv** - Player biographical information
   - `player_id`, `date_of_birth`, `height_in_cm`, `foot`, `position`, `country_of_citizenship`

2. **player_valuations.csv** - Historical market values
   - `player_id`, `date`, `market_value_in_eur`, `player_club_domestic_competition_id`

3. **game_events.csv** - Match events
   - `player_id`, `game_id`, `date`, `type`, `description`, `minute`

## Feature Engineering

The project implements three progressively more complex feature sets:

### 1. Basic Features (`data_aggregating.ipynb`)
- Static features: height, age, position (one-hot), foot preference (one-hot), Big-5 league flag
- Per-game features: goals, assists, yellow cards, red cards, substitutions
- Aggregated features: mean and sum over last 20 games

### 2. Cumulative + Lag Features (`data_agg_cumlag.ipynb`)
- All basic features
- Cumulative statistics: total goals, assists, cards, substitutions up to valuation date
- Lag features: sum of events over last 10 games before valuation

### 3. Nationality-Enhanced Features (`data_agg_nationality.ipynb`)
- All cumulative + lag features
- Nationality one-hot encoding: 184 country features (1/0 encoding)
- All categorical features use 1/0 encoding instead of boolean

## Data Processing Workflow

### Step 1: Run Data Aggregation Scripts

Choose the feature set you want:

```bash
# Basic features
jupyter notebook Code/Implementations/data_aggregating.ipynb

# Cumulative + lag features
jupyter notebook Code/Implementations/data_agg_cumlag.ipynb

# Nationality-enhanced features
jupyter notebook Code/Implementations/data_agg_nationality.ipynb
```

These scripts generate:
- Tabular datasets (CSV) for traditional ML models
- RNN sequence datasets (NPZ) for recurrent models
- Metadata files (CSV) with player IDs and dates

### Step 2: Convert Large CSVs to Parquet (Optional)

For datasets exceeding 100MB, convert to Parquet format for Git:

```bash
python Code/Implementations/convert_csv_to_parquet.py
```

This creates compressed Parquet files that are much smaller (typically 95%+ reduction).

### Step 3: Run Analysis Notebooks

```bash
# Ridge regression baseline
jupyter notebook Code/Notebooks/ridge_analysis.ipynb

# Neural network analysis
jupyter notebook Code/Notebooks/NN_analysis.ipynb

# Neural network with nationality features
jupyter notebook Code/Notebooks/NN_analysis_nationality.ipynb
```

## Output Files

All processed datasets are saved to `Code/Data_Processed/`:

**Tabular Datasets:**
- `nn_tabular_dataset.csv` - Basic features (28 columns)
- `cumlag_nn_tabular_dataset.csv` - Cumulative + lag (28 columns)
- `nat_nn_tabular_dataset.csv` - With nationality (212 columns)
- `nat_nn_tabular_dataset.parquet` - Compressed version (3.3 MB vs 136 MB)

**RNN Datasets:**
- `rnn_dataset.npz` - Sequences for basic features
- `cumlag_rnn_dataset.npz` - Sequences for cumulative features
- `nat_rnn_dataset.npz` - Sequences with nationality features

**Metadata:**
- `meta.csv`, `cumlag_meta.csv`, `nat_meta.csv` - Player IDs and valuation dates

## Model Architectures

### Ridge Regression
- Baseline linear model with L2 regularization
- Hyperparameter tuning for alpha
- Group-based train/test split (by player_id) to prevent data leakage

### Feed-Forward Neural Network (MLP)
- Multi-layer perceptron with dropout
- Architecture: Input → 128 → 64 → 1
- Activation: ReLU with dropout (0.1)
- Optimizer: Adam with learning rate 5e-4

### Recurrent Neural Network (RNN)
- GRU-based architecture for sequential data
- Sequence length: 20 games
- Static features concatenated with hidden state
- Handles variable-length sequences with padding

## Key Features

- **Temporal Feature Engineering**: Cumulative statistics and lag features capture player performance trends
- **Nationality Analysis**: 184 country features explore geographic market value patterns
- **Data Leakage Prevention**: Group-based splitting ensures no player appears in both train and test sets
- **Multiple Model Comparison**: Ridge, MLP, and RNN models compared on same datasets
- **Efficient Storage**: Parquet format for large datasets (97% size reduction)

## Dependencies

```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn jupyter pyarrow
```

## Notes

- Large CSV files (>100MB) are gitignored; use Parquet format for version control
- All scripts use fixed random seeds for reproducibility
- Data processing can take 10-20 minutes depending on dataset size
- RNN models require more memory and computation time than tabular models

## Results and Analysis

See the analysis notebooks in `Code/Notebooks/` for:
- Model performance comparisons
- Feature importance analysis
- Hyperparameter optimization results
- Visualization of predictions and residuals

---

For detailed code documentation, see [Code/README.md](Code/README.md).

```

```markdown
# PROJECT 3 Code Directory

This directory contains all code for the Player Value Prediction project, including data processing scripts, model implementations, and analysis notebooks.

## Directory Structure

```
Code/
├── Data/                    # Raw input data (see Data/README.md)
├── Data_Processed/          # Processed datasets (output)
├── Implementations/         # Data processing and model scripts
├── Notebooks/               # Analysis and modeling notebooks
├── Plots/                   # Generated visualizations
├── Tables/                  # Result tables
└── README.md                # This file
```

## Data Processing Scripts (`Implementations/`)

### Data Aggregation Notebooks

1. **`data_aggregating.ipynb`**
   - Basic feature extraction
   - Creates: `nn_tabular_dataset.csv`, `rnn_dataset.npz`, `meta.csv`
   - Features: static player info, per-game events, aggregated statistics

2. **`data_agg_cumlag.ipynb`**
   - Adds cumulative and lag features
   - Creates: `cumlag_nn_tabular_dataset.csv`, `cumlag_rnn_dataset.npz`, `cumlag_meta.csv`
   - Features: all basic features + cumulative stats + lag_10 features

3. **`data_agg_nationality.ipynb`**
   - Adds nationality one-hot encoding (184 countries)
   - Creates: `nat_nn_tabular_dataset.csv`, `nat_rnn_dataset.npz`, `nat_meta.csv`
   - Features: all cumulative/lag features + nationality (1/0 encoding)
   - Uses 1/0 encoding for all categorical features (foot, position, nationality)

### Utility Scripts

- **`convert_csv_to_parquet.py`**: Converts large CSV files to Parquet format for efficient storage
- **`prepare_data.py`**: Data preparation utilities (train/test splitting, standardization)
- **`rnn.ipynb`**: RNN model implementation for sequential player data

### R Scripts

- **`descriptive_stats.R`**: Statistical analysis and visualizations
- **`descriptive_nationality.R`**: Nationality-specific analysis

## Analysis Notebooks (`Notebooks/`)

### Regression Models

- **`ridge_analysis.ipynb`**: Ridge regression baseline model
  - Hyperparameter tuning
  - Performance evaluation
  - Feature importance

### Neural Network Models

- **`NN_analysis.ipynb`**: Feed-forward neural network (MLP)
  - Basic feature set
  - Architecture exploration
  - Comparison with Ridge regression

- **`NN_analysis_nationality.ipynb`**: MLP with nationality features
  - Extended feature set (212 features)
  - Impact of nationality on predictions
  - Performance comparison

- **`test_NN.ipynb`**: Neural network testing and validation

## Data Flow

```
Raw Data (Data/)
    ↓
[Data Processing Scripts]
    ↓
Processed Data (Data_Processed/)
    ↓
[Analysis Notebooks]
    ↓
Results (Plots/, Tables/)
```

## Running the Code

### 1. Data Processing

Run the data aggregation notebooks in order:

```bash
# Start with basic features
jupyter notebook Implementations/data_aggregating.ipynb

# Then add cumulative/lag features
jupyter notebook Implementations/data_agg_cumlag.ipynb

# Finally add nationality features
jupyter notebook Implementations/data_agg_nationality.ipynb
```

### 2. Convert to Parquet (if needed)

For large datasets:

```bash
python Implementations/convert_csv_to_parquet.py
```

### 3. Run Analysis

```bash
# Baseline model
jupyter notebook Notebooks/ridge_analysis.ipynb

# Neural networks
jupyter notebook Notebooks/NN_analysis.ipynb
jupyter notebook Notebooks/NN_analysis_nationality.ipynb
```

## Output Files

All processed datasets are saved to `Data_Processed/`:

**File Naming Convention:**
- `{prefix}nn_tabular_dataset.csv` - Tabular data for ML models
- `{prefix}rnn_dataset.npz` - RNN sequence data
- `{prefix}meta.csv` - Metadata (player_id, valuation_date)

**Prefixes:**
- No prefix: Basic features
- `cumlag_`: Cumulative + lag features
- `nat_`: Nationality-enhanced features

**File Formats:**
- CSV: Human-readable, large file size
- Parquet: Compressed, efficient (97% size reduction)
- NPZ: NumPy compressed format for arrays

## Important Notes

1. **Data Location**: All scripts expect data in `Data/` and output to `Data_Processed/`
2. **Path Handling**: Scripts use `Path.cwd().parent` to work from any directory
3. **Memory**: Large datasets may require significant RAM (8GB+ recommended)
4. **Processing Time**: Data aggregation can take 10-20 minutes
5. **Git**: Large CSV files are gitignored; use Parquet for version control

## Dependencies

See `requirements.txt` for full list. Key packages:
- `pandas`, `numpy` - Data processing
- `torch` - Neural network models
- `scikit-learn` - Traditional ML models
- `matplotlib`, `seaborn` - Visualization
- `pyarrow` - Parquet file support

## Troubleshooting

**Issue**: "File not found" errors
- **Solution**: Ensure you're running from the correct directory or check paths in scripts

**Issue**: Out of memory errors
- **Solution**: Process datasets one at a time, or use Parquet format to reduce memory usage

**Issue**: Slow processing
- **Solution**: Data aggregation is computationally intensive; be patient or optimize scripts

---

For data file documentation, see [Data/README.md](Data/README.md).

```

```markdown
# Data Directory

This directory contains the required CSV files for the player value prediction project.

## Dataset Source

The data comes from the **Football Data from Transfermarkt** dataset on Kaggle:

🔗 **https://www.kaggle.com/datasets/davidcariboo/player-scores**

## Required Files

This directory must contain the following CSV files:

### 1. `players.csv`
Contains player biographical and physical information.

**Key columns used:**
- `player_id` - Unique identifier for each player
- `date_of_birth` - Player's date of birth (used to calculate age)
- `height_in_cm` - Player's height in centimeters
- `foot` - Preferred foot (left/right/both)
- `position` - Player's position (standardized to GK/DEF/MID/ATT)

### 2. `player_valuations.csv`
Contains historical market valuations for players over time.

**Key columns used:**
- `player_id` - Unique identifier for each player
- `date` - Date of the valuation
- `market_value_in_eur` - Player's market value in euros (target variable)
- `player_club_domestic_competition_id` - League/competition ID (used to identify Big-5 leagues: GB1, ES1, IT1, DE1, FR1)

### 3. `game_events.csv`
Contains match events (goals, assists, cards, substitutions) from football games.

**Key columns used:**
- `player_id` - Player who performed the event
- `player_assist_id` - Player who provided the assist (for goals)
- `player_in_id` - Player substituted in
- `game_id` - Unique identifier for the match
- `date` - Date of the match
- `type` - Type of event (Goals, Cards, Substitutions)
- `description` - Event description (used to distinguish yellow/red cards)
- `minute` - Minute when the event occurred

**Event types extracted:**
- Goals (for goals scored)
- Assists (from goal events)
- Yellow cards
- Red cards
- Substitutions (both in and out)

## Setup Instructions

1. Download the dataset from the Kaggle link above
2. Extract the ZIP file
3. Copy the three required CSV files (`players.csv`, `player_valuations.csv`, `game_events.csv`) into this directory
4. Ensure all files are in CSV format and properly formatted

## Data Processing

The data processing scripts in `Code/Implementations/` use this data to:
- Extract static player features (height, position, foot preference, age, nationality)
- Build per-game event features (goals, assists, cards, substitutions)
- Create time-series sequences for RNN models
- Generate aggregated features (cumulative statistics, lag features)
- Prepare training datasets for player value prediction

**Processing Scripts:**
- `data_aggregating.ipynb` - Basic feature extraction
- `data_agg_cumlag.ipynb` - Adds cumulative and lag features
- `data_agg_nationality.ipynb` - Adds nationality one-hot encoding

**Output Location:**
All processed datasets are saved to `Code/Data_Processed/` directory.

## Notes

- All date columns should be parseable as datetime
- Missing values in key columns are handled during processing
- The code expects numeric IDs for players and games
- Market values should be numeric values in euros


```

Updated all README files. Summary:

## Updated Files:

1. **Root `README.md`**
   - Updated Project 3 section with player value prediction focus
   - Added objectives, key components, and dependencies
   - Updated project structure diagram

2. **`PROJECT 3/README.md`**
   - Rewrote to focus on player value prediction
   - Added repository structure, data sources, feature engineering details
   - Included workflow instructions and model architecture descriptions
   - Documented output files and file naming conventions

3. **`PROJECT 3/Code/README.md`**
   - Added documentation for the code directory
   - Documented all scripts and notebooks
   - Included data flow diagram and running instructions
   - Added troubleshooting section

4. **`PROJECT 3/Code/Data/README.md`**
   - Updated data processing section to reference correct scripts
   - Added note about output location (`Data_Processed/`)

All README files now reflect:
- The actual project focus (player value prediction)
- The correct directory name (`Data_Processed` instead of `processed_player_value`)
- All three feature engineering approaches (basic, cumlag, nationality)
- Proper file naming conventions and output locations
- Complete workflow from raw data to final models

The documentation is now consistent and accurately describes the project structure and workflow.

---

**User**

Hi, now all csv files in Data_Processed are gitignored, please unignore them all, but still ignore this one: nat_nn_tabular_dataset

---

**Cursor**

Updating `.gitignore` to unignore all CSV files in `Data_Processed` except `nat_nn_tabular_dataset`:


Updating `.gitignore` to allow all CSV files except `nat_nn_tabular_dataset`:


```ignore
## Transfermarkt raw exports
Code/Data/*
!Code/Data/.gitkeep
!Code/Data/pl_team_features.csv
!Code/Data/pl_team_features.parquet
!Code/Data/game_features.parquet
# CSV is typically much larger, comment out if you need it
# !Code/Data/game_features.csv
!Code/Data/predictive_features.txt
!Code/Data/README.md

## Large processed datasets
# Ignore the very large nationality dataset (use Parquet version instead)
Code/Data_Processed/nat_nn_tabular_dataset.csv
Code/Data_Processed/nat_nn_tabular_dataset.parquet

# Allow all other CSV files and other formats
!Code/Data_Processed/*.csv
!Code/Data_Processed/*.parquet
!Code/Data_Processed/*.npz
!Code/Data_Processed/.gitkeep

## Notebook junk
*.ipynb_checkpoints

## Python cache
__pycache__/


```

Fixing the gitignore order: gitignore patterns are order-dependent. Updating it:


```ignore
## Transfermarkt raw exports
Code/Data/*
!Code/Data/.gitkeep
!Code/Data/pl_team_features.csv
!Code/Data/pl_team_features.parquet
!Code/Data/game_features.parquet
# CSV is typically much larger, comment out if you need it
# !Code/Data/game_features.csv
!Code/Data/predictive_features.txt
!Code/Data/README.md

## Large processed datasets
# Ignore the very large nationality dataset CSV (use Parquet version instead)
Code/Data_Processed/nat_nn_tabular_dataset.csv

# Allow all other files (CSV, Parquet, NPZ)
!Code/Data_Processed/*.csv
!Code/Data_Processed/*.parquet
!Code/Data_Processed/*.npz
!Code/Data_Processed/.gitkeep

## Notebook junk
*.ipynb_checkpoints

## Python cache
__pycache__/


```

Updating `.gitignore`: removing the general CSV ignore and adding a specific ignore for `nat_nn_tabular_dataset.csv`:


```ignore
## Transfermarkt raw exports
Code/Data/*
!Code/Data/.gitkeep
!Code/Data/pl_team_features.csv
!Code/Data/pl_team_features.parquet
!Code/Data/game_features.parquet
# CSV is typically much larger, comment out if you need it
# !Code/Data/game_features.csv
!Code/Data/predictive_features.txt
!Code/Data/README.md

## Large processed datasets
# Ignore the very large nationality dataset CSV (136MB - use Parquet version instead)
Code/Data_Processed/nat_nn_tabular_dataset.csv

# All other CSV, Parquet, and NPZ files are tracked
# (No general ignore pattern - only the specific file above is ignored)

## Notebook junk
*.ipynb_checkpoints

## Python cache
__pycache__/


```

Fixing the gitignore: removing the general CSV ignore and keeping only the specific ignore for `nat_nn_tabular_dataset.csv`:


```ignore
## Large processed datasets
# Ignore the very large nationality dataset CSV (136MB - use Parquet version instead)
Code/Data_Processed/nat_nn_tabular_dataset.csv

# All other CSV, Parquet, and NPZ files in Data_Processed are tracked
# (No general ignore pattern - only the specific file above is ignored)
```

Reading the current state of the file:


The `.gitignore` is configured correctly:

- Only `Code/Data_Processed/nat_nn_tabular_dataset.csv` is ignored
- All other CSV files in `Data_Processed/` are tracked
- All Parquet and NPZ files are tracked

The file now only ignores the specific large nationality dataset CSV (136MB), while all other files in `Data_Processed/` (including other CSV files like `nn_tabular_dataset.csv`, `cumlag_nn_tabular_dataset.csv`, `meta.csv`, etc.) will be tracked by git.

