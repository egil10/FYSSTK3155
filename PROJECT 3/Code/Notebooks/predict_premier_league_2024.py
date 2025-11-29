"""
Premier League 2024 Prediction Script
======================================

This script implements rolling time-series cross-validation to predict 
Premier League 2024 matches matchday by matchday. All matches in each matchday 
(e.g., "1. Matchday", "2. Matchday") are predicted together as a bundle.

IMPORTANT: Real-world temporal logic:
- For Matchday n+1, we use all ACTUAL results up to and including Matchday n for training
- This simulates the real-world scenario where we know all actual results from
  Matchday n before predicting Matchday n+1
- After each matchday, we update the league tables with actual results (if available)
  and use those for subsequent predictions
"""

## 1. Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import Parallel, delayed
import multiprocessing as mp

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Set number of parallel workers (use all available cores minus 1)
n_jobs = max(1, mp.cpu_count() - 1)
print(f"Using {n_jobs} parallel workers for preprocessing")

## 1.5. Speed Optimization Configuration
"""
SPEED OPTIMIZATIONS IMPLEMENTED:
1. Reduced model iterations: max_iter=200 (was 500) with more aggressive early stopping
2. Adaptive learning rate and batch size for faster convergence
3. Optional verbose output: set verbose=False to reduce printing overhead
4. Optional display control: can skip intermediate table displays
5. Optimized data operations: reduced copying, cached datetime conversions
6. Optimized label encoder: cached class-to-index mapping

TO ADJUST SPEED (see main prediction loop):
- Set DISPLAY_EVERY_N_ROUNDS = 5 to show tables every 5th round only
- Set SHOW_INTERMEDIATE_TABLES = False to skip all intermediate displays
- Both settings still save all results to CSV files at the end
"""

## 2. Load Data

# Determine the path to the data file
data_path = Path("../../Data/game_features.parquet")

# Alternative paths if running from different locations
if not data_path.exists():
    data_path = Path("../Data/game_features.parquet")
if not data_path.exists():
    data_path = Path("PROJECT 3/Code/Data/game_features.parquet")
if not data_path.exists():
    import os
    data_path = Path(os.getcwd()) / "PROJECT 3" / "Code" / "Data" / "game_features.parquet"

print(f"Loading data from: {data_path}")
df = pd.read_parquet(data_path)
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# Diagnostic: Show date range and season distribution
df['ID_DATE'] = pd.to_datetime(df['ID_DATE'])
print(f"\nDataset date range: {df['ID_DATE'].min().date()} to {df['ID_DATE'].max().date()}")
if 'ID_SEASON' in df.columns:
    season_counts = df['ID_SEASON'].value_counts().sort_index()
    print(f"Seasons in dataset: {sorted(df['ID_SEASON'].unique().tolist())}")
    print(f"Matches by season:\n{season_counts}")

## 3. Identify 2024 Premier League Rounds

# Filter to 2024 Premier League matches
pl_2024 = df[
    (df['ID_COMPETITION'] == 'GB1') & 
    (df['ID_SEASON'] == 2024)
].copy()

if len(pl_2024) == 0:
    raise ValueError("No 2024 Premier League data found! Check ID_COMPETITION and ID_SEASON columns.")

# Get unique matchdays (by ID_ROUND_NUMBER) - each matchday can span multiple dates
# All matches in the same matchday (ID_ROUND_NUMBER) are predicted together
matchday_info = pl_2024.groupby('ID_ROUND_NUMBER').agg({
    'ID_DATE': ['min', 'max', 'count'],
    'ID_ROUND': 'first'
}).reset_index()
matchday_info.columns = ['ID_ROUND_NUMBER', 'First_Date', 'Last_Date', 'Match_Count', 'Matchday_Name']

# Convert dates
matchday_info['First_Date'] = pd.to_datetime(matchday_info['First_Date'])
matchday_info['Last_Date'] = pd.to_datetime(matchday_info['Last_Date'])

print(f"\n2024 Premier League Schedule:")
print(f"Total unique matchdays: {len(matchday_info)}")
print(f"\nFirst 5 matchdays:")
print(matchday_info.head().to_string(index=False))
print(f"\nLast 5 matchdays:")
print(matchday_info.tail().to_string(index=False))

# Get matchday numbers and their date ranges (first and last date for training cutoff)
# All matches in the same matchday are predicted together as a bundle
matchday_first_dates = {}
matchday_last_dates = {}
for _, row in matchday_info.iterrows():
    matchday_num = int(row['ID_ROUND_NUMBER'])
    matchday_first_dates[matchday_num] = row['First_Date']
    matchday_last_dates[matchday_num] = row['Last_Date']

# Sort by matchday number
matchday_numbers = sorted(matchday_first_dates.keys())
print(f"\nMatchdays to predict: {min(matchday_numbers)} to {max(matchday_numbers)} (total: {len(matchday_numbers)})")
print(f"Note: All matches in each matchday are predicted together as a bundle.")
print(f"      For Matchday n+1, we use all actual results up to and including Matchday n.")

## 4. Prepare Features and Target

# Separate metadata, target, and predictors
metadata_cols = [col for col in df.columns if col.startswith('ID_')]
target_col = 'RESULT' if 'RESULT' in df.columns else None

if target_col is None:
    raise ValueError("RESULT column not found in dataset!")

# Get numeric predictor columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
predictor_cols = [col for col in numeric_cols if col not in metadata_cols and col != target_col]

print(f"\nFeature preparation:")
print(f"  Metadata columns: {len(metadata_cols)}")
print(f"  Target: {target_col}")
print(f"  Predictor features: {len(predictor_cols)}")

# Extract features and target
X_all = df[predictor_cols].copy()
y_all = df[target_col].copy()
dates_all = pd.to_datetime(df['ID_DATE'])

# Filter out predictors with >50% missing values
missing_threshold = 0.5
missing_pct = X_all.isnull().sum() / len(X_all)
valid_cols = missing_pct[missing_pct <= missing_threshold].index.tolist()
X_all = X_all[valid_cols]
predictor_cols = valid_cols

print(f"  Valid predictors after filtering (>50% missing): {len(predictor_cols)}")

# Fill remaining missing values with median
X_all = X_all.fillna(X_all.median())

# Encode target variable
le = LabelEncoder()
y_all_encoded = le.fit_transform(y_all)

print(f"  Target classes: {le.classes_}")
print(f"  Target distribution: {dict(zip(le.classes_, np.bincount(y_all_encoded)))}")

## 5. Create Team ID to Name Mapping

def create_team_mapping(df_all, season=2024, competition='GB1'):
    """Create mapping between team IDs and team names."""
    season_matches = df_all[
        (df_all['ID_COMPETITION'] == competition) & 
        (df_all['ID_SEASON'] == season)
    ]
    
    team_mapping = {}
    
    # Create mapping from home teams
    if 'ID_HOME_CLUB' in season_matches.columns and 'ID_HOME_TEAM' in season_matches.columns:
        home_mapping = season_matches[['ID_HOME_CLUB', 'ID_HOME_TEAM']].dropna().drop_duplicates()
        for _, row in home_mapping.iterrows():
            team_mapping[int(row['ID_HOME_CLUB'])] = row['ID_HOME_TEAM']
    
    # Add away teams (in case some teams only appear as away)
    if 'ID_AWAY_CLUB' in season_matches.columns and 'ID_AWAY_TEAM' in season_matches.columns:
        away_mapping = season_matches[['ID_AWAY_CLUB', 'ID_AWAY_TEAM']].dropna().drop_duplicates()
        for _, row in away_mapping.iterrows():
            team_id = int(row['ID_AWAY_CLUB'])
            if team_id not in team_mapping:
                team_mapping[team_id] = row['ID_AWAY_TEAM']
    
    return team_mapping

## 6. League Table Simulation Functions

def initialize_league_table(df_all, team_mapping, season=2024, competition='GB1'):
    """Initialize league table with all teams in the competition."""
    # Get all unique team IDs from the season
    team_ids = set()
    
    # Get teams from all matches in the competition/season
    season_matches = df_all[
        (df_all['ID_COMPETITION'] == competition) & 
        (df_all['ID_SEASON'] == season)
    ]
    
    if 'ID_HOME_CLUB' in season_matches.columns:
        team_ids.update(season_matches['ID_HOME_CLUB'].dropna().unique())
    if 'ID_AWAY_CLUB' in season_matches.columns:
        team_ids.update(season_matches['ID_AWAY_CLUB'].dropna().unique())
    
    # Convert IDs to names using mapping
    teams_data = []
    for team_id in sorted(team_ids):
        team_name = team_mapping.get(int(team_id), f"Team_{int(team_id)}")
        teams_data.append({
            'Team_ID': int(team_id),
            'Team': team_name,
            'Played': 0,
            'Won': 0,
            'Drawn': 0,
            'Lost': 0,
            'Goals_For': 0,
            'Goals_Against': 0,
            'Goal_Difference': 0,
            'Points': 0
        })
    
    table = pd.DataFrame(teams_data)
    
    return table

def update_league_table(table, match_results, use_actual_goals=True):
    """
    Update league table with match results.
    
    Parameters:
    -----------
    table : DataFrame
        Current league table
    match_results : DataFrame
        Matches with columns: ID_HOME_CLUB, ID_AWAY_CLUB, RESULT, 
                              ID_HOME_GOALS, ID_AWAY_GOALS (optional)
    use_actual_goals : bool
        If True, use actual goals from match_results. If False and goals missing, simulate.
    """
    table = table.copy()
    
    for _, match in match_results.iterrows():
        home_team = match['ID_HOME_CLUB']
        away_team = match['ID_AWAY_CLUB']
        result = match['RESULT']
        
        # Get goals if available, otherwise estimate from result
        if 'ID_HOME_GOALS' in match and pd.notna(match['ID_HOME_GOALS']) and use_actual_goals:
            home_goals = int(match['ID_HOME_GOALS'])
            away_goals = int(match['ID_AWAY_GOALS'])
        else:
            # Simulate goals based on result
            if result == 'W':  # Home win
                home_goals = np.random.randint(1, 4)
                away_goals = np.random.randint(0, home_goals)
            elif result == 'D':  # Draw
                goals = np.random.randint(0, 4)
                home_goals = goals
                away_goals = goals
            else:  # Away win (L)
                away_goals = np.random.randint(1, 4)
                home_goals = np.random.randint(0, away_goals)
        
        # Update home team (match by Team_ID)
        home_idx = table[table['Team_ID'] == int(home_team)].index
        if len(home_idx) > 0:
            table.loc[home_idx, 'Played'] += 1
            table.loc[home_idx, 'Goals_For'] += home_goals
            table.loc[home_idx, 'Goals_Against'] += away_goals
            table.loc[home_idx, 'Goal_Difference'] = table.loc[home_idx, 'Goals_For'] - table.loc[home_idx, 'Goals_Against']
            
            if result == 'W':
                table.loc[home_idx, 'Won'] += 1
                table.loc[home_idx, 'Points'] += 3
            elif result == 'D':
                table.loc[home_idx, 'Drawn'] += 1
                table.loc[home_idx, 'Points'] += 1
            else:  # L
                table.loc[home_idx, 'Lost'] += 1
        
        # Update away team (match by Team_ID)
        away_idx = table[table['Team_ID'] == int(away_team)].index
        if len(away_idx) > 0:
            table.loc[away_idx, 'Played'] += 1
            table.loc[away_idx, 'Goals_For'] += away_goals
            table.loc[away_idx, 'Goals_Against'] += home_goals
            table.loc[away_idx, 'Goal_Difference'] = table.loc[away_idx, 'Goals_For'] - table.loc[away_idx, 'Goals_Against']
            
            if result == 'W':  # Home win = away loss
                table.loc[away_idx, 'Lost'] += 1
            elif result == 'D':
                table.loc[away_idx, 'Drawn'] += 1
                table.loc[away_idx, 'Points'] += 1
            else:  # L = away win
                table.loc[away_idx, 'Won'] += 1
                table.loc[away_idx, 'Points'] += 3
    
    # Sort table: Points (desc), Goal Difference (desc), Goals For (desc)
    table = table.sort_values(['Points', 'Goal_Difference', 'Goals_For'], ascending=[False, False, False])
    table['Position'] = range(1, len(table) + 1)
    
    return table

def display_dual_tables(predicted_table, actual_table, title="League Table Comparison"):
    """
    Display predicted and actual league tables side by side.
    
    Parameters:
    -----------
    predicted_table : DataFrame
        Predicted league table
    actual_table : DataFrame
        Actual league table
    title : str
        Title for the display
    """
    # Prepare display columns: Position, Team, P, W, D, L, Pts (no goals)
    pred_display = predicted_table[['Position', 'Team', 'Played', 'Won', 'Drawn', 'Lost', 'Points']].copy()
    pred_display.columns = ['Pos', 'Team', 'P', 'W', 'D', 'L', 'Pts']
    
    actual_display = actual_table[['Position', 'Team', 'Played', 'Won', 'Drawn', 'Lost', 'Points']].copy()
    actual_display.columns = ['Pos', 'Team', 'P', 'W', 'D', 'L', 'Pts']
    
    # Merge on team name to align rows
    merged = pred_display.merge(
        actual_display,
        on='Team',
        how='outer',
        suffixes=('_Pred', '_Actual')
    )
    
    # Fill missing values with 0, but keep track of which teams have actually played
    merged = merged.fillna({'Pos_Pred': 0, 'Pos_Actual': 0, 'P_Pred': 0, 'P_Actual': 0,
                            'W_Pred': 0, 'W_Actual': 0, 'D_Pred': 0, 'D_Actual': 0,
                            'L_Pred': 0, 'L_Actual': 0, 'Pts_Pred': 0, 'Pts_Actual': 0})
    
    # Convert to int (handle NaN from merge)
    for col in ['Pos_Pred', 'Pos_Actual', 'P_Pred', 'P_Actual', 'W_Pred', 'W_Actual',
                'D_Pred', 'D_Actual', 'L_Pred', 'L_Actual', 'Pts_Pred', 'Pts_Actual']:
        merged[col] = merged[col].astype(int)
    
    # Sort by predicted position (0 goes to bottom)
    merged['_sort_key'] = merged.apply(
        lambda row: row['Pos_Pred'] if row['Pos_Pred'] > 0 else 999, axis=1
    )
    merged = merged.sort_values('_sort_key').drop('_sort_key', axis=1)
    
    print(f"\n{title}")
    print(f"{'='*120}")
    print(f"{'PREDICTED':<60} | {'ACTUAL':<60}")
    print(f"{'-'*60} | {'-'*60}")
    print(f"{'Pos':<4} {'Team':<35} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'Pts':<4} | "
          f"{'Pos':<4} {'Team':<35} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'Pts':<4}")
    print(f"{'-'*60} | {'-'*60}")
    
    for _, row in merged.iterrows():
        team_name = str(row['Team'])[:35]  # Truncate long team names
        pred_line = f"{row['Pos_Pred']:<4} {team_name:<35} {row['P_Pred']:<3} {row['W_Pred']:<3} " \
                   f"{row['D_Pred']:<3} {row['L_Pred']:<3} {row['Pts_Pred']:<4}"
        
        if row['Pos_Actual'] == 0 or pd.isna(row['Pos_Actual']):  # Team not in actual table yet
            actual_line = f"{'--':<4} {team_name:<35} {'--':<3} {'--':<3} {'--':<3} {'--':<3} {'--':<4}"
            pos_diff = 0
        else:
            actual_line = f"{int(row['Pos_Actual']):<4} {team_name:<35} {row['P_Actual']:<3} {row['W_Actual']:<3} " \
                         f"{row['D_Actual']:<3} {row['L_Actual']:<3} {row['Pts_Actual']:<4}"
            pos_diff = abs(row['Pos_Pred'] - row['Pos_Actual']) if row['Pos_Pred'] > 0 else 0
        
        # Highlight if positions differ significantly
        marker = " *" if pos_diff > 3 and row['Pos_Pred'] > 0 and row['Pos_Actual'] > 0 else "  "
        
        print(f"{pred_line} | {actual_line}{marker}")
    
    print(f"{'-'*60} | {'-'*60}")
    print(f"* = Position difference > 3 positions")
    print(f"{'='*120}\n")

def simulate_result(predicted_proba, random_state=None):
    """
    Simulate match result based on predicted probabilities.
    
    Parameters:
    -----------
    predicted_proba : array
        Array of shape (n_samples, n_classes) with probabilities
        Order should match le.classes_: [W, D, L] or similar
    
    Returns:
    --------
    results : array
        Simulated results using class labels
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = predicted_proba.shape[0]
    results = []
    
    for i in range(n_samples):
        # Sample from multinomial distribution based on probabilities
        result_idx = np.random.choice(len(predicted_proba[i]), p=predicted_proba[i])
        results.append(result_idx)
    
    return np.array(results)

## 6. Predict Matchday Function

def predict_round(
    X_all, 
    y_all_encoded, 
    dates_all,
    df_all,
    round_number,
    training_cutoff_date,
    predicted_league_table,
    actual_league_table,
    le,
    scaler=None,
    model=None,
    simulate=True,
    verbose=True
):
    """
    Predict all matches in a specific matchday using all actual results up to and including
    the previous matchday. All matches in the current matchday are predicted together as a bundle.
    
    Parameters:
    -----------
    X_all : DataFrame
        All features
    y_all_encoded : array
        Encoded target for all rows
    dates_all : Series
        Dates for all rows
    df_all : DataFrame
        Full dataframe with metadata
    round_number : int
        Matchday number (ID_ROUND_NUMBER) to predict (e.g., 1 for "1. Matchday")
    training_cutoff_date : datetime or None
        Last date of previous matchday (inclusive). All data <= this date is used for training.
        If None, use all data before the current matchday (for first matchday).
    predicted_league_table : DataFrame
        Current predicted league table
    actual_league_table : DataFrame
        Current actual league table
    le : LabelEncoder
        Fitted label encoder
    scaler : StandardScaler, optional
        Fitted scaler (if None, will fit new one)
    model : MLPClassifier, optional
        Trained model (if None, will train new one)
    simulate : bool
        If True, simulate results and update league table
    
    Returns:
    --------
    predictions : dict
        Contains predictions, probabilities, simulated results, and updated table
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Matchday {round_number} - Predicting all matches in this matchday")
        print(f"{'='*70}")
    
    # Split: training = all data up to and including the previous matchday
    # Test = all matches in the current matchday (predicted together as a bundle)
    if training_cutoff_date is not None:
        training_cutoff_dt = pd.to_datetime(training_cutoff_date)
        # Include all data up to and including the cutoff date (previous matchday's actual results)
        train_mask = dates_all <= training_cutoff_dt
        cutoff_info = f" (up to and including {training_cutoff_dt.date()})"
    else:
        # For first matchday, use all historical data before it (from all previous seasons)
        matchday_first_date = df_all[df_all['ID_ROUND_NUMBER'] == round_number]['ID_DATE'].min()
        matchday_first_date_dt = pd.to_datetime(matchday_first_date)
        train_mask = dates_all < matchday_first_date_dt
        cutoff_info = f" (before Matchday {round_number}, includes all historical data)"
    
    test_mask = (df_all['ID_ROUND_NUMBER'] == round_number) & \
                (df_all['ID_COMPETITION'] == 'GB1') & \
                (df_all['ID_SEASON'] == 2024)
    
    # Reduce copying - use views where possible for training (only copy test)
    X_train = X_all[train_mask]
    y_train = y_all_encoded[train_mask]
    X_test = X_all[test_mask].copy()  # Need copy for test to avoid issues
    
    if len(X_test) == 0:
        if verbose:
            print(f"  WARNING: No matches found for Matchday {round_number}")
        return None
    
    # Check if training set is empty and handle it (especially for first matchday)
    if len(X_train) == 0:
        if verbose:
            print(f"  WARNING: No training data found with date filter")
            print(f"  Falling back to all historical data from previous seasons...")
        
        # Fallback strategy: use all data that's not part of the 2024 Premier League season we're predicting
        # This includes:
        # 1. All data from seasons < 2024
        # 2. All data from other competitions in 2024
        # 3. All 2024 GB1 data except matches we're predicting (other matchdays or past rounds)
        
        fallback_train_mask = (
            (df_all['ID_SEASON'] < 2024) |  # Previous seasons
            (df_all['ID_COMPETITION'] != 'GB1') |  # Other competitions
            ((df_all['ID_COMPETITION'] == 'GB1') & (df_all['ID_SEASON'] == 2024) & 
             (df_all['ID_ROUND_NUMBER'] != round_number))  # Other 2024 GB1 matchdays
        )
        
        X_train = X_all[fallback_train_mask]
        y_train = y_all_encoded[fallback_train_mask]
        cutoff_info = f" (using all historical data: previous seasons + other competitions)"
        
        if len(X_train) == 0:
            if verbose:
                print(f"  ERROR: Still no training data available after fallback!")
            return None
    
    if verbose:
        print(f"  Training set: {len(X_train):,} matches{cutoff_info}")
        print(f"  Test set: {len(X_test):,} matches (all matches in Matchday {round_number})")
    
    # Scale features - reuse scaler if provided, otherwise fit new one
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        # For incremental updates, refit scaler on all training data
        # (This is necessary because feature distributions may shift)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    # Train model - optimized for speed
    if model is None:
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=min(256, len(X_train_scaled)),  # Adaptive batch size
            learning_rate='adaptive',  # Adaptive learning rate for faster convergence
            learning_rate_init=0.001,
            max_iter=200,  # Reduced from 500 for faster training
            shuffle=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,  # More aggressive early stopping
            tol=1e-4,  # Tolerance for early stopping
            verbose=False
        )
    
    if verbose:
        print(f"  Training model...", end=" ", flush=True)
    model.fit(X_train_scaled, y_train)
    if verbose:
        print(f"[OK] ({model.n_iter_} iterations)")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Get test set metadata for results - use iloc for faster indexing
    test_indices = df_all.index[test_mask]
    test_metadata = df_all.loc[test_indices, [
        'ID_GAME', 'ID_DATE', 'ID_ROUND', 'ID_ROUND_NUMBER', 'ID_SEASON',
        'ID_HOME_CLUB', 'ID_AWAY_CLUB',
        'ID_HOME_TEAM', 'ID_AWAY_TEAM',
        'ID_HOME_GOALS', 'ID_AWAY_GOALS', 'RESULT'
    ]].copy()
    
    # Optimize label encoder operations - cache class indices
    class_to_idx = {cls: idx for idx, cls in enumerate(le.classes_)}
    test_metadata['PREDICTED_RESULT'] = le.inverse_transform(y_pred)
    test_metadata['PRED_W_PROB'] = y_pred_proba[:, class_to_idx['W']]
    test_metadata['PRED_D_PROB'] = y_pred_proba[:, class_to_idx['D']]
    test_metadata['PRED_L_PROB'] = y_pred_proba[:, class_to_idx['L']]
    
    # Simulate results if requested
    if simulate:
        simulated_indices = simulate_result(y_pred_proba, random_state=round_number)
        test_metadata['SIMULATED_RESULT'] = le.inverse_transform(simulated_indices)
    else:
        test_metadata['SIMULATED_RESULT'] = test_metadata['PREDICTED_RESULT']
    
    # Update predicted league table with simulated results
    if simulate:
        # Use simulated results for table update
        pred_table_update_data = test_metadata[['ID_HOME_CLUB', 'ID_AWAY_CLUB', 'SIMULATED_RESULT', 
                                                 'ID_HOME_GOALS', 'ID_AWAY_GOALS']].copy()
        pred_table_update_data = pred_table_update_data.rename(columns={'SIMULATED_RESULT': 'RESULT'})
    else:
        # Use predicted results
        pred_table_update_data = test_metadata[['ID_HOME_CLUB', 'ID_AWAY_CLUB', 'PREDICTED_RESULT', 
                                                 'ID_HOME_GOALS', 'ID_AWAY_GOALS']].copy()
        pred_table_update_data = pred_table_update_data.rename(columns={'PREDICTED_RESULT': 'RESULT'})
    
    updated_predicted_table = update_league_table(predicted_league_table, pred_table_update_data, use_actual_goals=False)
    
    # Update actual league table with real results (if available)
    updated_actual_table = actual_league_table.copy()
    matches_with_results = test_metadata[test_metadata['RESULT'].notna()].copy()
    
    if len(matches_with_results) > 0:
        actual_table_update_data = matches_with_results[['ID_HOME_CLUB', 'ID_AWAY_CLUB', 'RESULT',
                                                          'ID_HOME_GOALS', 'ID_AWAY_GOALS']].copy()
        updated_actual_table = update_league_table(actual_league_table, actual_table_update_data, use_actual_goals=True)
    
    # Calculate accuracy if we have actual results
    if verbose:
        if 'RESULT' in test_metadata.columns and test_metadata['RESULT'].notna().any():
            actual_results = test_metadata['RESULT'].dropna()
            predicted_results = test_metadata.loc[actual_results.index, 'PREDICTED_RESULT']
            accuracy = (actual_results == predicted_results).mean()
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        else:
            print(f"  Note: Actual results not yet available for this matchday")
    
    return {
        'round_number': round_number,
        'predictions': test_metadata,
        'predicted_league_table': updated_predicted_table,
        'actual_league_table': updated_actual_table,
        'model': model,
        'scaler': scaler,
        'X_train_scaled': X_train_scaled,
        'y_train': y_train,
        'X_test_scaled': X_test_scaled,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

## 7. Execute Rolling Predictions Matchday by Matchday

# Create team ID to name mapping
team_mapping = create_team_mapping(df, season=2024, competition='GB1')
print(f"\nTeam mapping created: {len(team_mapping)} teams")

# Initialize both predicted and actual league tables
predicted_league_table = initialize_league_table(df, team_mapping, season=2024, competition='GB1')
actual_league_table = initialize_league_table(df, team_mapping, season=2024, competition='GB1')
print(f"\n{'='*70}")
print(f"LEAGUE TABLES INITIALIZED")
print(f"{'='*70}")
print(f"Teams: {len(predicted_league_table)}")

# Store all prediction results
all_predictions = {}

print(f"\n{'='*70}")
print(f"ROLLING PREDICTION PROCESS")
print(f"{'='*70}")
print(f"Processing matchdays sequentially: all matches in each matchday predicted together")
print(f"For Matchday n+1, we use all actual results up to and including Matchday n")

# Configuration for display frequency (set to None to show all, or N to show every Nth round)
DISPLAY_EVERY_N_ROUNDS = None  # Change to 5 to show every 5th round, or None to show all
SHOW_INTERMEDIATE_TABLES = True  # Set to False to skip intermediate table displays entirely

# Execute predictions for each matchday in order (all matches in each matchday predicted together)
print(f"Predicting {len(matchday_numbers)} matchdays...")
start_time_loop = pd.Timestamp.now()

for idx, matchday_num in enumerate(matchday_numbers, 1):
    # Determine training cutoff: use previous matchday's last date (actual results)
    # For the first matchday, use None (all data before first matchday)
    if idx == 1:
        training_cutoff_date = None
        prev_matchday_num = None
    else:
        prev_matchday_num = matchday_numbers[idx - 2]  # Previous matchday number
        training_cutoff_date = matchday_last_dates[prev_matchday_num]
    
    # Determine if we should show verbose output and tables for this matchday
    should_display_table = SHOW_INTERMEDIATE_TABLES and (
        DISPLAY_EVERY_N_ROUNDS is None or 
        idx % DISPLAY_EVERY_N_ROUNDS == 0 or 
        idx == len(matchday_numbers)
    )
    should_verbose = should_display_table  # Only verbose if displaying tables
    
    if should_verbose and training_cutoff_date is not None:
        print(f"\nUsing actual results from Matchday {prev_matchday_num} (ended {pd.to_datetime(training_cutoff_date).date()}) for training")
    
    # Predict all matches in this matchday together
    result = predict_round(
        X_all=X_all,
        y_all_encoded=y_all_encoded,
        dates_all=dates_all,
        df_all=df,
        round_number=matchday_num,
        training_cutoff_date=training_cutoff_date,
        predicted_league_table=predicted_league_table,
        actual_league_table=actual_league_table,
        le=le,
        simulate=True,  # Simulate results and update table
        verbose=should_verbose
    )
    
    if result is not None:
        all_predictions[matchday_num] = result
        # Update both league tables for next iteration
        predicted_league_table = result['predicted_league_table']
        actual_league_table = result['actual_league_table']
        
        # Display dual tables after this matchday (if enabled)
        if should_display_table:
            display_dual_tables(
                predicted_league_table, 
                actual_league_table,
                title=f"After Matchday {matchday_num}"
            )
        elif not should_verbose:
            # Minimal progress indicator
            print(f"Matchday {matchday_num}/{len(matchday_numbers)} completed", end="\r", flush=True)

# Clear progress line
if not SHOW_INTERMEDIATE_TABLES:
    print()  # New line after progress indicator

elapsed_loop = (pd.Timestamp.now() - start_time_loop).total_seconds()
print(f"\nPrediction loop completed in {elapsed_loop:.1f} seconds ({elapsed_loop/60:.1f} minutes)")

## 8. Aggregate and Display Results

print(f"\n{'='*70}")
print(f"PREDICTION SUMMARY")
print(f"{'='*70}")

# Combine all predictions
all_pred_df = pd.concat(
    [pred['predictions'] for pred in all_predictions.values()],
    ignore_index=True
)

print(f"\nTotal matches predicted: {len(all_pred_df)}")
print(f"\nPredictions by matchday:")
matchday_counts = all_pred_df.groupby('ID_ROUND_NUMBER').size()
print(matchday_counts)

# Show prediction distribution
print(f"\nPredicted result distribution:")
print(all_pred_df['PREDICTED_RESULT'].value_counts())

print(f"\nSimulated result distribution:")
print(all_pred_df['SIMULATED_RESULT'].value_counts())

# Show matches where we have actual results vs predictions
if 'RESULT' in all_pred_df.columns:
    matches_with_results = all_pred_df[all_pred_df['RESULT'].notna()].copy()
    matches_without_results = all_pred_df[all_pred_df['RESULT'].isna()].copy()
    
    print(f"\nMatches with actual results: {len(matches_with_results)}")
    print(f"Matches without results yet: {len(matches_without_results)}")
    
    if len(matches_with_results) > 0:
        overall_accuracy = (matches_with_results['RESULT'] == 
                           matches_with_results['PREDICTED_RESULT']).mean()
        print(f"\nOverall accuracy (on matches with results): {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

## 9. Final League Table

print(f"\n{'='*70}")
print(f"FINAL LEAGUE TABLES (After All Matchdays)")
print(f"{'='*70}")

# Display final dual table comparison
display_dual_tables(
    predicted_league_table,
    actual_league_table,
    title="FINAL LEAGUE TABLE COMPARISON"
)

# Show champions comparison
predicted_champion = predicted_league_table.iloc[0]
actual_champion = actual_league_table.iloc[0] if len(actual_league_table[actual_league_table['Played'] > 0]) > 0 else None

print(f"\n{'='*70}")
print(f"CHAMPIONS COMPARISON")
print(f"{'='*70}")
print(f"Predicted Champion: {predicted_champion['Team']} ({predicted_champion['Points']} points)")
if actual_champion is not None:
    print(f"Actual Champion: {actual_champion['Team']} ({actual_champion['Points']} points)")
    if predicted_champion['Team'] == actual_champion['Team']:
        print("✓ Correct prediction!")
else:
    print("Actual Champion: Not yet determined")

# Show top 4 comparison
print(f"\n{'='*70}")
print(f"TOP 4 COMPARISON (Champions League qualification)")
print(f"{'='*70}")

pred_top4 = predicted_league_table.head(4)[['Position', 'Team', 'Points']]
actual_top4 = actual_league_table.head(4)[['Position', 'Team', 'Points']] if len(actual_league_table[actual_league_table['Played'] > 0]) >= 4 else None

print(f"Predicted Top 4:")
print(pred_top4.to_string(index=False))
if actual_top4 is not None:
    print(f"\nActual Top 4:")
    print(actual_top4.to_string(index=False))
    
    # Count how many teams match
    pred_teams = set(pred_top4['Team'].values)
    actual_teams = set(actual_top4['Team'].values)
    matches = len(pred_teams & actual_teams)
    print(f"\nTeams correctly predicted in Top 4: {matches}/4")

# Show relegation zone comparison
print(f"\n{'='*70}")
print(f"RELEGATION ZONE COMPARISON (Bottom 3)")
print(f"{'='*70}")

pred_bottom3 = predicted_league_table.tail(3)[['Position', 'Team', 'Points']]
actual_bottom3 = actual_league_table.tail(3)[['Position', 'Team', 'Points']] if len(actual_league_table[actual_league_table['Played'] > 0]) >= 3 else None

print(f"Predicted Bottom 3:")
print(pred_bottom3.to_string(index=False))
if actual_bottom3 is not None:
    print(f"\nActual Bottom 3:")
    print(actual_bottom3.to_string(index=False))
    
    # Count how many teams match
    pred_teams = set(pred_bottom3['Team'].values)
    actual_teams = set(actual_bottom3['Team'].values)
    matches = len(pred_teams & actual_teams)
    print(f"\nTeams correctly predicted in Bottom 3: {matches}/3")

## 10. Display Predictions by Matchday

print(f"\n{'='*70}")
print(f"DETAILED PREDICTIONS BY MATCHDAY")
print(f"{'='*70}")

# Show sample predictions for first and last few matchdays
for matchday_num in [matchday_numbers[0], matchday_numbers[1], 
                     matchday_numbers[-2], matchday_numbers[-1]]:
    if matchday_num not in all_predictions:
        continue
    
    pred_data = all_predictions[matchday_num]['predictions'].copy()
    
    print(f"\nMatchday {matchday_num}")
    print(f"{'-'*70}")
    
    # Sort by date
    pred_data = pred_data.sort_values('ID_DATE')
    
    # Display key information - include both IDs and team names
    display_cols = ['ID_DATE', 'ID_HOME_TEAM', 'ID_AWAY_TEAM', 
                    'ID_HOME_CLUB', 'ID_AWAY_CLUB',
                    'PREDICTED_RESULT', 'SIMULATED_RESULT',
                    'PRED_W_PROB', 'PRED_D_PROB', 'PRED_L_PROB']
    
    # Add actual result if available
    if 'RESULT' in pred_data.columns:
        display_cols.insert(-3, 'RESULT')
    
    # Format probabilities as percentages
    pred_data_display = pred_data[display_cols].copy()
    pred_data_display['PRED_W_PROB'] = (pred_data_display['PRED_W_PROB'] * 100).round(1)
    pred_data_display['PRED_D_PROB'] = (pred_data_display['PRED_D_PROB'] * 100).round(1)
    pred_data_display['PRED_L_PROB'] = (pred_data_display['PRED_L_PROB'] * 100).round(1)
    
    # Show first 5 matches
    print(pred_data_display.head().to_string(index=False))

## 11. Save Results

# Save in the same directory as this script (Notebooks folder)
script_dir = Path(__file__).parent

output_path_pred = script_dir / "premier_league_2024_predictions.csv"
output_path_table_pred = script_dir / "premier_league_2024_final_table_predicted.csv"
output_path_table_actual = script_dir / "premier_league_2024_final_table_actual.csv"
output_path_table_combined = script_dir / "premier_league_2024_final_table_combined.csv"
output_path_table_evolution = script_dir / "premier_league_2024_table_evolution.csv"

print(f"\nSaving files to: {script_dir}")

print(f"\n{'='*70}")
print(f"Saving Results")
print(f"{'='*70}")

# Save predictions - ONE ROW PER MATCH with all prediction details
# Include both IDs and team names for easy identification
output_df = all_pred_df[[
    'ID_GAME', 'ID_DATE', 'ID_ROUND_NUMBER', 'ID_ROUND',
    'ID_HOME_TEAM', 'ID_AWAY_TEAM',
    'ID_HOME_CLUB', 'ID_AWAY_CLUB',
    'PREDICTED_RESULT', 'SIMULATED_RESULT',
    'PRED_W_PROB', 'PRED_D_PROB', 'PRED_L_PROB'
]].copy()

# Add actual results if available
if 'RESULT' in all_pred_df.columns:
    output_df['ACTUAL_RESULT'] = all_pred_df['RESULT']

output_df.to_csv(output_path_pred, index=False)
print(f"Predictions saved to: {output_path_pred}")
print(f"  Format: One row per match ({len(output_df)} matches)")
print(f"  Columns: Game info, Teams (names + IDs), Predictions, Probabilities, Actual results")

# Save final league tables (predicted and actual) - only P, W, D, L, Pts
predicted_table_display = predicted_league_table[['Position', 'Team', 'Played', 'Won', 'Drawn', 'Lost', 'Points']].copy()
predicted_table_display.columns = ['Pos', 'Team', 'P', 'W', 'D', 'L', 'Pts']
predicted_table_display.to_csv(output_path_table_pred, index=False)
print(f"Predicted league table saved to: {output_path_table_pred}")

actual_table_display = actual_league_table[['Position', 'Team', 'Played', 'Won', 'Drawn', 'Lost', 'Points']].copy()
actual_table_display.columns = ['Pos', 'Team', 'P', 'W', 'D', 'L', 'Pts']
actual_table_display.to_csv(output_path_table_actual, index=False)
print(f"Actual league table saved to: {output_path_table_actual}")

# Save combined final table (predicted and actual side-by-side)
combined_table = predicted_table_display.merge(
    actual_table_display,
    on='Team',
    how='outer',
    suffixes=('_Predicted', '_Actual')
).fillna(0).sort_values('Pos_Predicted')

# Convert position 0 to NaN for teams not yet in table
combined_table['Pos_Predicted'] = combined_table['Pos_Predicted'].replace(0, np.nan)
combined_table['Pos_Actual'] = combined_table['Pos_Actual'].replace(0, np.nan)

combined_table.to_csv(output_path_table_combined, index=False)
print(f"Combined final table saved to: {output_path_table_combined}")
print(f"  Format: One row per team with both predicted and actual standings")

# Save table evolution per matchday (both predicted and actual)
table_evolution_rows = []
for matchday_num in sorted(all_predictions.keys()):
    pred_result = all_predictions[matchday_num]
    
    # Get predicted table after this matchday
    pred_table_after = pred_result['predicted_league_table'][['Position', 'Team', 'Played', 'Won', 'Drawn', 'Lost', 'Points']].copy()
    pred_table_after['Matchday'] = matchday_num
    pred_table_after['Table_Type'] = 'Predicted'
    pred_table_after.columns = ['Pos', 'Team', 'P', 'W', 'D', 'L', 'Pts', 'Matchday', 'Table_Type']
    
    # Get actual table after this matchday
    actual_table_after = pred_result['actual_league_table'][['Position', 'Team', 'Played', 'Won', 'Drawn', 'Lost', 'Points']].copy()
    actual_table_after['Matchday'] = matchday_num
    actual_table_after['Table_Type'] = 'Actual'
    actual_table_after.columns = ['Pos', 'Team', 'P', 'W', 'D', 'L', 'Pts', 'Matchday', 'Table_Type']
    
    # Only include teams that have played
    pred_table_after = pred_table_after[pred_table_after['P'] > 0]
    actual_table_after = actual_table_after[actual_table_after['P'] > 0]
    
    table_evolution_rows.append(pred_table_after)
    table_evolution_rows.append(actual_table_after)

if table_evolution_rows:
    table_evolution = pd.concat(table_evolution_rows, ignore_index=True)
    table_evolution = table_evolution[['Matchday', 'Table_Type', 'Pos', 'Team', 'P', 'W', 'D', 'L', 'Pts']].sort_values(['Matchday', 'Table_Type', 'Pos'])
    table_evolution.to_csv(output_path_table_evolution, index=False)
    print(f"Table evolution per matchday saved to: {output_path_table_evolution}")
    print(f"  Format: One row per team per matchday ({len(table_evolution)} rows)")
    print(f"  Includes: Predicted and Actual standings after each matchday")

## 12. Summary Statistics

print(f"\n{'='*70}")
print(f"FINAL SUMMARY")
print(f"{'='*70}")
print(f"Total matches predicted: {len(all_pred_df)}")
print(f"Total matchdays processed: {len(all_predictions)}")
print(f"Total teams: {len(predicted_league_table)}")

if 'RESULT' in all_pred_df.columns:
    matches_with_results = all_pred_df[all_pred_df['RESULT'].notna()]
    if len(matches_with_results) > 0:
        final_accuracy = (matches_with_results['RESULT'] == 
                         matches_with_results['PREDICTED_RESULT']).mean()
        print(f"\nOverall Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"Matches evaluated: {len(matches_with_results)} / {len(all_pred_df)}")

print(f"\nPrediction breakdown (Predicted):")
print(f"  Home wins (W): {(all_pred_df['PREDICTED_RESULT'] == 'W').sum()}")
print(f"  Draws (D): {(all_pred_df['PREDICTED_RESULT'] == 'D').sum()}")
print(f"  Away wins (L): {(all_pred_df['PREDICTED_RESULT'] == 'L').sum()}")

print(f"\nPrediction breakdown (Simulated):")
print(f"  Home wins (W): {(all_pred_df['SIMULATED_RESULT'] == 'W').sum()}")
print(f"  Draws (D): {(all_pred_df['SIMULATED_RESULT'] == 'D').sum()}")
print(f"  Away wins (L): {(all_pred_df['SIMULATED_RESULT'] == 'L').sum()}")

print(f"\n{'='*70}")
print(f"Script completed successfully!")
print(f"{'='*70}")

