"""
Champions League 2023 Prediction Script (Rolling Daily)
=======================================================

This script predicts CL 2023 matches using all available data up to the day
before each match. This maximizes data usage and reflects a realistic prediction
scenario where we use the most recent information.

Models:
1. Logistic Regression
2. XGBoost
3. Neural Network (MLP - optimized)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ==========================================
# 1. Load Data
# ==========================================

def load_data():
    possible_paths = [
        Path("../../Data/game_features.parquet"),
        Path("../Data/game_features.parquet"),
        Path("PROJECT 3/Code/Data/game_features.parquet"),
        Path("Code/Data/game_features.parquet")
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
            
    if data_path is None:
        import os
        data_path = Path(os.getcwd()) / "PROJECT 3" / "Code" / "Data" / "game_features.parquet"
        
    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    return df

# ==========================================
# 2. Preprocessing & Setup
# ==========================================

def prepare_features(df):
    metadata_cols = [col for col in df.columns if col.startswith('ID_')]
    target_col = 'RESULT'
    
    if target_col not in df.columns:
        raise ValueError("RESULT column not found in dataset!")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictor_cols = [col for col in numeric_cols if col not in metadata_cols and col != target_col]
    
    # Fill missing values with median
    print("Filling missing values with median...")
    df[predictor_cols] = df[predictor_cols].fillna(df[predictor_cols].median())
    
    print(f"\nFeature preparation:")
    print(f"  Metadata columns: {len(metadata_cols)}")
    print(f"  Predictor features: {len(predictor_cols)}")
    
    return predictor_cols, target_col

def prepare_cl_2023_data(df, predictor_cols, target_col, le):
    # Convert date
    df['ID_DATE_DT'] = pd.to_datetime(df['ID_DATE'])
    
    # Get CL 2023 matches
    test_mask = (df['ID_COMPETITION'] == 'CL') & (df['ID_SEASON'] == 2023)
    test_meta = df.loc[test_mask, ['ID_GAME', 'ID_DATE', 'ID_DATE_DT', 'ID_ROUND', 
                                     'ID_HOME_TEAM', 'ID_AWAY_TEAM', 'RESULT']].copy()
    
    # Add Prediction Stage mapping
    stage_mapping = {
        'Group A': 'Group Stage', 'Group B': 'Group Stage', 'Group C': 'Group Stage', 'Group D': 'Group Stage',
        'Group E': 'Group Stage', 'Group F': 'Group Stage', 'Group G': 'Group Stage', 'Group H': 'Group Stage',
        'last 16 1st leg': 'Last 16', 'last 16 2nd leg': 'Last 16',
        'Quarter-Finals 1st leg': 'Quarter-Finals', 'Quarter-Finals 2nd leg': 'Quarter-Finals',
        'Semi-Finals 1st Leg': 'Semi-Finals', 'Semi-Finals 2nd Leg': 'Semi-Finals',
        'Final': 'Final'
    }
    test_meta['Prediction_Stage'] = test_meta['ID_ROUND'].map(stage_mapping)
    
    # Sort by date for rolling prediction
    test_meta = test_meta.sort_values('ID_DATE_DT').reset_index(drop=True)
    
    print(f"\nCL 2023 matches to predict: {len(test_meta)}")
    
    return test_meta

# ==========================================
# 3. Models
# ==========================================

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42, verbose=0)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train, verbose=False)
    return model

def train_mlp(X_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    return model

# ==========================================
# 4. Rolling Prediction
# ==========================================

def predict_match_by_match(df, test_meta, predictor_cols, target_col, le):
    """
    For each match, train on all data before that match's date.
    """
    
    X_all = df[predictor_cols].values
    y_all = le.transform(df[target_col])
    dates_all = df['ID_DATE_DT'].values
    
    all_predictions = []
    
    # Group matches by unique dates to train once per date
    unique_dates = test_meta['ID_DATE_DT'].unique()
    
    print(f"\nRolling prediction for {len(unique_dates)} unique match dates...")
    
    for i, match_date in enumerate(unique_dates):
        # Get matches on this date
        matches_on_date = test_meta[test_meta['ID_DATE_DT'] == match_date]
        
        print(f"\n[{i+1}/{len(unique_dates)}] Date: {match_date.date()} ({len(matches_on_date)} matches)")
        
        # Train on all data BEFORE this date
        train_mask = dates_all < np.datetime64(match_date)
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        
        if len(X_train) == 0:
            print(f"  WARNING: No training data available before {match_date.date()}")
            continue
        
        print(f"  Training samples: {len(X_train):,}")
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Get test data for this date
        test_indices = matches_on_date.index
        X_test = X_all[test_indices]
        X_test_scaled = scaler.transform(X_test)
        
        # Train models (quietly)
        lr_model = train_logistic_regression(X_train_scaled, y_train)
        xgb_model = train_xgboost(X_train_scaled, y_train)
        mlp_model = train_mlp(X_train_scaled, y_train)
        
        # Predict
        lr_preds = lr_model.predict(X_test_scaled)
        lr_probs = lr_model.predict_proba(X_test_scaled)
        
        xgb_preds = xgb_model.predict(X_test_scaled)
        xgb_probs = xgb_model.predict_proba(X_test_scaled)
        
        mlp_preds = mlp_model.predict(X_test_scaled)
        mlp_probs = mlp_model.predict_proba(X_test_scaled)
        
        # Store results
        date_results = matches_on_date.copy()
        date_results['LR_PRED'] = le.inverse_transform(lr_preds)
        date_results['XGB_PRED'] = le.inverse_transform(xgb_preds)
        date_results['MLP_PRED'] = le.inverse_transform(mlp_preds)
        
        for j, cls in enumerate(le.classes_):
            date_results[f'LR_PROB_{cls}'] = lr_probs[:, j]
            date_results[f'XGB_PROB_{cls}'] = xgb_probs[:, j]
            date_results[f'MLP_PROB_{cls}'] = mlp_probs[:, j]
        
        all_predictions.append(date_results)
    
    return pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

# ==========================================
# 5. Main Execution
# ==========================================

def main():
    print("="*80)
    print("Champions League 2023 - Rolling Daily Prediction")
    print("="*80)
    
    # 1. Load
    df = load_data()
    
    # 2. Prepare
    predictor_cols, target_col = prepare_features(df)
    
    le = LabelEncoder()
    le.fit(df[target_col].unique())
    print(f"Target classes: {le.classes_}")
    
    test_meta = prepare_cl_2023_data(df, predictor_cols, target_col, le)
    
    # 3. Rolling Prediction
    results = predict_match_by_match(df, test_meta, predictor_cols, target_col, le)
    
    if results.empty:
        print("No predictions generated!")
        return
    
    # 4. Save
    output_dir = Path("PROJECT 3/Code/Predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions_2023_rolling.csv"
    results.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Predictions saved to: {output_path}")
    
    # 5. Visualization & Evaluation
    print(f"\n{'='*80}")
    print("CHAMPIONS LEAGUE 2023 - ROLLING PREDICTIONS")
    print(f"{'='*80}")
    
    # Sort by date
    results = results.sort_values('ID_DATE')
    
    # Overall Accuracy
    if 'RESULT' in results.columns and results['RESULT'].notna().any():
        valid_res = results[results['RESULT'].notna()]
        print("\nOVERALL ACCURACY:")
        print(f"  Logistic Regression: {accuracy_score(valid_res['RESULT'], valid_res['LR_PRED']):.4f}")
        print(f"  XGBoost:             {accuracy_score(valid_res['RESULT'], valid_res['XGB_PRED']):.4f}")
        print(f"  MLP:                 {accuracy_score(valid_res['RESULT'], valid_res['MLP_PRED']):.4f}")
    
    # Stage-by-stage visualization
    stages_order = ['Group Stage', 'Last 16', 'Quarter-Finals', 'Semi-Finals', 'Final']
    
    for stage in stages_order:
        stage_data = results[results['Prediction_Stage'] == stage]
        if stage_data.empty:
            continue
            
        print(f"\n>>> {stage.upper()} <<<")
        print(f"{'-'*80}")
        print(f"{'Date':<12} {'Home Team':<25} {'Away Team':<25} {'Actual':<8} {'LR':<5} {'XGB':<5} {'MLP':<5}")
        print(f"{'-'*80}")
        
        for _, row in stage_data.iterrows():
            date_str = row['ID_DATE'].strftime('%Y-%m-%d') if pd.notna(row['ID_DATE']) else "N/A"
            actual = row['RESULT'] if pd.notna(row['RESULT']) else "-"
            
            # Highlight correct predictions
            lr_mark = "✓" if actual != "-" and row['LR_PRED'] == actual else row['LR_PRED']
            xgb_mark = "✓" if actual != "-" and row['XGB_PRED'] == actual else row['XGB_PRED']
            mlp_mark = "✓" if actual != "-" and row['MLP_PRED'] == actual else row['MLP_PRED']
            
            print(f"{date_str:<12} {str(row['ID_HOME_TEAM'])[:24]:<25} {str(row['ID_AWAY_TEAM'])[:24]:<25} {actual:<8} {lr_mark:<5} {xgb_mark:<5} {mlp_mark:<5}")
            
        # Stage Accuracy
        if 'RESULT' in stage_data.columns and stage_data['RESULT'].notna().any():
            valid_stage = stage_data[stage_data['RESULT'].notna()]
            if len(valid_stage) > 0:
                acc_xgb = accuracy_score(valid_stage['RESULT'], valid_stage['XGB_PRED'])
                print(f"{'-'*80}")
                print(f"Stage Accuracy (XGBoost): {acc_xgb:.2%}")

if __name__ == "__main__":
    main()
