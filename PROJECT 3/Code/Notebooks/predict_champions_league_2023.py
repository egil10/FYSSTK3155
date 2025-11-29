"""
Champions League 2023 Prediction Script (Full Season)
=====================================================

This script trains models on all data prior to the 2023 Champions League season
and predicts the outcome of the entire 2023 tournament.

Models:
1. Logistic Regression
2. XGBoost
3. RNN (PyTorch with MLP fallback)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys

# Models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Will fallback to MLPClassifier.")
except OSError:
    TORCH_AVAILABLE = False
    print("PyTorch DLL load failed. Will fallback to MLPClassifier.")

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

def prepare_train_test(df, predictor_cols, target_col, le):
    # Convert date
    df['ID_DATE_DT'] = pd.to_datetime(df['ID_DATE'])
    
    # Define 2023 Season Start (Group Stage start)
    # We want to train on EVERYTHING before the CL 2023 season starts
    season_start_date = pd.Timestamp("2023-09-19")
    
    print(f"\nSplitting data...")
    print(f"  Season Start Date: {season_start_date}")
    
    # Train: All games before 2023 season start
    train_mask = df['ID_DATE_DT'] < season_start_date
    
    # Test: CL 2023 matches
    test_mask = (df['ID_COMPETITION'] == 'CL') & (df['ID_SEASON'] == 2023)
    
    X_all = df[predictor_cols].values
    y_all = le.transform(df[target_col])
    
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask] # Can be used for evaluation if available
    
    # Metadata for test set
    test_meta = df.loc[test_mask, ['ID_GAME', 'ID_DATE', 'ID_ROUND', 'ID_HOME_TEAM', 'ID_AWAY_TEAM', 'RESULT']].copy()
    
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
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples (CL 2023): {len(X_test):,}")
    
    return X_train, y_train, X_test, y_test, test_meta

# ==========================================
# 3. Models
# ==========================================

# --- PyTorch RNN ---
if TORCH_AVAILABLE:
    class CL_RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(CL_RNN, self).__init__()
            self.hidden_size = hidden_size
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            x = x.unsqueeze(1) # (batch, 1, features)
            out, _ = self.rnn(x)
            out = out[:, -1, :] 
            out = self.fc(out)
            return out

    def train_pytorch_rnn(X_train, y_train, input_size, num_classes, epochs=20, hidden_size=64):
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        
        model = CL_RNN(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        return model

# --- Training Functions ---

def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression...")
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_rnn_or_mlp(X_train, y_train, num_classes):
    if TORCH_AVAILABLE:
        try:
            print("Training PyTorch RNN...")
            model = train_pytorch_rnn(X_train, y_train, X_train.shape[1], num_classes)
            return model, "PyTorch"
        except Exception as e:
            print(f"PyTorch training failed: {e}. Fallback to MLP.")
    
    print("Training MLPClassifier (Fallback)...")
    # Optimized for speed: smaller network, fewer iterations
    model = MLPClassifier(
        hidden_layer_sizes=(50,),  # Reduced from (100, 50) to single layer
        max_iter=200,              # Reduced from 500
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    return model, "MLP"

# ==========================================
# 4. Main Execution
# ==========================================

def main():
    print("Starting Champions League 2023 Full Season Prediction...")
    
    # 1. Load
    df = load_data()
    
    # 2. Prepare
    predictor_cols, target_col = prepare_features(df)
    
    le = LabelEncoder()
    le.fit(df[target_col].unique())
    print(f"Target classes: {le.classes_}")
    
    X_train, y_train, X_test, y_test, test_meta = prepare_train_test(df, predictor_cols, target_col, le)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train Models
    # Logistic Regression
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    
    # XGBoost
    xgb_model = train_xgboost(X_train_scaled, y_train)
    
    # RNN (PyTorch or MLP)
    rnn_model, rnn_type = train_rnn_or_mlp(X_train_scaled, y_train, len(le.classes_))
    
    # 4. Predict
    print("\nGenerating predictions...")
    
    # LR
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)
    
    # XGB
    xgb_preds = xgb_model.predict(X_test_scaled)
    xgb_probs = xgb_model.predict_proba(X_test_scaled)
    
    # RNN
    if rnn_type == "PyTorch":
        rnn_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            rnn_outputs = rnn_model(X_test_tensor)
            rnn_probs = torch.softmax(rnn_outputs, dim=1).numpy()
            rnn_preds = torch.argmax(rnn_outputs, dim=1).numpy()
    else: # MLP
        rnn_preds = rnn_model.predict(X_test_scaled)
        rnn_probs = rnn_model.predict_proba(X_test_scaled)
        
    # 5. Store Results
    results = test_meta.copy()
    
    # Add predictions
    results['LR_PRED'] = le.inverse_transform(lr_preds)
    results['XGB_PRED'] = le.inverse_transform(xgb_preds)
    results['RNN_PRED'] = le.inverse_transform(rnn_preds)
    
    # Add probabilities (W/D/L)
    for i, cls in enumerate(le.classes_):
        results[f'LR_PROB_{cls}'] = lr_probs[:, i]
        results[f'XGB_PROB_{cls}'] = xgb_probs[:, i]
        results[f'RNN_PROB_{cls}'] = rnn_probs[:, i]
        
    # 6. Save
    output_dir = Path("PROJECT 3/Code/Predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions_2023_full.csv"
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # 7. Visualization & Evaluation
    print(f"\n{'='*80}")
    print("CHAMPIONS LEAGUE 2023 - PREDICTED VS ACTUAL")
    print(f"{'='*80}")
    
    # Sort by date
    results = results.sort_values('ID_DATE')
    
    # Evaluation Metrics
    if 'RESULT' in results.columns and results['RESULT'].notna().any():
        valid_res = results[results['RESULT'].notna()]
        print("\nOVERALL ACCURACY:")
        print(f"  Logistic Regression: {accuracy_score(valid_res['RESULT'], valid_res['LR_PRED']):.4f}")
        print(f"  XGBoost:             {accuracy_score(valid_res['RESULT'], valid_res['XGB_PRED']):.4f}")
        print(f"  RNN ({rnn_type}):          {accuracy_score(valid_res['RESULT'], valid_res['RNN_PRED']):.4f}")
    
    # Bracket / Stage Visualization
    stages_order = ['Group Stage', 'Last 16', 'Quarter-Finals', 'Semi-Finals', 'Final']
    
    for stage in stages_order:
        stage_data = results[results['Prediction_Stage'] == stage]
        if stage_data.empty:
            continue
            
        print(f"\n>>> {stage.upper()} <<<")
        print(f"{'-'*80}")
        print(f"{'Date':<12} {'Home Team':<25} {'Away Team':<25} {'Actual':<8} {'LR':<5} {'XGB':<5} {'RNN':<5}")
        print(f"{'-'*80}")
        
        for _, row in stage_data.iterrows():
            date_str = row['ID_DATE'].strftime('%Y-%m-%d') if pd.notna(row['ID_DATE']) else "N/A"
            actual = row['RESULT'] if pd.notna(row['RESULT']) else "-"
            
            # Highlight correct predictions
            lr_mark = "✓" if actual != "-" and row['LR_PRED'] == actual else row['LR_PRED']
            xgb_mark = "✓" if actual != "-" and row['XGB_PRED'] == actual else row['XGB_PRED']
            rnn_mark = "✓" if actual != "-" and row['RNN_PRED'] == actual else row['RNN_PRED']
            
            print(f"{date_str:<12} {str(row['ID_HOME_TEAM'])[:24]:<25} {str(row['ID_AWAY_TEAM'])[:24]:<25} {actual:<8} {lr_mark:<5} {xgb_mark:<5} {rnn_mark:<5}")
            
        # Stage Accuracy
        if 'RESULT' in stage_data.columns and stage_data['RESULT'].notna().any():
            valid_stage = stage_data[stage_data['RESULT'].notna()]
            if len(valid_stage) > 0:
                acc_xgb = accuracy_score(valid_stage['RESULT'], valid_stage['XGB_PRED'])
                print(f"{'-'*80}")
                print(f"Stage Accuracy (XGBoost): {acc_xgb:.2%}")

if __name__ == "__main__":
    main()
