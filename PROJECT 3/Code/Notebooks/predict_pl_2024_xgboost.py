"""
Premier League 2024 Prediction (XGBoost Only - Rolling)

Strategy:
- Rolling Origin / Walk-Forward Validation
- Train on history -> Predict Matchweek N -> Add Matchweek N to history -> Retrain -> Predict Matchweek N+1
"""

import os
import gc
import logging
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
TARGET_COL = "RESULT"
RESULT_MAP = {"L": 0, "D": 1, "W": 2}
DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "game_features.parquet"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Predictions"

class TabularPreprocessor:
    """Handles data loading and preprocessing for tabular models."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        
    def load_and_process(self) -> pd.DataFrame:
        """Load data, handle NAs, and prepare for modeling."""
        logger.info(f"Loading data from {self.file_path}...")
        df = pd.read_parquet(self.file_path)
        
        # Sort by date
        if "ID_DATE" in df.columns:
            df["ID_DATE"] = pd.to_datetime(df["ID_DATE"])
            df = df.sort_values("ID_DATE").reset_index(drop=True)
            
        # Handle Target
        if TARGET_COL in df.columns:
            df = df[df[TARGET_COL].isin(["W", "D", "L"])].copy()
            df["target_encoded"] = df[TARGET_COL].map(RESULT_MAP)
        
        # Identify feature columns
        metadata_cols = [c for c in df.columns if c.startswith("ID_")]
        exclude_cols = metadata_cols + [TARGET_COL, "target_encoded"]
        
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        logger.info(f"Features identified: {len(self.feature_cols)} total")
        
        # Handle NAs (Simple imputation)
        df[self.feature_cols] = df[self.feature_cols].fillna(0)
        
        return df

    def fit(self, df_train: pd.DataFrame):
        """Fit scaler."""
        logger.info("Fitting scaler...")
        self.scaler.fit(df_train[self.feature_cols])
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df_scaled

class XGBoostPredictor:
    def __init__(self):
        self.params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "nthread": -1
        }
        self.model = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = []
        early_stopping_rounds = None
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dval, "val")]
            early_stopping_rounds = 20
            
        self.model = xgb.train(
            self.params, 
            dtrain, 
            num_boost_round=100,  # Reduced from 500 for speed
            evals=evals, 
            early_stopping_rounds=early_stopping_rounds, 
            verbose_eval=False
        )
        
    def predict_proba(self, X):
        return self.model.predict(xgb.DMatrix(X))

def main():
    print("="*80)
    print("PREMIER LEAGUE 2024 PREDICTION (XGBOOST ONLY)")
    print("="*80)
    
    # 1. Load Data
    prep = TabularPreprocessor(DATA_PATH)
    df = prep.load_and_process()
    
    # 2. Split History (<2024) vs 2024
    # Strategy: Train on ALL leagues (more data), predict ONLY PL 2024
    if "ID_SEASON" not in df.columns:
        df["ID_SEASON"] = df["ID_DATE"].dt.year
        
    # Training: All leagues, all seasons < 2024
    df_history = df[df["ID_SEASON"] < 2024].copy()
    
    # Testing: Premier League 2024 ONLY
    df_2024_all = df[df["ID_SEASON"] == 2024].copy()
    if "ID_COMPETITION" in df_2024_all.columns:
        df_2024 = df_2024_all[df_2024_all["ID_COMPETITION"] == "GB1"].copy()
    else:
        df_2024 = df_2024_all
    
    print(f"Training: {len(df_history)} games (all leagues, all seasons < 2024)")
    print(f"Testing: {len(df_2024)} games (Premier League 2024 only)")
    
    # 3. Initial Fit
    prep.fit(df_history)
    
    # Prepare Initial Training Data
    X_train = prep.transform(df_history)[prep.feature_cols].values
    y_train = df_history["target_encoded"].values
    
    # 4. Initial Training
    print("\nTraining Initial XGBoost Model...")
    xgb_model = XGBoostPredictor()
    xgb_model.train(X_train, y_train)
    
    # 5. Rolling Loop
    print("\nStarting Rolling Prediction Loop for 2024 (by Matchday)...")
    
    # Filter for Premier League only if possible to ensure clean matchdays
    # But assuming dataset is what it is.
    # Group by ID_ROUND if available.
    
    if "ID_ROUND" in df_2024.columns:
        # Sort rounds naturally (1. Matchday, 2. Matchday...)
        # Extract number from string "X. Matchday"
        def extract_round_num(s):
            try:
                return int(str(s).split(".")[0])
            except:
                return 999
                
        df_2024["round_num"] = df_2024["ID_ROUND"].apply(extract_round_num)
        unique_rounds = sorted(df_2024["ID_ROUND"].unique(), key=extract_round_num)
        
        # Check if we have valid rounds
        if len(unique_rounds) > 1:
            loop_keys = unique_rounds
            group_col = "ID_ROUND"
            print(f"Found {len(unique_rounds)} matchdays.")
        else:
            print("Warning: ID_ROUND not found or singular. Falling back to ID_DATE.")
            loop_keys = sorted(df_2024["ID_DATE"].unique())
            group_col = "ID_DATE"
    else:
        loop_keys = sorted(df_2024["ID_DATE"].unique())
        group_col = "ID_DATE"
    
    all_predictions = []
    
    for idx, key in enumerate(loop_keys, 1):
        batch = df_2024[df_2024[group_col] == key].copy()
        print(f"[{idx}/{len(loop_keys)}] Processing {key} ({len(batch)} games)...", flush=True)
        
        # Prepare Batch
        X_batch = prep.transform(batch)[prep.feature_cols].values
        y_batch = batch["target_encoded"].values
        
        # Predict
        preds = xgb_model.predict_proba(X_batch)
        
        # Store
        for i, idx in enumerate(batch.index):
            row = batch.loc[idx]
            res = {
                "Round": row.get("ID_ROUND", ""),
                "Date": row["ID_DATE"], "Home": row["ID_HOME_TEAM"], "Away": row["ID_AWAY_TEAM"], "Actual": row[TARGET_COL],
                "XGB_W": preds[i, 2], "XGB_D": preds[i, 1], "XGB_L": preds[i, 0]
            }
            all_predictions.append(res)
            
        # Update Training Data
        X_train = np.vstack([X_train, X_batch])
        y_train = np.concatenate([y_train, y_batch])
        
        # Retrain
        xgb_model.train(X_train, y_train)
        
    # 6. Save Results
    results_df = pd.DataFrame(all_predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "predictions_2024_xgboost.csv"
    results_df.to_csv(out_path, index=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Calculate Accuracy
    pred_probs = results_df[["XGB_L", "XGB_D", "XGB_W"]].values
    pred_class = np.argmax(pred_probs, axis=1)
    actual_class = results_df["Actual"].map(RESULT_MAP).values
    
    acc = accuracy_score(actual_class, pred_class)
    loss = log_loss(actual_class, pred_probs)
    
    print(f"XGBoost: Accuracy = {acc:.4f}, LogLoss = {loss:.4f}")
    print(f"\nSaved predictions to: {out_path}")

if __name__ == "__main__":
    main()
