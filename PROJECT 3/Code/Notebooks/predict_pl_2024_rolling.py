# Set env vars before ANY imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import TensorFlow FIRST to avoid DLL conflicts
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: TensorFlow import failed: {e}")
    TF_AVAILABLE = False

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
SEQ_LEN = 10  # Length of match history sequence
TARGET_COL = "RESULT"
RESULT_MAP = {"L": 0, "D": 1, "W": 2}
REVERSE_RESULT_MAP = {0: "L", 1: "D", 2: "W"}
DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "game_features.parquet"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Predictions"

class SequencePreprocessor:
    """Handles data loading, preprocessing, and sequence generation with stateful history."""
    
    def __init__(self, file_path: Path, seq_len: int = 10):
        self.file_path = file_path
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.home_cols: List[str] = []
        self.away_cols: List[str] = []
        self.diff_cols: List[str] = []
        
        # Stateful history: team_id -> list of feature vectors
        self.history: Dict[str, List[np.ndarray]] = {}
        
        # Indices for fast access
        self.home_indices: List[int] = []
        self.away_indices: List[int] = []
        self.diff_indices: List[int] = []
        
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
        
        # Identify static/diff features vs team specific
        self.home_cols = [c for c in self.feature_cols if c.startswith("home_")]
        self.away_cols = [c for c in self.feature_cols if c.startswith("away_")]
        self.diff_cols = [c for c in self.feature_cols if c.startswith("diff_")]
        
        logger.info(f"Features identified: {len(self.feature_cols)} total")
        
        # Handle NAs (Simple imputation)
        df[self.feature_cols] = df[self.feature_cols].fillna(0)
        
        return df

    def fit(self, df_train: pd.DataFrame):
        """Fit scaler and build initial history from training data."""
        logger.info("Fitting scaler and building initial history...")
        self.scaler.fit(df_train[self.feature_cols])
        
        # Pre-calculate indices relative to feature_cols (not full dataframe)
        self.home_indices = [self.feature_cols.index(c) for c in self.home_cols]
        self.away_indices = [self.feature_cols.index(c) for c in self.away_cols]
        self.diff_indices = [self.feature_cols.index(c) for c in self.diff_cols]
        
        # Build history from training data
        self._update_history(df_train)
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df_scaled
        
    def _update_history(self, df: pd.DataFrame):
        """Update team history with games from df."""
        # Scale data first
        vals = self.scaler.transform(df[self.feature_cols])
        
        for idx, row in df.iterrows():
            # Adjust idx if df is a slice (iloc vs loc issue)
            # We use enumerate on values to be safe
            pass
            
        # Faster iteration
        home_ids = df["ID_HOME_TEAM"].values
        away_ids = df["ID_AWAY_TEAM"].values
        
        for i in range(len(df)):
            h_id = home_ids[i]
            a_id = away_ids[i]
            
            # Get scaled stats
            curr_home_stats = vals[i, self.home_indices]
            curr_away_stats = vals[i, self.away_indices]
            
            if h_id not in self.history: self.history[h_id] = []
            self.history[h_id].append(curr_home_stats)
            
            if a_id not in self.history: self.history[a_id] = []
            self.history[a_id].append(curr_away_stats)

    def create_sequences(self, df: pd.DataFrame, update_history: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for RNN/TCN models.
        If update_history is True, updates history AFTER generating sequences (for rolling loop).
        """
        n_samples = len(df)
        n_features = len(self.home_cols)
        
        X_home_seq = np.zeros((n_samples, self.seq_len, n_features))
        X_away_seq = np.zeros((n_samples, self.seq_len, n_features))
        
        # Static features (diffs)
        # Need to scale them
        vals = self.scaler.transform(df[self.feature_cols])
        X_static = vals[:, self.diff_indices]
        
        y = df["target_encoded"].values if "target_encoded" in df.columns else None
        
        home_ids = df["ID_HOME_TEAM"].values
        away_ids = df["ID_AWAY_TEAM"].values
        
        for i in range(n_samples):
            h_id = home_ids[i]
            a_id = away_ids[i]
            
            # Get history (past games only)
            h_hist = self.history.get(h_id, [])
            a_hist = self.history.get(a_id, [])
            
            # Pad sequences
            if len(h_hist) < self.seq_len:
                pad = self.seq_len - len(h_hist)
                seq_h = np.concatenate([np.zeros((pad, n_features)), np.array(h_hist)]) if h_hist else np.zeros((self.seq_len, n_features))
            else:
                seq_h = np.array(h_hist[-self.seq_len:])
                
            if len(a_hist) < self.seq_len:
                pad = self.seq_len - len(a_hist)
                seq_a = np.concatenate([np.zeros((pad, n_features)), np.array(a_hist)]) if a_hist else np.zeros((self.seq_len, n_features))
            else:
                seq_a = np.array(a_hist[-self.seq_len:])
                
            X_home_seq[i] = seq_h
            X_away_seq[i] = seq_a
            
        if update_history:
            self._update_history(df)
            
        return X_home_seq, X_away_seq, X_static, y


# ==============================================================================
# MODEL CLASSES
# ==============================================================================

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
            num_boost_round=500, 
            evals=evals, 
            early_stopping_rounds=early_stopping_rounds, 
            verbose_eval=False
        )
        
    def predict_proba(self, X):
        return self.model.predict(xgb.DMatrix(X))

class FFNNPredictor:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build()
    def _build(self):
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(256, activation="relu"), layers.Dropout(0.3),
            layers.Dense(128, activation="relu"), layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(3, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
    def train(self, X, y, X_val=None, y_val=None, epochs=20):
        self.model.fit(X, y, validation_data=(X_val, y_val) if X_val is not None else None, epochs=epochs, batch_size=32, verbose=0)
    def predict_proba(self, X): return self.model.predict(X, verbose=0)

class DualTowerRNNPredictor:
    def __init__(self, seq_len, n_feat, n_static):
        self.model = self._build(seq_len, n_feat, n_static)
    def _build(self, seq_len, n_feat, n_static):
        h_in = keras.Input((seq_len, n_feat))
        a_in = keras.Input((seq_len, n_feat))
        s_in = keras.Input((n_static,))
        h_vec = layers.GRU(64)(h_in)
        a_vec = layers.GRU(64)(a_in)
        x = layers.Concatenate()([h_vec, a_vec, s_in])
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(3, activation="softmax")(x)
        model = keras.Model([h_in, a_in, s_in], out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
    def train(self, X, y, X_val=None, y_val=None, epochs=20):
        self.model.fit(X, y, validation_data=(X_val, y_val) if X_val is not None else None, epochs=epochs, batch_size=32, verbose=0)
    def predict_proba(self, X): return self.model.predict(X, verbose=0)

class TCNPredictor:
    def __init__(self, seq_len, n_feat, n_static):
        self.model = self._build(seq_len, n_feat, n_static)
    def _build(self, seq_len, n_feat, n_static):
        h_in = keras.Input((seq_len, n_feat))
        a_in = keras.Input((seq_len, n_feat))
        s_in = keras.Input((n_static,))
        def block(x):
            x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
            x = layers.GlobalAveragePooling1D()(x)
            return x
        x = layers.Concatenate()([block(h_in), block(a_in), s_in])
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(3, activation="softmax")(x)
        model = keras.Model([h_in, a_in, s_in], out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
    def train(self, X, y, X_val=None, y_val=None, epochs=20):
        self.model.fit(X, y, validation_data=(X_val, y_val) if X_val is not None else None, epochs=epochs, batch_size=32, verbose=0)
    def predict_proba(self, X): return self.model.predict(X, verbose=0)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("="*80)
    print("PREMIER LEAGUE 2024 PREDICTION SUITE (ROLLING)")
    print("="*80)
    
    # 1. Load Data
    prep = SequencePreprocessor(DATA_PATH, SEQ_LEN)
    df = prep.load_and_process()
    
    # 2. Split History (<2024) vs 2024
    # Assuming ID_SEASON is available
    if "ID_SEASON" not in df.columns:
        # Infer from date if needed, but let's assume it's there
        df["ID_SEASON"] = df["ID_DATE"].dt.year # Fallback
        
    df_history = df[df["ID_SEASON"] < 2024].copy()
    df_2024 = df[df["ID_SEASON"] == 2024].copy()
    
    print(f"History (Train): {len(df_history)} games")
    print(f"2024 (Test): {len(df_2024)} games")
    
    # 3. Initial Fit
    prep.fit(df_history)
    
    # Prepare Initial Training Data
    # Tabular
    X_train_tab = prep.transform(df_history)[prep.feature_cols].values
    y_train = df_history["target_encoded"].values
    
    # Sequence (re-create sequences from history)
    # Note: fit() already built the history, so we can just generate sequences
    # But wait, create_sequences uses the history. 
    # We need to create sequences for the training data using the history BUILT from the training data.
    # Actually, create_sequences uses PAST history.
    # So we need to rebuild history incrementally to get correct sequences for training?
    # Or just assume for training we can use the full history available up to that point?
    # For simplicity/speed in this script, we'll generate training sequences using the full history built by fit().
    # This is slightly leaky for the very first training samples (padding), but fine for bulk training.
    
    X_h_train, X_a_train, X_s_train, _ = prep.create_sequences(df_history, update_history=False)
    
    # 4. Initialize Models
    xgb_model = XGBoostPredictor()
    ffnn_model = FFNNPredictor(input_dim=X_train_tab.shape[1])
    rnn_model = DualTowerRNNPredictor(SEQ_LEN, X_h_train.shape[2], X_s_train.shape[1])
    tcn_model = TCNPredictor(SEQ_LEN, X_h_train.shape[2], X_s_train.shape[1])
    
    # 5. Initial Training
    print("\nTraining Initial Models on History (Sequential for stability)...")
    
    # Train sequentially to avoid CPU thrashing during heavy initial fit
    xgb_model = XGBoostPredictor()
    xgb_model.train(X_train_tab, y_train)
    
    ffnn_model = FFNNPredictor(X_train_tab.shape[1])
    ffnn_model.train(X_train_tab, y_train, epochs=5)
    
    rnn_model = DualTowerRNNPredictor(SEQ_LEN, X_h_train.shape[2], X_s_train.shape[1])
    rnn_model.train([X_h_train, X_a_train, X_s_train], y_train, epochs=5)
    
    tcn_model = TCNPredictor(SEQ_LEN, X_h_train.shape[2], X_s_train.shape[1])
    tcn_model.train([X_h_train, X_a_train, X_s_train], y_train, epochs=5)
    
    # Import for rolling loop
    from concurrent.futures import ThreadPoolExecutor
    
    # 6. Rolling Loop
    print("\nStarting Rolling Prediction Loop for 2024...")
    
    unique_dates = sorted(df_2024["ID_DATE"].unique())
    all_predictions = []
    
    # Optimization: Retrain every N dates/games to save time?
    # User asked for "predict 1. Matchday... then use all data... for every matchday"
    # So we must retrain (or at least update) every matchday.
    
    for date in unique_dates:
        batch = df_2024[df_2024["ID_DATE"] == date].copy()
        print(f"Processing {date.date()} ({len(batch)} games)...")
        
        X_batch_tab = prep.transform(batch)[prep.feature_cols].values
        X_h_batch, X_a_batch, X_s_batch, y_batch = prep.create_sequences(batch, update_history=True)
        
        # Predict (Parallel)
        # Prediction is fast, sequential is fine, but let's parallelize for consistency
        with ThreadPoolExecutor(max_workers=4) as executor:
            p_xgb = executor.submit(xgb_model.predict_proba, X_batch_tab)
            p_ffnn = executor.submit(ffnn_model.predict_proba, X_batch_tab)
            p_rnn = executor.submit(rnn_model.predict_proba, [X_h_batch, X_a_batch, X_s_batch])
            p_tcn = executor.submit(tcn_model.predict_proba, [X_h_batch, X_a_batch, X_s_batch])
            
            preds_xgb = p_xgb.result()
            preds_ffnn = p_ffnn.result()
            preds_rnn = p_rnn.result()
            preds_tcn = p_tcn.result()
        
        # Store
        for i, idx in enumerate(batch.index):
            row = batch.loc[idx]
            res = {
                "Date": row["ID_DATE"], "Home": row["ID_HOME_TEAM"], "Away": row["ID_AWAY_TEAM"], "Actual": row[TARGET_COL],
                "XGB_W": preds_xgb[i, 2], "XGB_D": preds_xgb[i, 1], "XGB_L": preds_xgb[i, 0],
                "FFNN_W": preds_ffnn[i, 2], "FFNN_D": preds_ffnn[i, 1], "FFNN_L": preds_ffnn[i, 0],
                "RNN_W": preds_rnn[i, 2], "RNN_D": preds_rnn[i, 1], "RNN_L": preds_rnn[i, 0],
                "TCN_W": preds_tcn[i, 2], "TCN_D": preds_tcn[i, 1], "TCN_L": preds_tcn[i, 0],
            }
            all_predictions.append(res)
            
        # Update Training Data
        X_train_tab = np.vstack([X_train_tab, X_batch_tab])
        y_train = np.concatenate([y_train, y_batch])
        X_h_train = np.vstack([X_h_train, X_h_batch])
        X_a_train = np.vstack([X_a_train, X_a_batch])
        X_s_train = np.vstack([X_s_train, X_s_batch])
        
        # Retrain (Parallel)
        # Use incremental training (1 epoch) for DL models to be fast
        def update_xgb(model, X, y):
            model.train(X, y) # XGB needs full retrain usually, or xgb.train(..., xgb_model=prev_model)
            return model
            
        def update_dl(model, X, y):
            model.train(X, y, epochs=1) # Incremental update
            return model
            
        def update_seq_dl(model, X_h, X_a, X_s, y):
            model.train([X_h, X_a, X_s], y, epochs=1)
            return model

        with ThreadPoolExecutor(max_workers=4) as executor:
            # XGB: Full retrain is safest/best for accuracy
            f_xgb = executor.submit(update_xgb, xgb_model, X_train_tab, y_train)
            f_ffnn = executor.submit(update_dl, ffnn_model, X_train_tab, y_train)
            f_rnn = executor.submit(update_seq_dl, rnn_model, X_h_train, X_a_train, X_s_train, y_train)
            f_tcn = executor.submit(update_seq_dl, tcn_model, X_h_train, X_a_train, X_s_train, y_train)
            
            # Wait for all
            f_xgb.result()
            f_ffnn.result()
            f_rnn.result()
            f_tcn.result()
            
    # 7. Save Results
        
    # 7. Save Results
    results_df = pd.DataFrame(all_predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "predictions_2024_rolling.csv"
    results_df.to_csv(out_path, index=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Calculate Accuracy
    for model in ["XGB", "FFNN", "RNN", "TCN"]:
        # Get predicted class (max prob)
        pred_probs = results_df[[f"{model}_L", f"{model}_D", f"{model}_W"]].values
        pred_class = np.argmax(pred_probs, axis=1)
        actual_class = results_df["Actual"].map(RESULT_MAP).values
        
        acc = accuracy_score(actual_class, pred_class)
        loss = log_loss(actual_class, pred_probs)
        
        print(f"{model}: Accuracy = {acc:.4f}, LogLoss = {loss:.4f}")
        
    print(f"\nSaved detailed predictions to: {out_path}")

if __name__ == "__main__":
    main()
