import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ==========================================
# 1. Load Data
# ==========================================

def load_data():
    # Determine the path to the data file
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
        # Fallback to absolute path if relative paths fail (based on user context)
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
    # Separate metadata, target, and predictors
    metadata_cols = [col for col in df.columns if col.startswith('ID_')]
    target_col = 'RESULT'
    
    if target_col not in df.columns:
        raise ValueError("RESULT column not found in dataset!")
    
    # Get numeric predictor columns (excluding ID_ and RESULT)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictor_cols = [col for col in numeric_cols if col not in metadata_cols and col != target_col]
    
    # Fill missing values with median
    print("Filling missing values with median...")
    df[predictor_cols] = df[predictor_cols].fillna(df[predictor_cols].median())
    
    print(f"\nFeature preparation:")
    print(f"  Metadata columns: {len(metadata_cols)}")
    print(f"  Predictor features: {len(predictor_cols)}")
    
    return predictor_cols, target_col

def get_2023_schedule(df):
    # Filter to 2023 CL matches
    cl_2023 = df[
        (df['ID_COMPETITION'] == 'CL') & 
        (df['ID_SEASON'] == 2023)
    ].copy()
    
    if len(cl_2023) == 0:
        raise ValueError("No 2023 Champions League data found!")
        
    # Get unique rounds and dates
    schedule = cl_2023[['ID_DATE', 'ID_ROUND']].drop_duplicates().sort_values('ID_DATE')
    
    # Define the rounds we want to predict (in order)
    # Note: The user prompt had specific names, we match them with what's likely in the DB
    # Based on user input: Group A-H, last 16, Quarter-Finals, Semi-Finals, Final
    # We will group them by the main stage name if possible, or just iterate through unique rounds
    
    # Let's see what rounds are actually in the data
    rounds_in_data = schedule['ID_ROUND'].unique()
    print(f"\nRounds found in 2023 data: {rounds_in_data}")
    
    stage_mapping = {
        'Group A': 'Group Stage', 'Group B': 'Group Stage', 'Group C': 'Group Stage', 'Group D': 'Group Stage',
        'Group E': 'Group Stage', 'Group F': 'Group Stage', 'Group G': 'Group Stage', 'Group H': 'Group Stage',
        'last 16 1st leg': 'Last 16', 'last 16 2nd leg': 'Last 16',
        'Quarter-Finals 1st leg': 'Quarter-Finals', 'Quarter-Finals 2nd leg': 'Quarter-Finals',
        'Semi-Finals 1st Leg': 'Semi-Finals', 'Semi-Finals 2nd Leg': 'Semi-Finals',
        'Final': 'Final'
    }
    
    # Apply to main df to ensure it's available for prediction loop
    df['Prediction_Stage'] = df['ID_ROUND'].map(stage_mapping)
    
    # Update cl_2023 with the new column
    cl_2023 = df[
        (df['ID_COMPETITION'] == 'CL') & 
        (df['ID_SEASON'] == 2023)
    ].copy()
    
    # Get the start date for each of the 5 stages
    stages = ['Group Stage', 'Last 16', 'Quarter-Finals', 'Semi-Finals', 'Final']
    stage_start_dates = {}
    
    for stage in stages:
        stage_data = cl_2023[cl_2023['Prediction_Stage'] == stage]
        if not stage_data.empty:
            stage_start_dates[stage] = stage_data['ID_DATE'].min()
            
    print("\nPrediction Stages and Start Dates:")
    for stage, date in stage_start_dates.items():
        print(f"  {stage}: {date}")
        
    return stages, stage_start_dates, cl_2023

# ==========================================
# 3. Models
# ==========================================

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train, hidden_layer_sizes=(100, 50)):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=500,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    model.fit(X_train, y_train)
    return model

# ==========================================
# 4. Prediction Loop
# ==========================================

def run_predictions(df, stages, stage_start_dates, predictor_cols, target_col, le):
    all_predictions = []
    
    # Pre-calculate dates for faster filtering
    df['ID_DATE_DT'] = pd.to_datetime(df['ID_DATE'])
    
    X_all = df[predictor_cols].values
    y_all = le.transform(df[target_col])
    dates_all = df['ID_DATE_DT'].values
    
    # We need to filter by competition and season for the test set
    is_cl_2023 = (df['ID_COMPETITION'] == 'CL') & (df['ID_SEASON'] == 2023)
    
    for stage in stages:
        if stage not in stage_start_dates:
            print(f"Skipping {stage} (no start date found)")
            continue
            
        start_date = stage_start_dates[stage]
        print(f"\n{'='*60}")
        print(f"Predicting Stage: {stage}")
        print(f"Training cutoff: {start_date}")
        print(f"{'='*60}")
        
        # Define masks
        # Train: All games strictly before the stage start date
        train_mask = dates_all < np.datetime64(start_date)
        
        # Test: All games in this stage for CL 2023
        # We use the 'Prediction_Stage' column we added earlier
        test_mask = is_cl_2023 & (df['Prediction_Stage'] == stage)
        
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_mask]
        
        if len(X_test) == 0:
            print(f"No matches found for {stage}")
            continue
            
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # -----------------------------
        # Model 1: Logistic Regression
        # -----------------------------
        print("Training Logistic Regression...")
        lr_model = train_logistic_regression(X_train_scaled, y_train)
        lr_preds = lr_model.predict(X_test_scaled)
        lr_probs = lr_model.predict_proba(X_test_scaled)
        
        # -----------------------------
        # Model 2: Neural Network (MLP)
        # -----------------------------
        print("Training Neural Network (MLP)...")
        nn_model = train_neural_network(X_train_scaled, y_train)
        nn_preds = nn_model.predict(X_test_scaled)
        nn_probs = nn_model.predict_proba(X_test_scaled)
            
        # -----------------------------
        # Store Results
        # -----------------------------
        stage_indices = df.index[test_mask]
        stage_results = df.loc[stage_indices, ['ID_GAME', 'ID_DATE', 'ID_ROUND', 'ID_HOME_CLUB', 'ID_AWAY_CLUB', 'RESULT']].copy()
        
        # Add LR predictions
        stage_results['LR_PRED'] = le.inverse_transform(lr_preds)
        stage_results['LR_PROB_W'] = lr_probs[:, list(le.classes_).index('W')]
        stage_results['LR_PROB_D'] = lr_probs[:, list(le.classes_).index('D')]
        stage_results['LR_PROB_L'] = lr_probs[:, list(le.classes_).index('L')]
        
        # Add NN predictions
        stage_results['RNN_PRED'] = le.inverse_transform(nn_preds) # Keeping column name for consistency with request
        stage_results['RNN_PROB_W'] = nn_probs[:, list(le.classes_).index('W')]
        stage_results['RNN_PROB_D'] = nn_probs[:, list(le.classes_).index('D')]
        stage_results['RNN_PROB_L'] = nn_probs[:, list(le.classes_).index('L')]
        
        # Calculate accuracy for this stage
        if 'RESULT' in stage_results.columns:
            lr_acc = accuracy_score(stage_results['RESULT'], stage_results['LR_PRED'])
            nn_acc = accuracy_score(stage_results['RESULT'], stage_results['RNN_PRED'])
            print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
            print(f"Neural Network Accuracy: {nn_acc:.4f}")
            
        all_predictions.append(stage_results)
        
    return pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    print("Starting Champions League 2023 Prediction...")
    
    # Load and Prepare
    df = load_data()
    predictor_cols, target_col = prepare_features(df)
    stages, stage_start_dates, cl_2023 = get_2023_schedule(df)
    
    # Encode target
    le = LabelEncoder()
    le.fit(df[target_col].unique())
    print(f"Target classes: {le.classes_}")
    
    # Run Predictions
    results_df = run_predictions(df, stages, stage_start_dates, predictor_cols, target_col, le)
    
    if not results_df.empty:
        # Save Predictions
        output_dir = Path("PROJECT 3/Code/Predictions")
        if not output_dir.exists():
            # Try relative to current dir
            output_dir = Path("Code/Predictions")
            if not output_dir.exists():
                 # Create if doesn't exist (assuming we are in project root or close)
                 output_dir = Path("PROJECT 3/Code/Predictions")
                 output_dir.mkdir(parents=True, exist_ok=True)
                 
        output_path = output_dir / "predictions_2023.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        
        # Final Evaluation
        print(f"\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}")
        
        if 'RESULT' in results_df.columns:
            lr_acc = accuracy_score(results_df['RESULT'], results_df['LR_PRED'])
            rnn_acc = accuracy_score(results_df['RESULT'], results_df['RNN_PRED'])
            
            print(f"Overall Logistic Regression Accuracy: {lr_acc:.4f}")
            print(f"Overall RNN Accuracy: {rnn_acc:.4f}")
            
            print("\nConfusion Matrix (RNN):")
            cm = confusion_matrix(results_df['RESULT'], results_df['RNN_PRED'], labels=le.classes_)
            print(pd.DataFrame(cm, index=[f"True {c}" for c in le.classes_], columns=[f"Pred {c}" for c in le.classes_]))
    else:
        print("No predictions were generated.")

