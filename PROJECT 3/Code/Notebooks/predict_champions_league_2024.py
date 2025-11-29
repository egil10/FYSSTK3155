"""
Champions League 2024 Prediction Script
========================================

This script implements rolling time-series cross-validation to predict 
Champions League 2024 matches round by round. At each step, we use all 
available data up to (but not including) the next round to make predictions.
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

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

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

## 3. Identify 2024 Champions League Rounds

# Filter to 2024 CL matches to get the rounds we need to predict
cl_2024 = df[
    (df['ID_COMPETITION'] == 'CL') & 
    (df['ID_SEASON'] == 2024)
].copy()

if len(cl_2024) == 0:
    raise ValueError("No 2024 Champions League data found! Check ID_COMPETITION and ID_SEASON columns.")

# Get unique rounds and dates for prediction schedule
cl_schedule = cl_2024[['ID_DATE', 'ID_SEASON', 'ID_ROUND']].drop_duplicates().sort_values('ID_DATE')
print(f"\n2024 Champions League Schedule:")
print(f"Total match dates: {len(cl_schedule)}")
print(f"\nRound breakdown:")
print(cl_schedule.groupby('ID_ROUND')['ID_DATE'].agg(['min', 'max', 'count']))

# Define prediction rounds in order
prediction_rounds = [
    "Group Stage",
    "intermediate stage 1st leg",
    "intermediate stage 2nd leg",
    "last 16 1st leg",
    "last 16 2nd leg",
    "Quarter-Finals 1st leg",
]

# Get the first date of each round
round_first_dates = {}
for round_name in prediction_rounds:
    round_matches = cl_schedule[cl_schedule['ID_ROUND'] == round_name]
    if len(round_matches) > 0:
        round_first_dates[round_name] = round_matches['ID_DATE'].min()

print(f"\nPrediction schedule (first date of each round):")
for round_name, first_date in sorted(round_first_dates.items(), key=lambda x: x[1]):
    print(f"  {round_name}: {first_date}")

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

## 5. Rolling Prediction Function

def predict_round(
    X_all, 
    y_all_encoded, 
    dates_all,
    df_all,
    round_name,
    round_first_date,
    scaler=None,
    model=None
):
    """
    Predict a specific round using all data before the round's first date.
    
    Parameters:
    -----------
    X_all : DataFrame
        All features
    y_all_encoded : array
        Encoded target for all rows
    dates_all : Series
        Dates for all rows
    round_name : str
        Name of the round to predict
    round_first_date : datetime
        First date of the round (exclusive cutoff for training)
    scaler : StandardScaler, optional
        Fitted scaler (if None, will fit new one)
    model : MLPClassifier, optional
        Trained model (if None, will train new one)
    
    Returns:
    --------
    predictions : dict
        Contains predictions, probabilities, and model info
    """
    print(f"\n{'='*70}")
    print(f"Predicting: {round_name}")
    print(f"{'='*70}")
    
    # Split: training = all data before round_first_date
    train_mask = dates_all < pd.to_datetime(round_first_date)
    test_mask = (dates_all >= pd.to_datetime(round_first_date)) & \
                (df_all['ID_ROUND'] == round_name) & \
                (df_all['ID_COMPETITION'] == 'CL') & \
                (df_all['ID_SEASON'] == 2024)
    
    X_train = X_all[train_mask].copy()
    y_train = y_all_encoded[train_mask]
    X_test = X_all[test_mask].copy()
    
    if len(X_test) == 0:
        print(f"  WARNING: No test matches found for {round_name}")
        return None
    
    print(f"  Training set: {len(X_train):,} matches (before {round_first_date.date()})")
    print(f"  Test set: {len(X_test):,} matches")
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        # Refit scaler on current training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model is None:
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
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
            n_iter_no_change=10,
            verbose=False
        )
    
    print(f"  Training model...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Get test set metadata for results
    test_metadata = df_all.loc[test_mask, ['ID_GAME', 'ID_DATE', 'ID_ROUND', 'ID_SEASON',
                                            'ID_HOME_CLUB', 'ID_AWAY_CLUB', 
                                            'ID_HOME_GOALS', 'ID_AWAY_GOALS', 'RESULT']].copy()
    test_metadata['PREDICTED_RESULT'] = le.inverse_transform(y_pred)
    test_metadata['PRED_W_PROB'] = y_pred_proba[:, list(le.classes_).index('W')]
    test_metadata['PRED_D_PROB'] = y_pred_proba[:, list(le.classes_).index('D')]
    test_metadata['PRED_L_PROB'] = y_pred_proba[:, list(le.classes_).index('L')]
    
    # Calculate accuracy if we have actual results
    if 'RESULT' in test_metadata.columns and test_metadata['RESULT'].notna().any():
        actual_results = test_metadata['RESULT'].dropna()
        predicted_results = test_metadata.loc[actual_results.index, 'PREDICTED_RESULT']
        accuracy = (actual_results == predicted_results).mean()
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show confusion matrix for available results
        if len(actual_results) > 0:
            cm = confusion_matrix(actual_results, predicted_results, labels=le.classes_)
            print(f"\n  Confusion Matrix (for {len(actual_results)} matches with results):")
            print(f"  {'':<10} {'Predicted W':<12} {'Predicted D':<12} {'Predicted L':<12}")
            for i, true_label in enumerate(le.classes_):
                print(f"  True {true_label:<6} {cm[i][0]:<12} {cm[i][1]:<12} {cm[i][2]:<12}")
    else:
        print(f"  Note: Actual results not yet available for this round")
    
    return {
        'round_name': round_name,
        'predictions': test_metadata,
        'model': model,
        'scaler': scaler,
        'X_train_scaled': X_train_scaled,
        'y_train': y_train,
        'X_test_scaled': X_test_scaled,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

## 6. Execute Rolling Predictions

# Store all prediction results
all_predictions = {}

# Get the first date we should use for training cutoff (before first CL 2024 match)
first_cl_2024_date = cl_schedule['ID_DATE'].min()
print(f"\n{'='*70}")
print(f"ROLLING PREDICTION PROCESS")
print(f"{'='*70}")
print(f"Initial training cutoff: all data before {first_cl_2024_date.date()}")

# Execute predictions for each round in order
for round_name in prediction_rounds:
    if round_name not in round_first_dates:
        print(f"\nSkipping {round_name} (not found in schedule)")
        continue
    
    round_first_date = round_first_dates[round_name]
    
    # Predict this round
    result = predict_round(
        X_all=X_all,
        y_all_encoded=y_all_encoded,
        dates_all=dates_all,
        df_all=df,
        round_name=round_name,
        round_first_date=round_first_date
    )
    
    if result is not None:
        all_predictions[round_name] = result

## 7. Aggregate and Display Results

print(f"\n{'='*70}")
print(f"PREDICTION SUMMARY")
print(f"{'='*70}")

# Combine all predictions
all_pred_df = pd.concat(
    [pred['predictions'] for pred in all_predictions.values()],
    ignore_index=True
)

print(f"\nTotal matches predicted: {len(all_pred_df)}")
print(f"\nPredictions by round:")
print(all_pred_df.groupby('ID_ROUND').size())

# Show prediction distribution
print(f"\nPredicted result distribution:")
print(all_pred_df['PREDICTED_RESULT'].value_counts())

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
        
        # Accuracy by round
        print(f"\nAccuracy by round:")
        for round_name in matches_with_results['ID_ROUND'].unique():
            round_data = matches_with_results[matches_with_results['ID_ROUND'] == round_name]
            round_accuracy = (round_data['RESULT'] == round_data['PREDICTED_RESULT']).mean()
            print(f"  {round_name}: {round_accuracy:.4f} ({round_accuracy*100:.2f}%) - {len(round_data)} matches")

## 8. Display Detailed Predictions

print(f"\n{'='*70}")
print(f"DETAILED PREDICTIONS")
print(f"{'='*70}")

# Display predictions for each round
for round_name in prediction_rounds:
    if round_name not in all_predictions:
        continue
    
    pred_data = all_predictions[round_name]['predictions'].copy()
    
    print(f"\n{round_name.upper()}")
    print(f"{'-'*70}")
    
    # Sort by date
    pred_data = pred_data.sort_values('ID_DATE')
    
    # Display key information
    display_cols = ['ID_DATE', 'ID_HOME_CLUB', 'ID_AWAY_CLUB', 'PREDICTED_RESULT', 
                    'PRED_W_PROB', 'PRED_D_PROB', 'PRED_L_PROB']
    
    # Add actual result if available
    if 'RESULT' in pred_data.columns:
        display_cols.insert(-3, 'RESULT')
        if 'ID_HOME_GOALS' in pred_data.columns and 'ID_AWAY_GOALS' in pred_data.columns:
            pred_data['SCORE'] = pred_data['ID_HOME_GOALS'].astype(str) + '-' + pred_data['ID_AWAY_GOALS'].astype(str)
            display_cols.insert(-3, 'SCORE')
    
    # Format probabilities as percentages
    pred_data_display = pred_data[display_cols].copy()
    pred_data_display['PRED_W_PROB'] = (pred_data_display['PRED_W_PROB'] * 100).round(1)
    pred_data_display['PRED_D_PROB'] = (pred_data_display['PRED_D_PROB'] * 100).round(1)
    pred_data_display['PRED_L_PROB'] = (pred_data_display['PRED_L_PROB'] * 100).round(1)
    
    # Rename columns for display
    pred_data_display.columns = [col.replace('ID_', '').replace('PRED_', '').replace('_', ' ') 
                                 for col in pred_data_display.columns]
    
    print(pred_data_display.to_string(index=False))
    
    # Show accuracy if results available
    if 'RESULT' in pred_data.columns:
        round_matches_with_results = pred_data[pred_data['RESULT'].notna()]
        if len(round_matches_with_results) > 0:
            round_acc = (round_matches_with_results['RESULT'] == 
                        round_matches_with_results['PREDICTED_RESULT']).mean()
            print(f"\nAccuracy: {round_acc:.2%} ({len(round_matches_with_results)} matches with results)")

## 9. Save Predictions to CSV

output_path = Path("../../Data/champions_league_2024_predictions.csv")
if not output_path.exists():
    output_path = Path("../Data/champions_league_2024_predictions.csv")

print(f"\n{'='*70}")
print(f"Saving predictions to: {output_path}")

# Prepare output DataFrame
output_df = all_pred_df[[
    'ID_GAME', 'ID_DATE', 'ID_ROUND', 
    'ID_HOME_CLUB', 'ID_AWAY_CLUB',
    'PREDICTED_RESULT', 'PRED_W_PROB', 'PRED_D_PROB', 'PRED_L_PROB'
]].copy()

# Add actual results if available
if 'RESULT' in all_pred_df.columns:
    output_df['ACTUAL_RESULT'] = all_pred_df['RESULT']
if 'ID_HOME_GOALS' in all_pred_df.columns and 'ID_AWAY_GOALS' in all_pred_df.columns:
    output_df['ACTUAL_SCORE'] = all_pred_df['ID_HOME_GOALS'].astype(str) + '-' + all_pred_df['ID_AWAY_GOALS'].astype(str)

# Save
output_df.to_csv(output_path, index=False)
print(f"Predictions saved successfully!")

## 10. Summary Statistics

print(f"\n{'='*70}")
print(f"FINAL SUMMARY")
print(f"{'='*70}")
print(f"Total matches predicted: {len(all_pred_df)}")
print(f"Rounds predicted: {len(all_predictions)}")

if 'RESULT' in all_pred_df.columns:
    matches_with_results = all_pred_df[all_pred_df['RESULT'].notna()]
    if len(matches_with_results) > 0:
        final_accuracy = (matches_with_results['RESULT'] == 
                         matches_with_results['PREDICTED_RESULT']).mean()
        print(f"\nOverall Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"Matches evaluated: {len(matches_with_results)} / {len(all_pred_df)}")

print(f"\nPrediction breakdown:")
print(f"  Home wins (W): {(all_pred_df['PREDICTED_RESULT'] == 'W').sum()}")
print(f"  Draws (D): {(all_pred_df['PREDICTED_RESULT'] == 'D').sum()}")
print(f"  Away wins (L): {(all_pred_df['PREDICTED_RESULT'] == 'L').sum()}")

print(f"\n{'='*70}")
print(f"Script completed successfully!")
print(f"{'='*70}")

