import pandas as pd
# CHANGE 1: Added precision_score to imports
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score, precision_score
from sklearn.preprocessing import LabelEncoder
import io

# --- 1. Paste Your Data Here ---
# Replace the example data below with your own entries.
# Ensure you have the 'IGK/IGL' column.
pasted_data = """
Prediction-VLAmY-Pred	Probability	Actual	IGK/IGL
Amyloid	0,822192774	Amyloid	IGK
...
Non-amyloid	0,474399526	Non-amyloid	IGK
"""

# --- 2. Specify Column Names and Positive Class ---
PREDICTION_COLUMN = 'Prediction-VLAmY-Pred'
PROBABILITY_COLUMN = 'Probability'
ACTUAL_COLUMN = 'Actual'
IG_COLUMN = 'IGK/IGL'  # Column name for the Kappa/Lambda distinction
POSITIVE_CLASS_LABEL = 'Amyloid'  # The label for the positive class

# --- 3. Run the Analysis ---
def analyze_subset(df, positive_label, subset_name="Overall"):
    """
    Calculates and prints key classification metrics for a given DataFrame (or subset).
    """
    if df.empty:
        print(f"\n--- Skipping analysis for '{subset_name}': No data available. ---")
        return

    # --- Data Preparation ---
    df = df.copy()
    df[PROBABILITY_COLUMN] = df[PROBABILITY_COLUMN].str.replace(',', '.').astype(float)

    # Use LabelEncoder
    le = LabelEncoder()
    all_labels = pd.concat([df[ACTUAL_COLUMN], df[PREDICTION_COLUMN]]).unique()
    le.fit(all_labels)

    y_true_encoded = le.transform(df[ACTUAL_COLUMN])
    y_pred_encoded = le.transform(df[PREDICTION_COLUMN])
    y_prob = df[PROBABILITY_COLUMN] 

    # Create binary true label for AUC
    y_true_for_auc = (df[ACTUAL_COLUMN] == positive_label).astype(int)

    # Get encoded labels
    try:
        positive_label_encoded = le.transform([positive_label])[0]
        negative_label_encoded = le.transform([l for l in le.classes_ if l != positive_label])[0]
    except (ValueError, IndexError):
        print(f"Warning for '{subset_name}': The positive_label '{positive_label}' or a negative label was not found.")
        return

    # --- Metric Calculations ---
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    sensitivity = recall_score(y_true_encoded, y_pred_encoded, pos_label=positive_label_encoded, zero_division=0)
    specificity = recall_score(y_true_encoded, y_pred_encoded, pos_label=negative_label_encoded, zero_division=0)
    
    # CHANGE 2: Calculate PPV (Precision)
    ppv = precision_score(y_true_encoded, y_pred_encoded, pos_label=positive_label_encoded, zero_division=0)

    # AUC Calculations
    try:
        roc_auc = roc_auc_score(y_true_for_auc, y_prob)
        pr_auc = average_precision_score(y_true_for_auc, y_prob)
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0
        print("  (Warning: Only one class present in subset, AUC set to 0)")

    # --- Print Results ---
    class_counts = df[ACTUAL_COLUMN].value_counts(normalize=True)
    print(f"\n--- Analysis for: {subset_name} ---")
    print("Class Distribution:")
    print(class_counts)
    print("\nClassification Metrics:")
    print("-----------------------")
    print(f"Total Entries: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    # CHANGE 3: Print PPV
    print(f"PPV (Precision): {ppv:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")


# --- Main Execution Logic ---
main_df = pd.read_csv(io.StringIO(pasted_data), sep='\t')

# 1. Perform Overall Analysis
analyze_subset(main_df, POSITIVE_CLASS_LABEL, "Overall")

# 2. Perform Sub-Analyses (IGK vs IGL)
if IG_COLUMN in main_df.columns:
    subsets = main_df[IG_COLUMN].unique()
    for subset_value in subsets:
        subset_df = main_df[main_df[IG_COLUMN] == subset_value]
        analyze_subset(subset_df, POSITIVE_CLASS_LABEL, f"Subset: {subset_value}")
else:
    print(f"\nWarning: The specified column '{IG_COLUMN}' for sub-analysis was not found.")