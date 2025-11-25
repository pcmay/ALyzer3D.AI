import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- BioPython for Sequence Extraction ---
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1

# --- Metrics ---
from sklearn.metrics import (
    roc_auc_score, 
    precision_score,
    average_precision_score, 
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score
)

# --- Configuration ---

# 1. Pattern for the Training Data Directory (changes per seed)
# We scan this to find the True Labels (Amyloid vs Non-Amyloid) for the sequences in the CSV
DATA_DIR_PATTERN = "/Users/PeterMay/Downloads/amyloidosis/combined_80_20_seed{}" 

# 2. Pattern for the Saved Model Results
MODEL_DIR_PATTERN = "paper_model_scalar_pathway_v1_minus5_stripped_80_20_seed{}"

KAPPA_LAMBDA_CSV = "kappaorlambda.csv"
SEED_RANGE = range(0, 13)  # Seeds 1 to 10
N_FOLDS = 6 # Usually 5-fold CV is done within the 80% training set

# --- 1. Helper Functions ---

def load_sequence_from_pdb(pdb_file):
    try:
        parser = PDBParser(QUIET=True)
        chain = parser.get_structure("s", pdb_file)[0].get_chains().__next__()
        return "".join(
            protein_letters_3to1.get(r.get_resname().upper(), 'X')
            for r in chain.get_residues()
            if is_aa(r, standard=True)
        )
    except Exception:
        return None

def load_kappa_lambda_map(csv_path):
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path, header=None, names=["sequence", "type"])
        df['sequence'] = df['sequence'].astype(str).str.strip()
        df['type'] = df['type'].astype(str).str.strip()
        return pd.Series(df.type.values, index=df.sequence).to_dict()
    except:
        return {}

def build_metadata_map_for_seed(seed_root_dir, lc_map):
    """
    Scans both 'train' and 'test' folders inside the seed directory 
    to create a complete map of Sequence -> True Label.
    """
    meta_map = {}
    if not os.path.exists(seed_root_dir):
        print(f"Data directory not found: {seed_root_dir}")
        return meta_map

    print(f"Scanning {seed_root_dir} to map sequences to True Labels...")
    
    # Check both train and test subfolders to ensure we find all sequences
    subsets = ['train', 'test']
    
    for subset in subsets:
        subset_path = os.path.join(seed_root_dir, subset)
        if not os.path.isdir(subset_path):
            continue
            
        for class_label, class_name in enumerate(["non_amyloid", "amyloid"]):
            class_dir = os.path.join(subset_path, class_name)
            if not os.path.isdir(class_dir): 
                continue
            
            pdb_files = [f for f in os.listdir(class_dir) if f.endswith('.pdb')]
            # Only process if we haven't mapped this file's sequence yet (optimization)
            # However, filenames might differ, so we scan sequence content.
            
            for f in pdb_files:
                pdb_path = os.path.join(class_dir, f)
                seq = load_sequence_from_pdb(pdb_path)
                if seq and seq not in meta_map:
                    meta_map[seq] = {
                        "label": class_label,
                        "type": lc_map.get(seq, "Other")
                    }
                    
    print(f"Metadata mapping complete for Seed. Found {len(meta_map)} unique sequences.")
    return meta_map

def calculate_metrics_for_subset(y_true, y_prob):
    """Calculates metrics for a single fold/subset. Returns dict."""
    if len(y_true) < 1: return {}
    
    # Need at least 2 classes for AUC
    try:
        if len(np.unique(y_true)) > 1:
            auc_val = roc_auc_score(y_true, y_prob)
            pr_auc_val = average_precision_score(y_true, y_prob)
        else:
            auc_val = np.nan
            pr_auc_val = np.nan
    except ValueError:
        auc_val = np.nan
        pr_auc_val = np.nan

    # Threshold at 0.5
    y_pred = (y_prob > 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    ppv = precision_score(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix for Sens/Spec
    # labels=[0,1] ensures shape is 2x2 even if a class is missing
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "Accuracy": acc,
        "AUC": auc_val,
        "PR_AUC": pr_auc_val,
        "PPV": ppv,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }

# --- 2. Main Loop ---

def main():
    lc_map = load_kappa_lambda_map(KAPPA_LAMBDA_CSV)
    table_summary = []

    for seed_num in SEED_RANGE:
        # Define paths for this specific seed
        model_dir = MODEL_DIR_PATTERN.format(seed_num)
        data_dir = DATA_DIR_PATTERN.format(seed_num)
        csv_path = os.path.join(model_dir, "prediction_results.csv")
        
        print(f"\n" + "="*50)
        print(f"Processing Seed {seed_num}")
        print(f"Model Dir: {model_dir}")
        
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found. Skipping Seed {seed_num}.")
            continue
            
        # Build Metadata Map specific to this seed's data split
        truth_map = build_metadata_map_for_seed(data_dir, lc_map)
        if not truth_map:
            print("Skipping due to missing data/mapping.")
            continue

        df = pd.read_csv(csv_path)
        
        # ---------------------------------------------------------
        # PART 1: CALCULATE TABLE METRICS (MEAN OF FOLDS)
        # ---------------------------------------------------------
        fold_stats_accumulator = []

        for fold_i in range(1, N_FOLDS + 1):
            col_name = f"fold_{fold_i}_prob"
            if col_name not in df.columns: continue
            
            # Filter rows that have a prediction for this fold
            fold_data = df[df[col_name].notna()]
            if fold_data.empty: continue
            
            y_true_fold = []
            y_prob_fold = []
            types_fold = []
            
            for _, row in fold_data.iterrows():
                seq = row['sequence']
                # Retrieve truth from map
                if seq in truth_map:
                    y_true_fold.append(truth_map[seq]['label'])
                    types_fold.append(truth_map[seq]['type'])
                    y_prob_fold.append(row[col_name])
            
            y_true_fold = np.array(y_true_fold)
            y_prob_fold = np.array(y_prob_fold)
            types_fold = np.array(types_fold)
            
            if len(y_true_fold) == 0:
                continue

            subsets = {
                "All": np.full(len(y_true_fold), True),
                "IGK": (types_fold == 'IGK'),
                "IGL": (types_fold == 'IGL')
            }
            
            fold_res = {}
            for name, mask in subsets.items():
                if np.sum(mask) > 0:
                    m = calculate_metrics_for_subset(y_true_fold[mask], y_prob_fold[mask])
                    for k, v in m.items():
                        fold_res[f"{k}_{name}"] = v
            
            fold_stats_accumulator.append(fold_res)
        
        # Mean across Folds for this Seed
        if fold_stats_accumulator:
            df_folds = pd.DataFrame(fold_stats_accumulator)
            mean_metrics = df_folds.mean(skipna=True).to_dict()
            mean_metrics["Seed"] = seed_num
            table_summary.append(mean_metrics)
        else:
            print(f"No fold data processed for Seed {seed_num}")

        # ---------------------------------------------------------
        # PART 2: GENERATE PLOTS (POOLED PREDICTIONS)
        # ---------------------------------------------------------
        y_true_pooled = []
        y_prob_pooled = []
        types_pooled = []
        
        for _, row in df.iterrows():
            seq = row['sequence']
            if seq in truth_map:
                y_true_pooled.append(truth_map[seq]['label'])
                types_pooled.append(truth_map[seq]['type'])
                # Use average_prob which is usually the ensemble of CV models
                if 'average_prob' in row:
                    y_prob_pooled.append(row['average_prob'])
                else:
                    # Fallback if average_prob missing: mean of existing fold cols
                    fold_cols = [c for c in df.columns if 'fold_' in c and '_prob' in c]
                    vals = row[fold_cols].values.astype(float)
                    y_prob_pooled.append(np.nanmean(vals))
        
        y_true_pooled = np.array(y_true_pooled)
        y_prob_pooled = np.array(y_prob_pooled)
        types_pooled = np.array(types_pooled)
        
        if len(y_true_pooled) == 0:
            continue

        # Plot Logic
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = {"All": "black", "IGK": "blue", "IGL": "red"}
        styles = {"All": "-", "IGK": "--", "IGL": "-."}
        
        subsets_pooled = [
            ("All", np.full(len(y_true_pooled), True)),
            ("IGK", types_pooled == 'IGK'),
            ("IGL", types_pooled == 'IGL')
        ]
        
        has_plot_data = False
        for name, mask in subsets_pooled:
            if np.sum(mask) == 0: continue
            
            # Check classes
            if len(np.unique(y_true_pooled[mask])) < 2:
                continue

            sub_auc = roc_auc_score(y_true_pooled[mask], y_prob_pooled[mask])
            sub_pr = average_precision_score(y_true_pooled[mask], y_prob_pooled[mask])
            has_plot_data = True
            
            # ROC
            fpr, tpr, _ = roc_curve(y_true_pooled[mask], y_prob_pooled[mask])
            axes[0].plot(fpr, tpr, color=colors[name], linestyle=styles[name], 
                         label=f'{name} (Pooled AUC={sub_auc:.3f})')
            
            # PR
            prec, rec, _ = precision_recall_curve(y_true_pooled[mask], y_prob_pooled[mask])
            axes[1].plot(rec, prec, color=colors[name], linestyle=styles[name], 
                         label=f'{name} (Pooled PR={sub_pr:.3f})')

        if has_plot_data:
            axes[0].plot([0, 1], [0, 1], 'k:', alpha=0.5)
            axes[0].set_title(f"Seed {seed_num} Pooled ROC (Validation)")
            axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
            axes[0].legend()
            
            axes[1].set_title(f"Seed {seed_num} Pooled PR Curve (Validation)")
            axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
            axes[1].legend()
            
            plt.tight_layout()
            save_path = os.path.join(model_dir, f"validation_plots_mean_vs_pooled_seed{seed_num}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plots to {save_path}")

    # --- Final Table Output ---
    if table_summary:
        df_res = pd.DataFrame(table_summary)
        
        # Reorder columns
        core_cols = ["Seed"]
        metric_order = ["Accuracy", "AUC", "PR_AUC", "PPV", "Sensitivity", "Specificity"]
        sub_order = ["All", "IGK", "IGL"]
        
        final_cols = core_cols[:]
        for m in metric_order:
            for s in sub_order:
                col_name = f"{m}_{s}"
                if col_name in df_res.columns:
                    final_cols.append(col_name)
        
        df_res = df_res[final_cols]
        
        print("\n\n" + "="*40 + " SEED VALIDATION SUMMARY (MEAN OF FOLDS) " + "="*40)
        print(df_res.round(4))
        df_res.to_csv("validation_metrics_80_20_seeds_mean_of_folds.csv", index=False)
        print("\nTable saved to 'validation_metrics_80_20_seeds_mean_of_folds.csv'")

if __name__ == "__main__":
    main()