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
    accuracy_score  # <--- Added
)

# --- Configuration ---
TRAIN_DATA_DIR = "/Users/PeterMay/Downloads/amyloidosis/colabfold_combined_Kopie"
KAPPA_LAMBDA_CSV = "kappaorlambda.csv"
DIR_PATTERN = "paper_model_scalar_pathway_v{}_minus5_stripped"
VERSION_RANGE = range(1, 13) 
N_FOLDS = 6

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

def build_metadata_map(data_dir, lc_map):
    meta_map = {}
    print(f"Scanning {data_dir} to map sequences to True Labels...")
    for class_label, class_name in enumerate(["non_amyloid", "amyloid"]):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir): continue
        
        pdb_files = [f for f in os.listdir(class_dir) if f.endswith('.pdb')]
        for f in tqdm(pdb_files, desc=f"Mapping {class_name}"):
            pdb_path = os.path.join(class_dir, f)
            seq = load_sequence_from_pdb(pdb_path)
            if seq:
                meta_map[seq] = {
                    "label": class_label,
                    "type": lc_map.get(seq, "Other")
                }
    print(f"Metadata mapping complete. Found {len(meta_map)} unique sequences.")
    return meta_map

def calculate_metrics_for_subset(y_true, y_prob):
    """Calculates metrics for a single fold/subset. Returns dict."""
    if len(y_true) < 1: return {}
    
    # Need at least 2 classes for AUC
    try:
        auc_val = roc_auc_score(y_true, y_prob)
        pr_auc_val = average_precision_score(y_true, y_prob)
    except ValueError:
        # Happens if a fold contains only 1 class
        auc_val = np.nan
        pr_auc_val = np.nan

    # Threshold at 0.5
    y_pred = (y_prob > 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    ppv = precision_score(y_true, y_pred, zero_division=0)
    
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
    truth_map = build_metadata_map(TRAIN_DATA_DIR, lc_map)
    
    table_summary = []

    for version_num in VERSION_RANGE:
        model_dir = DIR_PATTERN.format(version_num)
        csv_path = os.path.join(model_dir, "prediction_results.csv")
        
        print(f"\nProcessing {model_dir}...")
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found. Skipping.")
            continue
            
        df = pd.read_csv(csv_path)
        
        # ---------------------------------------------------------
        # PART 1: CALCULATE TABLE METRICS (MEAN OF FOLDS)
        # ---------------------------------------------------------
        fold_stats_accumulator = []

        for fold_i in range(1, N_FOLDS + 1):
            col_name = f"fold_{fold_i}_prob"
            if col_name not in df.columns: continue
            
            fold_data = df[df[col_name].notna()]
            if fold_data.empty: continue
            
            y_true_fold = []
            y_prob_fold = []
            types_fold = []
            
            for _, row in fold_data.iterrows():
                seq = row['sequence']
                if seq in truth_map:
                    y_true_fold.append(truth_map[seq]['label'])
                    types_fold.append(truth_map[seq]['type'])
                    y_prob_fold.append(row[col_name])
            
            y_true_fold = np.array(y_true_fold)
            y_prob_fold = np.array(y_prob_fold)
            types_fold = np.array(types_fold)
            
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
        
        # Mean across Folds
        df_folds = pd.DataFrame(fold_stats_accumulator)
        mean_metrics = df_folds.mean(skipna=True).to_dict()
        mean_metrics["Version"] = f"v{version_num}"
        table_summary.append(mean_metrics)

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
                y_prob_pooled.append(row['average_prob'])
        
        y_true_pooled = np.array(y_true_pooled)
        y_prob_pooled = np.array(y_prob_pooled)
        types_pooled = np.array(types_pooled)
        
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
            axes[0].set_title(f"v{version_num} Pooled ROC (Validation)")
            axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
            axes[0].legend()
            
            axes[1].set_title(f"v{version_num} Pooled PR Curve (Validation)")
            axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
            axes[1].legend()
            
            plt.tight_layout()
            save_path = os.path.join(model_dir, f"validation_plots_mean_vs_pooled_v{version_num}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plots to {save_path}")

    # --- Final Table Output ---
    if table_summary:
        df_res = pd.DataFrame(table_summary)
        
        # Reorder columns
        core_cols = ["Version"]
        # Added Accuracy here
        metric_order = ["Accuracy", "AUC", "PR_AUC", "PPV", "Sensitivity", "Specificity"]
        sub_order = ["All", "IGK", "IGL"]
        
        final_cols = core_cols[:]
        for m in metric_order:
            for s in sub_order:
                col_name = f"{m}_{s}"
                if col_name in df_res.columns:
                    final_cols.append(col_name)
        
        df_res = df_res[final_cols]
        
        print("\n\n" + "="*40 + " TRAINING STYLE SUMMARY (MEAN OF FOLDS) " + "="*40)
        print(df_res.round(4))
        df_res.to_csv("validation_metrics_mean_of_folds.csv", index=False)
        print("\nTable saved to 'validation_metrics_mean_of_folds.csv'")

if __name__ == "__main__":
    main()