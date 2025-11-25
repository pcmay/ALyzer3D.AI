import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- PLM Dependencies ---
import torch
from transformers import AutoTokenizer, EsmModel

# --- Deep Learning and Data Processing Libraries ---
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    precision_score,
    average_precision_score, # This is PR-AUC
    roc_curve,
    precision_recall_curve,
    auc
)

# --- BioPython for PDB & Sequence Analysis ---
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# --- Configuration ---
# 1. Directory where the trained models are saved
MODEL_DIR_PATTERN = "paper_model_scalar_pathway_v1_minus5_stripped_80_20_seed{}"

# 2. Directory where the specific test data for that seed is located
TEST_DATA_PATTERN = "/Users/PeterMay/Downloads/amyloidosis/combined_80_20_seed{}/test"

# 3. Seeds to evaluate
SEED_RANGE = range(0, 12)

MAX_LENGTH = 120
KAPPA_LAMBDA_CSV = "kappaorlambda.csv" 

PLM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
EMBEDDING_DIM = 320

# --- NEW: Classification Threshold ---
CLASSIFICATION_THRESHOLD = 0.6  # <--- CHANGE THIS VALUE AS NEEDED

# --- 1. Feature Calculation & Data Prep Functions ---

def load_kappa_lambda_map(csv_path):
    """
    Loads the kappa/lambda lookup dictionary.
    Expected format: sequence,type (e.g., ...VEIN,IGK)
    """
    if not os.path.exists(csv_path):
        print(f"WARNING: Lookup file '{csv_path}' not found. Stratification will be skipped.")
        return {}
    
    try:
        df = pd.read_csv(csv_path, header=None, names=["sequence", "type"])
        df['sequence'] = df['sequence'].astype(str).str.strip()
        df['type'] = df['type'].astype(str).str.strip()
        mapping = pd.Series(df.type.values, index=df.sequence).to_dict()
        print(f"Loaded {len(mapping)} sequences from lookup file.")
        return mapping
    except Exception as e:
        print(f"Error loading lookup csv: {e}")
        return {}

def calculate_rog(pdb_path):
    """Calculates the Radius of Gyration (RoG) normalized by sqrt(N_residues)."""
    try:
        parser = PDBParser(QUIET=True)
        model = parser.get_structure("s", pdb_path)[0]
        atoms = list(model.get_atoms())
        if not atoms:
            return 0.0
        com = sum(a.coord for a in atoms) / len(atoms)
        rog_sq = sum(np.sum((a.coord - com)**2) for a in atoms)
        n_res = len(list(model.get_residues()))
        return np.sqrt(rog_sq / len(atoms)) / np.sqrt(n_res) if n_res > 0 else 0.0
    except Exception:
        return 0.0

def calculate_biochemical_features(sequence):
    try:
        seq = "".join(c for c in sequence if c in "ACDEFGHIKLMNPQRSTVWY")
        pa = ProteinAnalysis(seq)
        return {
            'pI': pa.isoelectric_point(),
            'gravy': pa.gravy(),
            'aromaticity': pa.aromaticity(),
            'mol_weight': pa.molecular_weight()
        }
    except Exception:
        return {'pI': 7.0, 'gravy': 0.0, 'aromaticity': 0.0, 'mol_weight': 12000.0}

def get_protein_embedding(sequence, tokenizer, plm_model, device):
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=1022
    ).to(device)
    with torch.no_grad():
        outputs = plm_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()

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

def prepare_test_data(data_dir, max_length, tokenizer, plm_model, device, type_mapping):
    """Prepares all data and maps sequences to Light Chain Type."""
    base_lists = {
        "pae": [], "plddt": [], "embedding": [], "labels": [],
        "lengths": [], "pae_row": [], "pae_col": [],
        "lc_type": [] 
    }
    feature_lists = {
        "biochem": [], "advanced_struct": []
    }
    
    print(f"Preparing test data from: {data_dir}")
    
    for class_label, class_name in enumerate(["non_amyloid", "amyloid"]):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        protein_files = {}
        for f in os.listdir(class_dir):
            base_name = f.split('_scores_rank_')[0].split('_unrelaxed_rank_')[0]
            if base_name not in protein_files:
                protein_files[base_name] = []
            protein_files[base_name].append(f)

        for base_name, files in tqdm(protein_files.items(), desc=f"Processing {class_name}"):
            json_file = next((f for f in files if ('_rank_001' in f or '_rank_1' in f) and f.endswith('.json')),
                             next((f for f in files if f.endswith('.json')), None))
            pdb_file = next((f for f in files if ('_rank_001' in f or '_rank_1' in f) and f.endswith('.pdb')),
                            next((f for f in files if f.endswith('.pdb')), None))
            if not json_file or not pdb_file:
                continue
            
            json_path = os.path.join(class_dir, json_file)
            pdb_path = os.path.join(class_dir, pdb_file)

            sequence = load_sequence_from_pdb(pdb_path)
            if not sequence:
                continue

            # Load PDB Data
            with open(json_path, 'r') as f:
                colabfold_data = json.load(f)
            plddt = np.array(colabfold_data['plddt'])
            pae = np.array(colabfold_data['pae'])
            
            # --- Robust Length Check ---
            seq_len = len(sequence)
            L_struct = min(seq_len, len(plddt), pae.shape[0], pae.shape[1])
            if L_struct <= 5:
                continue

            effective_seq_len = L_struct - 5
            
            plddt_effective = plddt[:effective_seq_len]
            pae_effective = pae[:effective_seq_len, :effective_seq_len]
            
            base_lists["lengths"].append(effective_seq_len)
            slice_len = min(effective_seq_len, max_length)
            
            padded_pae = np.zeros((max_length, max_length))
            padded_pae[:slice_len, :slice_len] = pae_effective[:slice_len, :slice_len]
            padded_plddt = np.zeros(max_length)
            padded_plddt[:slice_len] = plddt_effective[:slice_len]

            padded_pae_row = np.zeros(max_length)
            if slice_len > 0:
                padded_pae_row[:slice_len] = np.mean(pae_effective[:slice_len, :slice_len], axis=1) 
            padded_pae_col = np.zeros(max_length)
            if slice_len > 0:
                padded_pae_col[:slice_len] = np.mean(pae_effective[:slice_len, :slice_len], axis=0)
            
            lc_type = type_mapping.get(sequence, "Other")

            base_lists["pae"].append(padded_pae)
            base_lists["plddt"].append(padded_plddt)
            base_lists["embedding"].append(get_protein_embedding(sequence, tokenizer, plm_model, device))
            base_lists["labels"].append(class_label)
            base_lists["pae_row"].append(padded_pae_row)
            base_lists["pae_col"].append(padded_pae_col)
            base_lists["lc_type"].append(lc_type)
            
            bio_feats = calculate_biochemical_features(sequence)
            feature_lists["biochem"].append([
                bio_feats['pI'], bio_feats['gravy'], bio_feats['aromaticity'], bio_feats['mol_weight']
            ])
            feature_lists["advanced_struct"].append([calculate_rog(pdb_path)])

    for key in base_lists:
        base_lists[key] = np.array(base_lists[key])
    for key in feature_lists:
        feature_lists[key] = np.array(feature_lists[key])

    return base_lists, feature_lists

# --- 2. Evaluation Logic per Seed ---

def calculate_subgroup_metrics(y_true, y_prob, name, threshold): # <--- Added threshold argument
    """Helper to calc metrics for a specific subgroup."""
    if len(y_true) < 1:
        return None
    
    # --- 1. Threshold-Independent Metrics ---
    if len(np.unique(y_true)) < 2:
        auc_val = np.nan 
        pr_auc_val = np.nan
    else:
        auc_val = roc_auc_score(y_true, y_prob)
        pr_auc_val = average_precision_score(y_true, y_prob)

    # --- 2. Threshold-Dependent Metrics (using variable threshold) ---
    # Convert probabilities to class labels using the specific threshold
    y_pred_class = (y_prob > threshold).astype("int32")
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred_class)
    
    # PPV (Precision)
    ppv = precision_score(y_true, y_pred_class, zero_division=0)
    
    # Sensitivity & Specificity
    # labels=[0,1] ensures we get a 2x2 matrix even if one class is missing in predictions
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class, labels=[0, 1]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "n": len(y_true),
        "auc": auc_val,
        "pr_auc": pr_auc_val,
        "accuracy": acc,
        "ppv": ppv,
        "sensitivity": sensitivity,
        "specificity": specificity
    }

def evaluate_seed(seed_num, tokenizer, plm_model, device, lc_map, threshold): # <--- Added threshold argument
    """
    Evaluates a specific 80_20 Seed.
    """
    model_dir = MODEL_DIR_PATTERN.format(seed_num)
    data_dir = TEST_DATA_PATTERN.format(seed_num)
    
    print(f"\n" + "="*60)
    print(f" PROCESSING SEED {seed_num}")
    print(f" Model Dir: {model_dir}")
    print(f" Data Dir:  {data_dir}")
    print(f" Threshold: {threshold}")
    
    # 1. Check Paths
    if not os.path.exists(model_dir):
        print(f"Model directory for seed {seed_num} not found. Skipping.")
        return None
    if not os.path.exists(data_dir):
        print(f"Test data directory for seed {seed_num} not found. Skipping.")
        return None

    # 2. Load Models & Scalers
    model_paths = sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.keras')])
    scaler_paths = sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.joblib')])

    if not model_paths or len(model_paths) != len(scaler_paths):
        print("Model/Scaler mismatch or missing. Skipping.")
        return None

    # 3. Load & Prepare Data
    base_test, features_test = prepare_test_data(data_dir, MAX_LENGTH, tokenizer, plm_model, device, lc_map)
    
    if len(base_test["labels"]) == 0:
        print("No test data found for this seed. Skipping.")
        return None

    # 4. Prepare Inputs
    y_true = base_test["labels"]
    lc_types = base_test["lc_type"]
    
    scalar_features_unscaled = np.hstack([features_test["biochem"], features_test["advanced_struct"]])
    
    X_template = {
        "pae_input": np.expand_dims(base_test["pae"], -1),
        "plddt_input": np.expand_dims(base_test["plddt"], -1),
        "embedding_input": base_test["embedding"],
        "pae_row_input": np.expand_dims(base_test["pae_row"], -1),
        "pae_col_input": np.expand_dims(base_test["pae_col"], -1),
        "length_input": base_test["lengths"],
    }

    # 5. Ensemble Prediction
    all_predictions = []
    for m_path, s_path in zip(model_paths, scaler_paths):
        try:
            model = tf.keras.models.load_model(m_path, safe_mode=False)
            scaler = joblib.load(s_path)
            X_fold = X_template.copy()
            X_fold["scalar_features_input"] = scaler.transform(scalar_features_unscaled)
            all_predictions.append(model.predict(X_fold, batch_size=64, verbose=0))
        except Exception as e:
            print(f"Error evaluating model {os.path.basename(m_path)}: {e}")

    if not all_predictions:
        return None

    y_pred_ensemble_proba = np.mean(np.hstack(all_predictions), axis=1)

    # 6. Stratification & Metrics
    mask_igk = (lc_types == 'IGK')
    mask_igl = (lc_types == 'IGL')
    
    subsets = [
        ("All", np.full(len(y_true), True, dtype=bool)),
        ("IGK", mask_igk),
        ("IGL", mask_igl)
    ]

    results_row = {"Seed": seed_num}
    
    # Prepare Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"All": "black", "IGK": "blue", "IGL": "red"}
    styles = {"All": "-", "IGK": "--", "IGL": "-."}

    print(f"Stratification Counts: All={len(y_true)}, IGK={np.sum(mask_igk)}, IGL={np.sum(mask_igl)}")

    for name, mask in subsets:
        if np.sum(mask) == 0:
            continue

        y_sub_true = y_true[mask]
        y_sub_prob = y_pred_ensemble_proba[mask]
        
        # Pass the threshold down to metric calculation
        metrics = calculate_subgroup_metrics(y_sub_true, y_sub_prob, name, threshold)
        
        # Add to results dictionary
        results_row[f"Accuracy_{name}"] = metrics["accuracy"]
        results_row[f"AUC_{name}"] = metrics["auc"]
        results_row[f"PR_AUC_{name}"] = metrics["pr_auc"]
        results_row[f"PPV_{name}"] = metrics["ppv"]
        results_row[f"Sensitivity_{name}"] = metrics["sensitivity"]
        results_row[f"Specificity_{name}"] = metrics["specificity"]
        
        # --- Plotting Logic ---
        if len(np.unique(y_sub_true)) == 2:
            # ROC
            fpr, tpr, _ = roc_curve(y_sub_true, y_sub_prob)
            axes[0].plot(fpr, tpr, color=colors[name], linestyle=styles[name], lw=2, 
                         label=f'{name} (AUC={metrics["auc"]:.3f})')
            # PR
            prec, rec, _ = precision_recall_curve(y_sub_true, y_sub_prob)
            axes[1].plot(rec, prec, color=colors[name], linestyle=styles[name], lw=2, 
                         label=f'{name} (PR={metrics["pr_auc"]:.3f})')

    # Finalize Plots
    axes[0].plot([0, 1], [0, 1], 'k:', alpha=0.5)
    axes[0].set_title(f"Seed {seed_num} ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f"Seed {seed_num} Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(model_dir, f"evaluation_stratified_seed{seed_num}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Stratified plots saved to: {save_path}")

    return results_row

# --- 3. Main Execution ---

def main():
    print(f"Loading PLM: {PLM_MODEL_NAME}")
    print(f"Classification Threshold: {CLASSIFICATION_THRESHOLD}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
    plm_model = EsmModel.from_pretrained(PLM_MODEL_NAME).to(device)
    plm_model.eval()
    
    # Load Lookup
    lc_map = load_kappa_lambda_map(KAPPA_LAMBDA_CSV)

    results_summary = []
    
    # Iterate over Seeds
    for seed in SEED_RANGE:
        # Pass CLASSIFICATION_THRESHOLD here
        res = evaluate_seed(seed, tokenizer, plm_model, device, lc_map, CLASSIFICATION_THRESHOLD)
        if res:
            results_summary.append(res)

    if results_summary:
        df = pd.DataFrame(results_summary)
        
        # Reorder columns nicely
        core_cols = ["Seed"]
        metric_types = ["Accuracy", "AUC", "PR_AUC", "PPV", "Sensitivity", "Specificity"]
        sub_groups = ["All", "IGK", "IGL"]
        
        final_cols = core_cols[:]
        for m in metric_types:
            for s in sub_groups:
                col_name = f"{m}_{s}"
                if col_name in df.columns:
                    final_cols.append(col_name)
        
        df = df[final_cols]
        
        print("\n\n" + "="*30 + " 80-20 SEED STRATIFIED SUMMARY " + "="*30)
        print(df.round(4))
        
        output_csv = f"stratified_metrics_80_20_seeds_summary_thresh_{CLASSIFICATION_THRESHOLD}.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nSaved to '{output_csv}'")

if __name__ == "__main__":
    main()