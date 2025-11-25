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
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

# --- BioPython for PDB & Sequence Analysis ---
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# --- Configuration ---
# Make sure this folder name matches your new training run
SAVE_DIR = "paper_model_scalar_pathway_v1_minus5_stripped_80_20_seed3"
FOLD_MODELS_DIR = SAVE_DIR

MAX_LENGTH = 120
TEST_DATA_DIR = "/Users/PeterMay/Downloads/amyloidosis/combined_80_20_seed3/test"
PLM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
EMBEDDING_DIM = 320

# --- 1. Feature Calculation & Data Prep Functions ---

def calculate_middle_plddt(plddt_array, sequence_length):
    """Calculates the mean pLDDT in the 50-55% region of the sequence."""
    try:
        start, end = int(sequence_length * 0.50), int(sequence_length * 0.55)
        if start >= end:
            return np.mean(plddt_array[start:]) if sequence_length > 0 else 0.0
        return np.mean(plddt_array[start:end]) if plddt_array[start:end].size > 0 else 0.0
    except Exception:
        return 0.0

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
    """Calculates biochemical properties from a protein sequence."""
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
        return {
            'pI': 7.0,
            'gravy': 0.0,
            'aromaticity': 0.0,
            'mol_weight': 12000.0
        }

def get_protein_embedding(sequence, tokenizer, plm_model, device):
    """Generates a fixed-size embedding for a protein sequence."""
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
    """Loads a protein sequence from a PDB file."""
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

def prepare_test_data(data_dir, max_length, tokenizer, plm_model, device):
    """Prepares all data, including all engineered features for the test set."""
    # --- FIX START ---
    # Add "sequences" to the dictionary to store sequence strings.
    base_lists = {
        "pae": [],
        "plddt": [],
        "embedding": [],
        "labels": [],
        "lengths": [],
        "pae_row": [],
        "pae_col": [],
        "sequences": []  # This line was missing
    }
    # --- FIX END ---
    feature_lists = {
        "biochem": [],
        "struct_summary": [],
        "advanced_struct": []  # will be RoG only, to match training
    }
    
    print("Preparing test data and calculating all engineered features...")

    for class_label, class_name in enumerate(["non_amyloid", "amyloid"]):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
    
        # Group all files by base name
        protein_files = {}
        for f in os.listdir(class_dir):
            base_name = f.split('_scores_rank_')[0].split('_unrelaxed_rank_')[0]
            protein_files.setdefault(base_name, []).append(f)
    
        for base_name, files in tqdm(protein_files.items(), desc=f"Processing {class_name}"):
            json_filename = next(
                (f for f in files
                 if ('_rank_001' in f or '_rank_1' in f) and f.endswith('.json')),
                next((f for f in files if f.endswith('.json')), None)
            )
            pdb_filename = next(
                (f for f in files
                 if ('_rank_001' in f or '_rank_1' in f) and f.endswith('.pdb')),
                next((f for f in files if f.endswith('.pdb')), None)
            )
            if not json_filename or not pdb_filename:
                continue
    
            json_path = os.path.join(class_dir, json_filename)
            pdb_path = os.path.join(class_dir, pdb_filename)
    
            sequence = load_sequence_from_pdb(pdb_path)
            if not sequence:
                continue
    
            with open(json_path, 'r') as f:
                colabfold_data = json.load(f)
            plddt = np.array(colabfold_data['plddt'])
            pae = np.array(colabfold_data['pae'])
    
            # --- SAFE LENGTH HANDLING ---
            seq_len = len(sequence)
            L_struct = min(seq_len, len(plddt), pae.shape[0], pae.shape[1])
            if L_struct <= 5:
                print(f"[SKIP] {class_name}/{base_name}: too short or mismatch "
                      f"(seq={seq_len}, plddt={len(plddt)}, pae={pae.shape})")
                continue
    
            effective_seq_len = L_struct - 5
            base_lists["lengths"].append(effective_seq_len)
            slice_len = min(effective_seq_len, max_length)
    
            plddt_effective = plddt[:effective_seq_len]
            pae_effective = pae[:effective_seq_len, :effective_seq_len]
    
            # Pad the effective (sliced) data
            padded_pae = np.zeros((max_length, max_length))
            padded_pae[:slice_len, :slice_len] = pae_effective[:slice_len, :slice_len]
    
            padded_plddt = np.zeros(max_length)
            padded_plddt[:slice_len] = plddt_effective[:slice_len]
    
            padded_pae_row = np.zeros(max_length)
            if slice_len > 0:
                padded_pae_row[:slice_len] = np.mean(
                    pae_effective[:slice_len, :slice_len], axis=1
                )
    
            padded_pae_col = np.zeros(max_length)
            if slice_len > 0:
                padded_pae_col[:slice_len] = np.mean(
                    pae_effective[:slice_len, :slice_len], axis=0
                )
    
            base_lists["sequences"].append(sequence) # This will now work
            base_lists["pae"].append(padded_pae)
            base_lists["plddt"].append(padded_plddt)
            base_lists["embedding"].append(get_protein_embedding(sequence, tokenizer, plm_model, device))
            base_lists["labels"].append(class_label)
            base_lists["pae_row"].append(padded_pae_row)
            base_lists["pae_col"].append(padded_pae_col)
    
            bio_feats = calculate_biochemical_features(sequence)
            feature_lists["biochem"].append([
                bio_feats['pI'], bio_feats['gravy'],
                bio_feats['aromaticity'], bio_feats['mol_weight']
            ])
    
            rog = calculate_rog(pdb_path)
            feature_lists["advanced_struct"].append([rog])

    # Pop 'sequences' after the loop to avoid converting it to a numpy array
    sequences = base_lists.pop("sequences")

    for key in base_lists:
        base_lists[key] = np.array(base_lists[key])
    for key in feature_lists:
        feature_lists[key] = np.array(feature_lists[key])

    # Return sequences separately if needed, otherwise just return the dictionaries
    return base_lists, feature_lists

# --- 2. Main Evaluation Loop ---

def evaluate_ensemble():
    print(f"\n--- SCRIPT: Evaluating Ensemble from '{FOLD_MODELS_DIR}' ---")

    print(f"Loading Protein Language Model: {PLM_MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
    plm_model = EsmModel.from_pretrained(PLM_MODEL_NAME).to(device)
    plm_model.eval()
    
    base_test, features_test = prepare_test_data(TEST_DATA_DIR, MAX_LENGTH, tokenizer, plm_model, device)
    y_true = base_test["labels"]
    
    if len(y_true) == 0:
        print("No test data found. Exiting.")
        return
    if not os.path.exists(FOLD_MODELS_DIR):
        print(f"ERROR: Directory '{FOLD_MODELS_DIR}' not found.")
        return
    
    model_paths = sorted(
        os.path.join(FOLD_MODELS_DIR, f)
        for f in os.listdir(FOLD_MODELS_DIR)
        if f.endswith('.keras')
    )
    scaler_paths = sorted(
        os.path.join(FOLD_MODELS_DIR, f)
        for f in os.listdir(FOLD_MODELS_DIR)
        if f.endswith('.joblib')
    )

    if not model_paths:
        print(f"ERROR: No models (.keras files) found in '{FOLD_MODELS_DIR}'.")
        return
    if not scaler_paths:
        print(f"ERROR: No scalers (.joblib files) found in '{FOLD_MODELS_DIR}'.")
        return
    if len(model_paths) != len(scaler_paths):
        print("ERROR: Mismatch between the number of models and scalers. Aborting.")
        return

    # --- Scalar features: ONLY [pI, GRAVY, aromaticity, mol_weight, RoG] ---
    scalar_features_unscaled = np.hstack([
        features_test["biochem"],          # 4 dims
        features_test["advanced_struct"]   # 1 dim (RoG)
    ])

    X_test_dict = {
        "pae_input": np.expand_dims(base_test["pae"], -1),
        "plddt_input": np.expand_dims(base_test["plddt"], -1),
        "embedding_input": base_test["embedding"],
        "pae_row_input": np.expand_dims(base_test["pae_row"], -1),
        "pae_col_input": np.expand_dims(base_test["pae_col"], -1),
        "length_input": base_test["lengths"],
        "scalar_features_input": scalar_features_unscaled  # will be scaled per fold
    }
    
    print(f"\nFound {len(model_paths)} models and {len(scaler_paths)} scalers for ensembling.")
    
    all_predictions = []
    for i, (model_path, scaler_path) in enumerate(zip(model_paths, scaler_paths)):
        print(f"-> Predicting with fold {i+1} model and scaler...")
        
        model = tf.keras.models.load_model(model_path, safe_mode=False)
        scaler = joblib.load(scaler_path)
        
        X_test_fold = X_test_dict.copy()
        X_test_fold["scalar_features_input"] = scaler.transform(scalar_features_unscaled)
        
        y_pred_fold = model.predict(X_test_fold, batch_size=64, verbose=0)
        all_predictions.append(y_pred_fold)
    
    y_pred_ensemble_proba = np.mean(np.hstack(all_predictions), axis=1)

    print("\n\n" + "="*20 + " FINAL ENSEMBLE EVALUATION " + "="*20)
    auc_score = roc_auc_score(y_true, y_pred_ensemble_proba)
    print(f"Overall Test Set AUC: {auc_score:.4f}\n")
    
    print("\n" + "="*20 + " Metrics at 0.5 Threshold " + "="*20)
    y_pred_class_05 = (y_pred_ensemble_proba > 0.5).astype("int32")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class_05).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}\n")
    print("Classification Report:\n",
          classification_report(y_true, y_pred_class_05,
                               target_names=["non_amyloid", "amyloid"]))
    
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred_class_05)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["non_amyloid", "amyloid"],
        yticklabels=["non_amyloid", "amyloid"]
    )
    plt.title(f'Ensemble Confusion Matrix (Threshold = 0.5)\nTest AUC = {auc_score:.3f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_ensemble()