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
from Bio.PDB import PDBParser, SASA
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# --- Configuration ---
MODEL_VERSION = 8 # <--- CHANGE THIS VALUE to 6, 7, 8, or 9

# --- Automatically set paths based on version ---
if MODEL_VERSION == 6:
    SAVE_DIR = "champion_v6_models"
    SCALER_FILENAME = "scalar_scaler.joblib" # V6 had a different name
elif MODEL_VERSION == 7:
    SAVE_DIR = "champion_v7_models_sasa"
    SCALER_FILENAME = f"scalar_scaler_v{MODEL_VERSION}.joblib"
elif MODEL_VERSION == 8:
    SAVE_DIR = "champion_v8_models_full_features"
    SCALER_FILENAME = f"scalar_scaler_v{MODEL_VERSION}.joblib"
elif MODEL_VERSION == 9:
    SAVE_DIR = "champion_v9_final_models"
    SCALER_FILENAME = f"scalar_scaler_v{MODEL_VERSION}.joblib"
else:
    raise ValueError("Invalid MODEL_VERSION. Choose from 6, 7, 8, or 9.")

FOLD_MODELS_DIR = SAVE_DIR
SCALER_PATH = os.path.join(FOLD_MODELS_DIR, SCALER_FILENAME)

MAX_LENGTH = 120
TEST_DATA_DIR = "/Users/PeterMay/Downloads/amyloidosis/colab_test"
PLM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
EMBEDDING_DIM = 320

# --- 1. Feature Calculation & Data Prep Functions (Master Version) ---
# [This section contains all your helper functions: calculate_longest_disordered_segment, 
# calculate_middle_plddt, calculate_rog, calculate_sasa, calculate_biochemical_features, 
# get_protein_embedding, load_sequence_from_pdb, and prepare_test_data. 
# They remain the same as the last version and are omitted here for brevity, 
# but should be included in your script.]

def calculate_longest_disordered_segment(plddt_array, sequence_length):
    if sequence_length == 0: return 0
    is_disordered = plddt_array[:sequence_length] < 70
    max_len = current_len = 0
    for val in is_disordered:
        current_len = current_len + 1 if val else 0
        max_len = max(max_len, current_len)
    return max_len

def calculate_middle_plddt(plddt_array, sequence_length):
    try:
        start, end = int(sequence_length * 0.50), int(sequence_length * 0.55)
        if start >= end: return np.mean(plddt_array[start:]) if sequence_length > 0 else 0.0
        return np.mean(plddt_array[start:end]) if plddt_array[start:end].size > 0 else 0.0
    except Exception: return 0.0

def calculate_rog(pdb_path):
    try:
        parser = PDBParser(QUIET=True)
        model = parser.get_structure("s", pdb_path)[0]
        atoms = list(model.get_atoms())
        if not atoms: return 0.0
        com = sum(a.coord for a in atoms) / len(atoms)
        rog_sq = sum(np.sum((a.coord - com)**2) for a in atoms)
        n_res = len(list(model.get_residues()))
        return np.sqrt(rog_sq / len(atoms)) / np.sqrt(n_res) if n_res > 0 else 0.0
    except Exception: return 0.0

def calculate_sasa(pdb_path):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", pdb_path)
        sr = SASA.ShrakeRupley()
        sr.compute(structure, level="R")
        total_sasa, n_res = 0, 0
        for res in structure.get_residues():
            if "EXP_SASA" in res.xtra:
                total_sasa += res.xtra["EXP_SASA"]
                n_res += 1
        return total_sasa / n_res if n_res > 0 else 0.0
    except Exception: return 0.0

def calculate_biochemical_features(sequence):
    try:
        seq = "".join(c for c in sequence if c in "ACDEFGHIKLMNPQRSTVWY")
        pa = ProteinAnalysis(seq)
        return {'pI': pa.isoelectric_point(), 'gravy': pa.gravy(), 'aromaticity': pa.aromaticity(), 'mol_weight': pa.molecular_weight()}
    except Exception: return {'pI': 7.0, 'gravy': 0.0, 'aromaticity': 0.0, 'mol_weight': 12000.0}

def get_protein_embedding(sequence, tokenizer, plm_model, device):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1022).to(device)
    with torch.no_grad(): outputs = plm_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()

def load_sequence_from_pdb(pdb_file):
    try:
        parser = PDBParser(QUIET=True)
        chain = parser.get_structure("s", pdb_file)[0].get_chains().__next__()
        return "".join(protein_letters_3to1.get(r.get_resname().upper(), 'X') for r in chain.get_residues() if is_aa(r, standard=True))
    except Exception: return None
    
def prepare_test_data(data_dir, max_length, tokenizer, plm_model, device):
    base_lists = {"pae": [], "plddt": [], "embedding": [], "labels": [], "lengths": [], "pae_row": [], "pae_col": []}
    feature_lists = {"biochem": [], "struct_summary": [], "sasa": [], "advanced_struct": [], "disorder_segment": []}
    
    print("Preparing test data and calculating all engineered features...")
    for class_label, class_name in enumerate(["non_amyloid", "amyloid"]):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir): continue
        protein_files = {}
        for f in os.listdir(class_dir):
            base_name = f.split('_scores_rank_')[0].split('_unrelaxed_rank_')[0]
            if base_name not in protein_files: protein_files[base_name] = []
            protein_files[base_name].append(f)

        for base_name, files in tqdm(protein_files.items(), desc=f"Processing {class_name}"):
            json_file = next((f for f in files if ('_rank_001' in f or '_rank_1' in f) and f.endswith('.json')), next((f for f in files if f.endswith('.json')), None))
            pdb_file = next((f for f in files if ('_rank_001' in f or '_rank_1' in f) and f.endswith('.pdb')), next((f for f in files if f.endswith('.pdb')), None))
            if not json_file or not pdb_file: continue
            
            json_path, pdb_path = os.path.join(class_dir, json_file), os.path.join(class_dir, pdb_file)
            sequence = load_sequence_from_pdb(pdb_path)
            if not sequence: continue
            
            with open(json_path, 'r') as f: colabfold_data = json.load(f)
            plddt, pae = np.array(colabfold_data['plddt']), np.array(colabfold_data['pae'])
            
            seq_len = len(sequence)
            base_lists["lengths"].append(seq_len)
            slice_len = min(len(plddt), max_length)
            
            padded_pae = np.zeros((max_length, max_length)); padded_pae[:slice_len, :slice_len] = pae[:slice_len, :slice_len]
            padded_plddt = np.zeros(max_length); padded_plddt[:slice_len] = plddt[:slice_len]
            padded_pae_row = np.zeros(max_length); padded_pae_row[:slice_len] = np.mean(pae[:slice_len, :slice_len], axis=1) if slice_len > 0 else 0
            padded_pae_col = np.zeros(max_length); padded_pae_col[:slice_len] = np.mean(pae[:slice_len, :slice_len], axis=0) if slice_len > 0 else 0
            
            base_lists["pae"].append(padded_pae); base_lists["plddt"].append(padded_plddt)
            base_lists["embedding"].append(get_protein_embedding(sequence, tokenizer, plm_model, device))
            base_lists["labels"].append(class_label)
            base_lists["pae_row"].append(padded_pae_row); base_lists["pae_col"].append(padded_pae_col)
            
            feature_lists["sasa"].append(calculate_sasa(pdb_path))
            bio_feats = calculate_biochemical_features(sequence)
            feature_lists["biochem"].append([bio_feats['pI'], bio_feats['gravy'], bio_feats['aromaticity'], bio_feats['mol_weight']])
            plddt_scores = plddt[:seq_len]
            feature_lists["struct_summary"].append([
                np.mean(plddt_scores) if seq_len > 0 else 0, np.std(plddt_scores) if seq_len > 0 else 0,
                np.mean(plddt_scores < 70) if seq_len > 0 else 0, np.mean(pae[:seq_len, :seq_len]) if seq_len > 0 else 0
            ])
            feature_lists["advanced_struct"].append([calculate_middle_plddt(plddt, seq_len), calculate_rog(pdb_path)])
            feature_lists["disorder_segment"].append(calculate_longest_disordered_segment(plddt, seq_len))

    for key in base_lists: base_lists[key] = np.array(base_lists[key])
    for key in feature_lists: feature_lists[key] = np.array(feature_lists[key])
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
    
    if len(y_true) == 0: print("No test data found. Exiting."); return
    if not os.path.exists(FOLD_MODELS_DIR): print(f"ERROR: Directory '{FOLD_MODELS_DIR}' not found."); return
    
    model_paths = [os.path.join(FOLD_MODELS_DIR, f) for f in os.listdir(FOLD_MODELS_DIR) if f.endswith('.keras')]
    if not model_paths: print(f"ERROR: No models found in '{FOLD_MODELS_DIR}'."); return
    
    if not os.path.exists(SCALER_PATH): print(f"ERROR: Scaler file not found at '{SCALER_PATH}'."); return
    scaler = joblib.load(SCALER_PATH)
    print(f"Loaded feature scaler from {SCALER_PATH}")

    # --- NEW: Automatically select the correct features based on MODEL_VERSION ---
    features_to_stack = []
    if MODEL_VERSION >= 6:
        features_to_stack.extend([features_test["biochem"], features_test["struct_summary"]])
    if MODEL_VERSION >= 7:
        features_to_stack.append(features_test["sasa"].reshape(-1, 1))
    if MODEL_VERSION >= 8:
        features_to_stack.append(features_test["advanced_struct"])
    if MODEL_VERSION >= 9:
        features_to_stack.append(features_test["disorder_segment"].reshape(-1, 1))

    scalar_features_test = np.hstack(features_to_stack)
    scalar_features_test = scaler.transform(scalar_features_test)
    
    X_test_dict = {
        "pae_input": np.expand_dims(base_test["pae"], -1),
        "plddt_input": np.expand_dims(base_test["plddt"], -1),
        "embedding_input": base_test["embedding"],
        "pae_row_input": np.expand_dims(base_test["pae_row"], -1),
        "pae_col_input": np.expand_dims(base_test["pae_col"], -1),
        "length_input": base_test["lengths"],
        "scalar_features_input": scalar_features_test
    }
    
    print(f"\nFound {len(model_paths)} models for ensembling.")
    models = [tf.keras.models.load_model(path, safe_mode=False) for path in sorted(model_paths)]
    
    all_predictions = []
    for i, model in enumerate(models):
        print(f"-> Predicting with fold {i+1} model...")
        y_pred_fold = model.predict(X_test_dict, batch_size=64, verbose=0)
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
    print(f"Accuracy: {accuracy:.4f}\nSensitivity (Recall): {sensitivity:.4f}\nSpecificity: {specificity:.4f}\n")
    print("Classification Report:\n", classification_report(y_true, y_pred_class_05, target_names=["non_amyloid", "amyloid"]))
    
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred_class_05)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["non_amyloid", "amyloid"], yticklabels=["non_amyloid", "amyloid"])
    plt.title(f'Ensemble Confusion Matrix (Threshold = 0.5)\nTest AUC = {auc_score:.3f}')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.show()

if __name__ == "__main__":
    evaluate_ensemble()