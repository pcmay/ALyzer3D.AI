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
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve

# --- BioPython for PDB & Sequence Analysis ---
from Bio.PDB import PDBParser, SASA
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# --- Configuration ---
MAX_LENGTH = 120
TEST_DATA_DIR = "amyloid_data_split/test"
MODEL_DIR = "champion_v8_ensemble_modelv3" # Ensure this matches the training script's SAVE_DIR
SCALER_PATH = os.path.join(MODEL_DIR, "scalar_scaler_v8_ensemble.joblib")

PLM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
EMBEDDING_DIM = 320

# --- Feature Engineering & Data Prep Functions (Must match training script) ---

def calculate_middle_plddt(plddt_array: np.ndarray, sequence_length: int) -> float:
    """Calculates the mean pLDDT in the 50-55% region of the sequence."""
    try:
        start_index = int(sequence_length * 0.50)
        end_index = int(sequence_length * 0.55)
        if start_index >= end_index:
            return np.mean(plddt_array[start_index:]) if sequence_length > 0 else 0.0
            
        middle_region = plddt_array[start_index:end_index]
        return np.mean(middle_region) if middle_region.size > 0 else 0.0
    except Exception:
        return 0.0

def calculate_rog(pdb_path: str) -> float:
    """Calculates the Radius of Gyration (RoG) normalized by sqrt(N_residues)."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]
        atoms = list(model.get_atoms())
        if not atoms: return 0.0
        center_of_mass = sum(atom.coord for atom in atoms) / len(atoms)
        rog_sq_sum = sum(np.sum((atom.coord - center_of_mass)**2) for atom in atoms)
        rog = np.sqrt(rog_sq_sum / len(atoms))
        num_residues = len(list(model.get_residues()))
        return rog / np.sqrt(num_residues) if num_residues > 0 else 0.0
    except Exception:
        return 0.0

def calculate_sasa(pdb_path: str) -> float:
    """Calculates the average SASA per residue."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        sr = SASA.ShrakeRupley()
        sr.compute(structure, level="R")
        total_sasa, num_residues = 0, 0
        for residue in structure.get_residues():
            if "EXP_SASA" in residue.xtra:
                total_sasa += residue.xtra["EXP_SASA"]
                num_residues += 1
        return total_sasa / num_residues if num_residues > 0 else 0.0
    except Exception:
        return 0.0

def calculate_biochemical_features(sequence: str) -> dict:
    """Calculates biochemical properties from a protein sequence."""
    try:
        clean_sequence = "".join([c for c in sequence if c in "ACDEFGHIKLMNPQRSTVWY"])
        analysed_seq = ProteinAnalysis(clean_sequence)
        return {
            'pI': analysed_seq.isoelectric_point(),
            'gravy': analysed_seq.gravy(),
            'aromaticity': analysed_seq.aromaticity(),
            'mol_weight': analysed_seq.molecular_weight()
        }
    except Exception:
        return {'pI': 7.0, 'gravy': 0.0, 'aromaticity': 0.0, 'mol_weight': 12000.0}

def get_protein_embedding(sequence: str, tokenizer, plm_model, device):
    """Generates a fixed-size embedding for a protein sequence."""
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1022).to(device)
    with torch.no_grad():
        outputs = plm_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()

def load_sequence_from_pdb(pdb_file):
    """Loads a protein sequence from a PDB file."""
    try:
        parser = PDBParser(QUIET=True)
        chain = next(parser.get_structure("s", pdb_file).get_models()).get_chains().__next__()
        return "".join([protein_letters_3to1.get(r.get_resname().upper(), 'X') for r in chain.get_residues() if is_aa(r, standard=True)])
    except Exception:
        return None

def prepare_test_data(data_dir, max_length, tokenizer, plm_model, device):
    """Prepares test data for evaluation."""
    base_data = {
        "pae": [], "plddt": [], "embedding": [], "labels": [], "lengths": [],
        "pae_row": [], "pae_col": []
    }
    feature_lists = {
        "biochem": [], "struct_summary": [], "sasa": [], "advanced_struct": []
    }

    print(f"Preparing test data from {data_dir}...");
    for class_label, class_name in enumerate(["non_amyloid", "amyloid"]):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir): continue

        files_by_base = {}
        for f in os.listdir(class_dir):
            base_name = f.split('_scores_rank_')[0].split('_unrelaxed_rank_')[0].replace('.pdb', '').replace('.json', '')
            if base_name not in files_by_base: files_by_base[base_name] = {}
            if f.endswith('.pdb'): files_by_base[base_name]['pdb'] = f
            elif f.endswith('.json'): files_by_base[base_name]['json'] = f

        for base_name, files in tqdm(files_by_base.items(), desc=f"Processing {class_name}"):
            if 'pdb' not in files or 'json' not in files: continue
            pdb_path, json_path = os.path.join(class_dir, files['pdb']), os.path.join(class_dir, files['json'])
            sequence = load_sequence_from_pdb(pdb_path)
            if not sequence: continue

            with open(json_path, 'r') as f: colabfold_data = json.load(f)
            plddt, pae = np.array(colabfold_data['plddt']), np.array(colabfold_data['pae'])
            seq_len = len(sequence) # Use actual sequence length
            
            base_data["lengths"].append(seq_len)
            slice_len = min(seq_len, max_length)
            padded_pae = np.zeros((max_length, max_length)); padded_pae[:slice_len, :slice_len] = pae[:slice_len, :slice_len]
            padded_plddt = np.zeros(max_length); padded_plddt[:slice_len] = plddt[:slice_len]
            padded_pae_row = np.zeros(max_length); padded_pae_row[:slice_len] = np.mean(pae[:slice_len, :slice_len], axis=1) if slice_len > 0 else 0
            padded_pae_col = np.zeros(max_length); padded_pae_col[:slice_len] = np.mean(pae[:slice_len, :slice_len], axis=0) if slice_len > 0 else 0

            base_data["pae"].append(padded_pae)
            base_data["plddt"].append(padded_plddt)
            base_data["embedding"].append(get_protein_embedding(sequence, tokenizer, plm_model, device))
            base_data["labels"].append(class_label)
            base_data["pae_row"].append(padded_pae_row)
            base_data["pae_col"].append(padded_pae_col)
            
            feature_lists["sasa"].append(calculate_sasa(pdb_path))
            bio_feats = calculate_biochemical_features(sequence)
            feature_lists["biochem"].append([bio_feats['pI'], bio_feats['gravy'], bio_feats['aromaticity'], bio_feats['mol_weight']])
            plddt_scores = plddt[:seq_len]
            feature_lists["struct_summary"].append([
                np.mean(plddt_scores) if seq_len > 0 else 0, np.std(plddt_scores) if seq_len > 0 else 0,
                np.mean(plddt_scores < 70) if seq_len > 0 else 0, np.mean(pae[:seq_len, :seq_len]) if seq_len > 0 else 0
            ])
            # MODIFIED: Added middle_plddt to match training script
            feature_lists["advanced_struct"].append([
                calculate_middle_plddt(plddt, seq_len),
                calculate_rog(pdb_path)
            ])
            
    for key in base_data: base_data[key] = np.array(base_data[key])
    for key in feature_lists: feature_lists[key] = np.array(feature_lists[key])
    return base_data, feature_lists

def evaluate_ensemble():
    """Main evaluation function for the V8 ensemble model."""
    print(f"--- SCRIPT: Evaluating V8 Ensemble Model on Test Set ---")
    
    print(f"Loading PLM...");
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
    plm_model = EsmModel.from_pretrained(PLM_MODEL_NAME).to(device)
    plm_model.eval()
    
    base_test, features_test = prepare_test_data(TEST_DATA_DIR, MAX_LENGTH, tokenizer, plm_model, device)
    y_true = base_test["labels"]
    
    if len(y_true) == 0:
        print("No test data found.")
        return
    if not os.path.exists(MODEL_DIR) or not os.path.exists(SCALER_PATH):
        print(f"ERROR: Model directory ('{MODEL_DIR}') or scaler ('{SCALER_PATH}') not found.")
        return

    scaler = joblib.load(SCALER_PATH)
    print(f"Loaded feature scaler from {SCALER_PATH}")

    test_scalars = np.hstack([
        features_test["biochem"], features_test["struct_summary"],
        features_test["sasa"].reshape(-1, 1), features_test["advanced_struct"]
    ])
    test_scalars_scaled = scaler.transform(test_scalars)
    
    X_test_dict = {
        "pae_input": np.expand_dims(base_test["pae"], -1), "plddt_input": np.expand_dims(base_test["plddt"], -1),
        "embedding_input": base_test["embedding"], "pae_row_input": np.expand_dims(base_test["pae_row"], -1),
        "pae_col_input": np.expand_dims(base_test["pae_col"], -1), "length_input": base_test["lengths"],
        "scalar_features_input": test_scalars_scaled
    }
    
    model_paths = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
    if not model_paths:
        print(f"ERROR: No models (.keras files) found in '{MODEL_DIR}'.")
        return
        
    print(f"\nFound {len(model_paths)} models for ensembling.")
    models = [load_model(path, safe_mode=False) for path in sorted(model_paths)]
    
    all_predictions = [model.predict(X_test_dict, batch_size=64, verbose=0) for model in models]
    y_pred_ensemble_proba = np.mean(np.hstack(all_predictions), axis=1)
    
    # --- Overall Evaluation ---
    print("\n\n" + "="*20 + " FINAL ENSEMBLE EVALUATION " + "="*20)
    auc_score = roc_auc_score(y_true, y_pred_ensemble_proba)
    print(f"Overall Test Set AUC: {auc_score:.4f}\n")
    y_pred_class_05 = (y_pred_ensemble_proba > 0.5).astype("int32")
    print("Classification Report (Threshold=0.5):\n", classification_report(y_true, y_pred_class_05, target_names=["non_amyloid", "amyloid"]))
    
    # --- Plotting Section ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_ensemble_proba)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig ('ROC.png', dpi=300)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_class_05)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-Amyloid", "Amyloid"],
                yticklabels=["Non-Amyloid", "Amyloid"],
                annot_kws={"size": 14})
    plt.title(f'Confusion Matrix (Threshold = 0.5)', fontsize=14)
    plt.ylabel('True Label', fontsize=12);
    plt.xlabel('Predicted Label', fontsize=12);
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_ensemble()