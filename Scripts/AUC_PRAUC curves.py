import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import gc 

# --- Force CPU to prevent TF/Torch Memory Conflicts ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

# --- PLM Dependencies ---
import torch
from transformers import AutoTokenizer, EsmModel

# --- Deep Learning ---
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    confusion_matrix, 
    precision_recall_curve, 
    average_precision_score
)

# --- BioPython ---
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm

# --- Configuration for SEED 3 ---
SEED_NUM = 3
MODEL_DIR = f"paper_model_scalar_pathway_v1_minus5_stripped_80_20_seed{SEED_NUM}"
TEST_DATA_DIR = f"/Users/PeterMay/Downloads/amyloidosis/combined_80_20_seed{SEED_NUM}/test"

MAX_LENGTH = 120
PLM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# --- Styling Constants ---
FONT_SIZE_LABEL = 20  # Increased
FONT_SIZE_TICK = 16   # Increased
FONT_SIZE_LEGEND = 14 # Increased
MARKER_SIZE = 180     # Increased slightly

CURVE_COLOR = 'orange'
SYMBOL_COLOR = 'darkblue'

# --- Helper Functions (Data Prep) ---

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
    except: return 0.0

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
    except:
        return {'pI': 7.0, 'gravy': 0.0, 'aromaticity': 0.0, 'mol_weight': 12000.0}

def get_protein_embedding(sequence, tokenizer, plm_model, device):
    inputs = tokenizer(sequence[:1022], return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = plm_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()

def load_sequence_from_pdb(pdb_file):
    try:
        parser = PDBParser(QUIET=True)
        chain = parser.get_structure("s", pdb_file)[0].get_chains().__next__()
        return "".join(protein_letters_3to1.get(r.get_resname().upper(), 'X') for r in chain.get_residues() if is_aa(r, standard=True))
    except: return None

def prepare_test_data(data_dir, max_length, tokenizer, plm_model, device):
    base_lists = { "pae": [], "plddt": [], "embedding": [], "labels": [], "lengths": [], "pae_row": [], "pae_col": [] }
    feature_lists = { "biochem": [], "advanced_struct": [] }

    print(f"Loading Test Data from: {data_dir}")
    for class_label, class_name in enumerate(["non_amyloid", "amyloid"]):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir): continue

        protein_files = {}
        for f in os.listdir(class_dir):
            base = f.split('_scores_rank_')[0].split('_unrelaxed_rank_')[0]
            protein_files.setdefault(base, []).append(f)

        for base, files in tqdm(protein_files.items(), desc=class_name):
            json_f = next((f for f in files if f.endswith('.json')), None)
            pdb_f = next((f for f in files if f.endswith('.pdb')), None)
            if not json_f or not pdb_f: continue
            
            pdb_full_path = os.path.join(class_dir, pdb_f)
            seq = load_sequence_from_pdb(pdb_full_path)
            if not seq: continue
            
            with open(os.path.join(class_dir, json_f), 'r') as f: d = json.load(f)
            plddt, pae = np.array(d['plddt']), np.array(d['pae'])
            
            L = min(len(seq), len(plddt), pae.shape[0])
            if L <= 5: continue
            eff_len = L - 5
            
            emb = get_protein_embedding(seq, tokenizer, plm_model, device)
            
            slice_len = min(eff_len, max_length)
            padded_pae = np.zeros((max_length, max_length), dtype=np.float32)
            padded_pae[:slice_len, :slice_len] = pae[:slice_len, :slice_len]
            padded_plddt = np.zeros(max_length, dtype=np.float32)
            padded_plddt[:slice_len] = plddt[:slice_len]
            
            padded_row = np.zeros(max_length, dtype=np.float32)
            padded_col = np.zeros(max_length, dtype=np.float32)
            if slice_len > 0:
                padded_row[:slice_len] = np.mean(pae[:slice_len, :slice_len], axis=1)
                padded_col[:slice_len] = np.mean(pae[:slice_len, :slice_len], axis=0)

            bio = calculate_biochemical_features(seq)
            rog = calculate_rog(pdb_full_path)

            base_lists["pae"].append(padded_pae)
            base_lists["plddt"].append(padded_plddt)
            base_lists["embedding"].append(emb)
            base_lists["labels"].append(class_label)
            base_lists["lengths"].append(eff_len)
            base_lists["pae_row"].append(padded_row)
            base_lists["pae_col"].append(padded_col)
            feature_lists["biochem"].append([bio['pI'], bio['gravy'], bio['aromaticity'], bio['mol_weight']])
            feature_lists["advanced_struct"].append([rog])

    for k in base_lists: base_lists[k] = np.array(base_lists[k])
    for k in feature_lists: feature_lists[k] = np.array(feature_lists[k])
    return base_lists, feature_lists

# --- Main Evaluation ---

def main():
    gc.collect()
    
    # 1. Setup PLM
    print("Initializing PLM on CPU...")
    device = torch.device("cpu") 
    tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
    plm_model = EsmModel.from_pretrained(PLM_MODEL_NAME).to(device)
    plm_model.eval()

    # 2. Load Data
    base, feats = prepare_test_data(TEST_DATA_DIR, MAX_LENGTH, tokenizer, plm_model, device)
    if len(base["labels"]) == 0:
        print("No data found.")
        return

    del plm_model
    del tokenizer
    gc.collect()

    # 3. Predict
    model_paths = sorted([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith('.keras')])
    scaler_paths = sorted([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')])
    scalars_raw = np.hstack([feats["biochem"], feats["advanced_struct"]])
    
    all_preds = []
    print("\nRunning Ensemble Predictions...")
    
    X_dict = {
        "pae_input": np.expand_dims(base["pae"], -1),
        "plddt_input": np.expand_dims(base["plddt"], -1),
        "embedding_input": base["embedding"],
        "pae_row_input": np.expand_dims(base["pae_row"], -1),
        "pae_col_input": np.expand_dims(base["pae_col"], -1),
        "length_input": base["lengths"],
    }
    
    for i, (m_path, s_path) in enumerate(zip(model_paths, scaler_paths)):
        scaler = joblib.load(s_path)
        X_current = X_dict.copy()
        X_current["scalar_features_input"] = scaler.transform(scalars_raw)
        
        try:
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(m_path, safe_mode=False)
            preds = model.predict(X_current, verbose=0, batch_size=32)
            all_preds.append(preds)
            del model
            gc.collect()
        except Exception as e:
            print(f"Error with model {m_path}: {e}")

    if not all_preds: return

    y_prob = np.mean(np.hstack(all_preds), axis=1)
    y_true = base["labels"]

    # --- METRICS CALCULATION ---
    auc_val = roc_auc_score(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    baseline_prevalence = np.sum(y_true) / len(y_true)

    thresholds_to_plot = [0.1, 0.5, 0.9]
    markers = ['o', 's', '^']

    # ==========================================
    # 1. PLOT ROC CURVE (Separate File)
    # ==========================================
    plt.figure(figsize=(9, 8))
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, color=CURVE_COLOR, lw=4, label=f'ROC (AUC = {auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    
    # Thresholds
    for thr, mark in zip(thresholds_to_plot, markers):
        y_pred_binary = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        x, y = (1 - spec), sens
        
        plt.scatter(x, y, color=SYMBOL_COLOR, s=MARKER_SIZE, marker=mark, zorder=5, 
                    label=f't={thr} (Sens={sens:.2f}, Spec={spec:.2f})')
        plt.annotate(f"t={thr}", (x, y), xytext=(10, -15), textcoords='offset points', 
                     fontsize=14, color=SYMBOL_COLOR, fontweight='bold')

    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=FONT_SIZE_LABEL)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    plt.legend(loc="lower right", fontsize=FONT_SIZE_LEGEND)
    plt.grid(False)
    
    save_path_roc = os.path.join(MODEL_DIR, f"eval_ROC_seed{SEED_NUM}.png")
    plt.tight_layout()
    plt.savefig(save_path_roc, dpi=600)
    plt.close() # Close to free memory/prevent overlapping
    print(f"ROC Plot saved to: {save_path_roc}")

    # ==========================================
    # 2. PLOT PR CURVE (Separate File)
    # ==========================================
    plt.figure(figsize=(9, 8))
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall_curve, precision_curve, color=CURVE_COLOR, lw=4, label=f'PR (AP = {pr_auc_val:.2f})')
    
    # Horizontal Baseline
    plt.plot([0, 1], [baseline_prevalence, baseline_prevalence], 
             color='gray', linestyle='--', lw=2, label=f'Baseline ({baseline_prevalence:.2f})')

    # Thresholds
    for thr, mark in zip(thresholds_to_plot, markers):
        y_pred_binary = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        x, y = sens, prec
        
        plt.scatter(x, y, color=SYMBOL_COLOR, s=MARKER_SIZE, marker=mark, zorder=5,
                    label=f't={thr} (Prec={prec:.2f}, Rec={sens:.2f})')
        plt.annotate(f"t={thr}", (x, y), xytext=(10, 5), textcoords='offset points', 
                     fontsize=14, color=SYMBOL_COLOR, fontweight='bold')

    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('Recall (Sensitivity)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Precision', fontsize=FONT_SIZE_LABEL)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    plt.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    plt.grid(False)
    
    save_path_pr = os.path.join(MODEL_DIR, f"eval_PR_seed{SEED_NUM}.png")
    plt.tight_layout()
    plt.savefig(save_path_pr, dpi=600)
    plt.close()
    print(f"PR Plot saved to: {save_path_pr}")

if __name__ == "__main__":
    main()