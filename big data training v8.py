import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import gc

# --- PLM Dependencies ---
import torch
from transformers import AutoTokenizer, EsmModel

# --- Deep Learning and Data Processing Libraries ---
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- BioPython for PDB & Sequence Analysis ---
from Bio.PDB import PDBParser, SASA
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# --- Configuration ---
MAX_LENGTH = 120
BASE_DATA_DIR = "amyloid_data_split"  # Main directory for the split data
SAVE_DIR = "champion_v8_ensemble_modelv3" # Directory to save the ensemble models
N_SPLITS = 6 # Using 6 folds for the ensemble
PLM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
EMBEDDING_DIM = 320

# --- Feature Engineering & Data Prep Functions ---

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

def prepare_data_from_splits(data_dirs, max_length, tokenizer, plm_model, device):
    """Prepares all data from a list of directories."""
    base_lists = {
        "pae": [], "plddt": [], "embedding": [], "labels": [], "lengths": [],
        "pae_row": [], "pae_col": []
    }
    feature_lists = {
        "biochem": [], "struct_summary": [], "sasa": [], "advanced_struct": []
    }
    
    for data_dir in data_dirs:
        print(f"Preparing data from directory: {data_dir}")
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
                # ADDED middle_plddt to advanced_struct features
                feature_lists["advanced_struct"].append([
                    calculate_middle_plddt(plddt, seq_len),
                    calculate_rog(pdb_path)
                ])
                
    for key in base_lists: base_lists[key] = np.array(base_lists[key])
    for key in feature_lists: feature_lists[key] = np.array(feature_lists[key])
    return base_lists, feature_lists

# --- Model Architecture ---
def create_model_with_features(max_length, embedding_dim, num_scalar_features):
    L2_REG = regularizers.l2(5e-4)
    pae_input = Input(shape=(max_length, max_length, 1), name="pae_input")
    plddt_input = Input(shape=(max_length, 1), name="plddt_input")
    embedding_input = Input(shape=(embedding_dim,), name="embedding_input")
    pae_row_input = Input(shape=(max_length, 1), name="pae_row_input")
    pae_col_input = Input(shape=(max_length, 1), name="pae_col_input")
    length_input = Input(shape=(1,), name="length_input")
    scalar_features_input = Input(shape=(num_scalar_features,), name="scalar_features_input")

    normalized_length = layers.Lambda(lambda x: x / max_length)(length_input)
    x_scalar = layers.BatchNormalization()(scalar_features_input)
    x_pae = layers.Conv2D(16, (5, 5), activation='relu')(pae_input); x_pae = layers.MaxPooling2D((3, 3))(x_pae); x_pae = layers.BatchNormalization()(x_pae)
    x_pae = layers.Conv2D(32, (3, 3), activation='relu')(x_pae); x_pae = layers.MaxPooling2D((3, 3))(x_pae); x_pae = layers.Flatten()(x_pae)
    x_plddt = layers.Conv1D(16, 5, activation='relu')(plddt_input); x_plddt = layers.MaxPooling1D(3)(x_plddt); x_plddt = layers.BatchNormalization()(x_plddt); x_plddt = layers.Flatten()(x_plddt)
    x_seq = layers.Dense(64, activation='relu')(embedding_input); x_seq = layers.BatchNormalization()(x_seq)
    x_pae_row = layers.Conv1D(8, 5, activation='relu')(pae_row_input); x_pae_row = layers.MaxPooling1D(3)(x_pae_row); x_pae_row = layers.BatchNormalization()(x_pae_row); x_pae_row = layers.Flatten()(x_pae_row)
    x_pae_col = layers.Conv1D(8, 5, activation='relu')(pae_col_input); x_pae_col = layers.MaxPooling1D(3)(x_pae_col); x_pae_col = layers.BatchNormalization()(x_pae_col); x_pae_col = layers.Flatten()(x_pae_col)
    
    combined = layers.concatenate([x_pae, x_plddt, x_seq, x_pae_row, x_pae_col, normalized_length, x_scalar])
    dense = layers.Dense(64, activation='relu', kernel_regularizer=L2_REG)(combined)
    dense = layers.Dropout(0.6)(dense)
    output = layers.Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=[pae_input, plddt_input, embedding_input, pae_row_input, pae_col_input, length_input, scalar_features_input], outputs=output)
    return model

# --- Main Training & Evaluation Loop ---
def main():
    print(f"--- SCRIPT: Training V8 Ensemble Model ---")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Loading PLM: {PLM_MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
    plm_model = EsmModel.from_pretrained(PLM_MODEL_NAME).to(device)
    plm_model.eval()

    # Combine train and validation sets for cross-validation
    train_val_dirs = [os.path.join(BASE_DATA_DIR, 'train'), os.path.join(BASE_DATA_DIR, 'validation')]
    base_all, features_all = prepare_data_from_splits(train_val_dirs, MAX_LENGTH, tokenizer, plm_model, device)

    scalar_features_all = np.hstack([
        features_all["biochem"], features_all["struct_summary"],
        features_all["sasa"].reshape(-1, 1), features_all["advanced_struct"]
    ])
    
    # Fit scaler on the entire combined training data
    scaler = StandardScaler()
    scalar_features_all = scaler.fit_transform(scalar_features_all)
    scaler_path = os.path.join(SAVE_DIR, "scalar_scaler_v8_ensemble.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"\nSaved feature scaler to {scaler_path}")

    # Stratify by label and length bin for better fold distribution
    length_bins = pd.qcut(base_all["lengths"], q=5, labels=False, duplicates='drop')
    strata = [f"{label}_{bin_}" for label, bin_ in zip(base_all["labels"], length_bins)]

    # Expand dims for Conv layers
    pae_all = np.expand_dims(base_all["pae"], -1)
    plddt_all = np.expand_dims(base_all["plddt"], -1)
    pae_row_all = np.expand_dims(base_all["pae_row"], -1)
    pae_col_all = np.expand_dims(base_all["pae_col"], -1)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    for fold_no, (train_index, val_index) in enumerate(skf.split(pae_all, strata), 1):
        print(f'\n{"-"*20} FOLD {fold_no} {"-"*20}')

        # Create data dictionaries for this fold
        X_train = {
            "pae_input": pae_all[train_index], "plddt_input": plddt_all[train_index],
            "embedding_input": base_all["embedding"][train_index], "pae_row_input": pae_row_all[train_index],
            "pae_col_input": pae_col_all[train_index], "length_input": base_all["lengths"][train_index],
            "scalar_features_input": scalar_features_all[train_index]
        }
        X_val = {
            "pae_input": pae_all[val_index], "plddt_input": plddt_all[val_index],
            "embedding_input": base_all["embedding"][val_index], "pae_row_input": pae_row_all[val_index],
            "pae_col_input": pae_col_all[val_index], "length_input": base_all["lengths"][val_index],
            "scalar_features_input": scalar_features_all[val_index]
        }
        y_train, y_val = base_all["labels"][train_index], base_all["labels"][val_index]
        
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
        
        # Create and compile a new model for each fold
        model = create_model_with_features(MAX_LENGTH, EMBEDDING_DIM, scalar_features_all.shape[1])
        # ADDED Recall (Sensitivity) to the metrics
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 
                               tf.keras.metrics.AUC(name='auc'),
                               tf.keras.metrics.Recall(name='sensitivity')])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=50, restore_best_weights=True)
        
        print("Training model for this fold...")
        model.fit(X_train, y_train, batch_size=32, epochs=250, validation_data=(X_val, y_val),
                  class_weight=class_weights, callbacks=[early_stopping], verbose=2)
        
        # --- ADDED SENSITIVITY/SPECIFICITY CALCULATION ---
        scores = model.evaluate(X_val, y_val, verbose=0)
        y_pred_proba = model.predict(X_val, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype("int32")
        
        # Calculate confusion matrix to get tn, fp, fn, tp
        try:
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except ValueError: # Handles cases where a class is not predicted
            specificity = 0.0
            print("Warning: Could not calculate confusion matrix. Check predictions.")

        print(f'Score for fold {fold_no}: Accuracy={scores[1]:.4f}, AUC={scores[2]:.4f}, Sensitivity={scores[3]:.4f}, Specificity={specificity:.4f}')
        # ----------------------------------------------------
        
        model_fold_path = os.path.join(SAVE_DIR, f"amyloid_champion_v8_fold_{fold_no}.keras")
        print(f"Saving model for fold {fold_no} to {model_fold_path}")
        model.save(model_fold_path)

        # Clean up to save memory
        del model, X_train, X_val, y_train, y_val
        gc.collect()
        tf.keras.backend.clear_session()

    print("\n--- V8 ensemble training complete. All fold models have been saved. ---")
    print("The next step is to run the evaluation script on the 'test' set.")

if __name__ == "__main__":
    main()