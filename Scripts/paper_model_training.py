import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

# --- PLM Dependencies ---
import torch
from transformers import AutoTokenizer, EsmModel

# --- Deep Learning and Data Processing Libraries ---
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- BioPython for PDB & Sequence Analysis ---
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# --- Configuration ---
MAX_LENGTH = 120
DATA_DIR = "/Users/PeterMay/Downloads/amyloidosis/colabfold_combined_Kopie" # Corrected Path
SAVE_DIR = "paper_model_scalar_pathway_v1_minus5_stripped"  # Directory for this new run
N_SPLITS = 6
PLM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
EMBEDDING_DIM = 320


# --- 1. Feature Engineering & Data Prep Functions ---

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
        structure = parser.get_structure("protein", pdb_file)
        chain = next(structure.get_models()).get_chains().__next__()
        return "".join([protein_letters_3to1.get(r.get_resname().upper(), 'X') for r in chain.get_residues() if is_aa(r, standard=True) and protein_letters_3to1.get(r.get_resname().upper(), 'X') != 'X'])
    except Exception:
        return None

def prepare_data(data_dir, max_length, tokenizer, plm_model, device):
    """Prepares all data, including all engineered features."""
    base_lists = {
        "pae": [], "plddt": [], "embedding": [], "labels": [], "lengths": [],
        "pae_row": [], "pae_col": [], "sequences": []
    }
    feature_lists = {
        "biochem": [], "struct_summary": [], "advanced_struct": []
    }
    
    print("Preparing data and engineering all new features...")
    for class_label, class_name in enumerate(["non_amyloid", "amyloid"]):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir): continue
        json_files = [f for f in os.listdir(class_dir) if f.endswith('.json')]
        
        for json_filename in tqdm(json_files, desc=f"Processing {class_name}"):
            json_base_name = json_filename.split('_scores')[0]
            pdb_file = next((f for f in os.listdir(class_dir) if f.startswith(json_base_name) and f.endswith('.pdb')), None)
            if not pdb_file: continue

            json_path, pdb_path = os.path.join(class_dir, json_filename), os.path.join(class_dir, pdb_file)
            sequence = load_sequence_from_pdb(pdb_path)
            if not sequence: continue

            with open(json_path, 'r') as f:
                colabfold_data = json.load(f)
            plddt, pae = np.array(colabfold_data['plddt']), np.array(colabfold_data['pae'])
            
            seq_len = len(sequence)
            
            # --- MODIFICATION START ---
            # Calculate an effective sequence length, ignoring the last 5 residues.
            # Use max(0, ...) to prevent negative lengths for very short sequences.
            effective_seq_len = max(0, seq_len - 5)
            
            # Slice plddt and pae to exclude the last 5 residues
            plddt_effective = plddt[:effective_seq_len]
            pae_effective = pae[:effective_seq_len, :effective_seq_len]
            
            # Use the effective length for padding and further calculations
            base_lists["lengths"].append(effective_seq_len)
            slice_len = min(effective_seq_len, max_length)
            
            # Pad the effective (sliced) data, not the original data
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
            # --- MODIFICATION END ---
            
            base_lists["sequences"].append(sequence)
            base_lists["pae"].append(padded_pae)
            base_lists["plddt"].append(padded_plddt)
            base_lists["embedding"].append(get_protein_embedding(sequence, tokenizer, plm_model, device))
            base_lists["labels"].append(class_label)
            base_lists["pae_row"].append(padded_pae_row)
            base_lists["pae_col"].append(padded_pae_col)
            
            bio_feats = calculate_biochemical_features(sequence)
            feature_lists["biochem"].append([bio_feats['pI'], bio_feats['gravy'], bio_feats['aromaticity'], bio_feats['mol_weight']])

            # Note: Scalar features below are still calculated on the full-length protein
            # Only RoG goes into the scalar feature vector
            rog = calculate_rog(pdb_path)
            feature_lists["advanced_struct"].append([rog])

            
    print(f"Data preparation complete. Total samples: {len(base_lists['labels'])}")
    sequences = base_lists.pop("sequences")
    for key in base_lists: base_lists[key] = np.array(base_lists[key])
    for key in feature_lists: feature_lists[key] = np.array(feature_lists[key])
    
    return base_lists, feature_lists, sequences

# --- 2. The Model Architecture (Accepts all scalar features) ---
def create_model_with_features(max_length, embedding_dim, num_scalar_features):
    """
    MODIFIED model architecture with a dedicated dense pathway for scalar features.
    """
    L2_REG = regularizers.l2(5e-4)
    
    # --- Input layers (unchanged) ---
    pae_input = Input(shape=(max_length, max_length, 1), name="pae_input")
    plddt_input = Input(shape=(max_length, 1), name="plddt_input")
    embedding_input = Input(shape=(embedding_dim,), name="embedding_input")
    pae_row_input = Input(shape=(max_length, 1), name="pae_row_input")
    pae_col_input = Input(shape=(max_length, 1), name="pae_col_input")
    length_input = Input(shape=(1,), name="length_input")
    scalar_features_input = Input(shape=(num_scalar_features,), name="scalar_features_input")

    # --- Processing for non-scalar features (unchanged) ---
    normalized_length = layers.Lambda(lambda x: x / max_length, name='normalize_length')(length_input)
    x_pae = layers.Conv2D(16, (5, 5), activation='relu')(pae_input); x_pae = layers.MaxPooling2D((3, 3))(x_pae); x_pae = layers.BatchNormalization()(x_pae)
    x_pae = layers.Conv2D(32, (3, 3), activation='relu')(x_pae); x_pae = layers.MaxPooling2D((3, 3))(x_pae); x_pae = layers.Flatten()(x_pae)
    x_plddt = layers.Conv1D(16, 5, activation='relu')(plddt_input); x_plddt = layers.MaxPooling1D(3)(x_plddt); x_plddt = layers.BatchNormalization()(x_plddt); x_plddt = layers.Flatten()(x_plddt)
    x_seq = layers.Dense(64, activation='relu')(embedding_input); x_seq = layers.BatchNormalization()(x_seq)
    x_pae_row = layers.Conv1D(8, 5, activation='relu')(pae_row_input); x_pae_row = layers.MaxPooling1D(3)(x_pae_row); x_pae_row = layers.BatchNormalization()(x_pae_row); x_pae_row = layers.Flatten()(x_pae_row)
    x_pae_col = layers.Conv1D(8, 5, activation='relu')(pae_col_input); x_pae_col = layers.MaxPooling1D(3)(x_pae_col); x_pae_col = layers.BatchNormalization()(x_pae_col); x_pae_col = layers.Flatten()(x_pae_col)
    
    # --- MODIFICATION START: Dedicated dense pathway for scalar features ---
    x_scalar = layers.BatchNormalization(name='scalar_bn')(scalar_features_input)
    x_scalar = layers.Dense(16, activation='relu', name='scalar_dense_1')(x_scalar)
    x_scalar = layers.Dropout(0.2, name='scalar_dropout')(x_scalar)
    x_scalar = layers.Dense(8, activation='relu', name='scalar_dense_2')(x_scalar)
    # --- MODIFICATION END ---
    
    # --- Combination and final layers (unchanged, but x_scalar is now processed) ---
    combined = layers.concatenate([x_pae, x_plddt, x_seq, x_pae_row, x_pae_col, normalized_length, x_scalar])
    dense = layers.Dense(64, activation='relu', kernel_regularizer=L2_REG)(combined)
    dense = layers.Dropout(0.6)(dense)
    output = layers.Dense(1, activation='sigmoid', name="output")(dense)
    
    model = Model(inputs=[pae_input, plddt_input, embedding_input, pae_row_input, pae_col_input, length_input, scalar_features_input], outputs=output)
    return model

# --- 3. Main Training Loop ---
def main():
    print(f"--- SCRIPT: Training Model with Dedicated Scalar Pathway ---")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Loading Protein Language Model: {PLM_MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
    plm_model = EsmModel.from_pretrained(PLM_MODEL_NAME).to(device)
    plm_model.eval()

    base_all, features_all, sequences = prepare_data(DATA_DIR, MAX_LENGTH, tokenizer, plm_model, device)

    # biochem: [pI, gravy, aromaticity, mol_weight]
    # advanced_struct: [RoG]
    scalar_features_all = np.hstack([
        features_all["biochem"],          # 4 dims
        features_all["advanced_struct"]   # 1 dim (RoG)
    ])
    print("Scalar feature dim:", scalar_features_all.shape[1])  # should be 5

    length_bins = pd.qcut(base_all["lengths"], q=4, labels=False, duplicates='drop')
    strata = [f"{label}_{bin_}" for label, bin_ in zip(base_all["labels"], length_bins)]

    pae_all = np.expand_dims(base_all["pae"], -1)
    plddt_all = np.expand_dims(base_all["plddt"], -1)
    pae_row_all = np.expand_dims(base_all["pae_row"], -1)
    pae_col_all = np.expand_dims(base_all["pae_col"], -1)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    fold_metrics = []
    prediction_df = pd.DataFrame({'sequence': sequences})

    for fold_no, (train_index, test_index) in enumerate(skf.split(pae_all, strata), 1):
        print(f'\n{"-"*20} FOLD {fold_no} {"-"*20}')

        scalar_train = scalar_features_all[train_index]
        scalar_test = scalar_features_all[test_index]

        scaler = StandardScaler()
        scalar_train_scaled = scaler.fit_transform(scalar_train)
        scalar_test_scaled = scaler.transform(scalar_test)
        
        scaler_path = os.path.join(SAVE_DIR, f"scalar_scaler_fold_{fold_no}.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"Saved feature scaler for fold {fold_no} to {scaler_path}")

        X_train = {
            "pae_input": pae_all[train_index], "plddt_input": plddt_all[train_index],
            "embedding_input": base_all["embedding"][train_index], "pae_row_input": pae_row_all[train_index],
            "pae_col_input": pae_col_all[train_index], "length_input": base_all["lengths"][train_index],
            "scalar_features_input": scalar_train_scaled
        }
        X_test = {
            "pae_input": pae_all[test_index], "plddt_input": plddt_all[test_index],
            "embedding_input": base_all["embedding"][test_index], "pae_row_input": pae_row_all[test_index],
            "pae_col_input": pae_col_all[test_index], "length_input": base_all["lengths"][test_index],
            "scalar_features_input": scalar_test_scaled
        }
        y_train, y_test = base_all["labels"][train_index], base_all["labels"][test_index]
        
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
        
        model = create_model_with_features(MAX_LENGTH, EMBEDDING_DIM, scalar_features_all.shape[1])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='sensitivity')])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=75, restore_best_weights=True)
        
        print("Training model...")
        model.fit(X_train, y_train, batch_size=32, epochs=300, validation_data=(X_test, y_test),
                  class_weight=class_weights, callbacks=[early_stopping], verbose=0)
        
        scores = model.evaluate(X_test, y_test, verbose=0)
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype("int32")
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        print(f'Score for fold {fold_no}: Accuracy={scores[1]:.4f}, AUC={scores[2]:.4f}, Sensitivity={scores[3]:.4f}, Specificity={specificity:.4f}')
        
        fold_metrics.append({
            'fold': fold_no, 'accuracy': scores[1], 'auc': scores[2], 
            'sensitivity': scores[3], 'specificity': specificity
        })
        prediction_df.loc[test_index, f'fold_{fold_no}_prob'] = y_pred_proba
        
        model_fold_path = os.path.join(SAVE_DIR, f"amyloid_champion_fold_{fold_no}.keras")
        print(f"Saving model for fold {fold_no} to {model_fold_path}")
        model.save(model_fold_path)

    print("\n--- Training complete. All fold models have been saved. ---")
    
    metrics_df = pd.DataFrame(fold_metrics)
    
    print("\n" + "="*50)
    print("           CROSS-VALIDATION PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Accuracy:    {metrics_df['accuracy'].mean():.4f} +/- {metrics_df['accuracy'].std():.4f}")
    print(f"AUC:         {metrics_df['auc'].mean():.4f} +/- {metrics_df['auc'].std():.4f}")
    print(f"Sensitivity: {metrics_df['sensitivity'].mean():.4f} +/- {metrics_df['sensitivity'].std():.4f}")
    print(f"Specificity: {metrics_df['specificity'].mean():.4f} +/- {metrics_df['specificity'].std():.4f}")
    print("="*50)

    prob_cols = [f'fold_{i}_prob' for i in range(1, N_SPLITS + 1)]
    prediction_df['average_prob'] = prediction_df[prob_cols].mean(axis=1)
    
    csv_save_path = os.path.join(SAVE_DIR, "prediction_results.csv")
    prediction_df.to_csv(csv_save_path, index=False)
    print(f"\nPrediction probabilities for each sequence saved to: {csv_save_path}")

if __name__ == "__main__":
    main()