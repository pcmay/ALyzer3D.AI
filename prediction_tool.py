import os
import json
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, EsmModel
import tensorflow as tf
from Bio.PDB import PDBParser, SASA
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# --- Configuration (can be adjusted) ---
MAX_LENGTH = 120
PLM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# --- Feature Engineering & Data Prep Functions (Copied from original script) ---
# (These functions remain the same as in your original code)

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

# --- Main Prediction Class ---

class AmyloidPredictor:
    """
    A single prediction tool for classifying proteins as amyloid or non-amyloid
    based on PDB and ColabFold JSON files.
    """
    def __init__(self, model_dir: str, scaler_path: str):
        """
        Initializes the predictor by loading all necessary models and scalers.

        Args:
            model_dir (str): Path to the directory containing the trained .keras models.
            scaler_path (str): Path to the saved .joblib scaler file.
        """
        print("Initializing AmyloidPredictor...")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found at: {model_dir}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at: {scaler_path}")

        # --- Load PLM ---
        print("Loading Protein Language Model (ESM-2)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
        self.plm_model = EsmModel.from_pretrained(PLM_MODEL_NAME).to(self.device)
        self.plm_model.eval()
        print(f"PLM loaded on device: {self.device}")

        # --- Load Feature Scaler ---
        print(f"Loading feature scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)

        # --- Load Keras Ensemble Models ---
        model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.keras')]
        if not model_paths:
            raise ValueError(f"No .keras models found in directory: {model_dir}")
        
        print(f"Loading {len(model_paths)} ensemble models...")
        self.ensemble_models = [tf.keras.models.load_model(path, safe_mode=False) for path in sorted(model_paths)]
        print("Initialization complete.")

    def predict(self, pdb_path: str, json_path: str) -> dict:
        """
        Makes a prediction for a single protein structure.

        Args:
            pdb_path (str): Path to the input .pdb file.
            json_path (str): Path to the corresponding ColabFold .json file.

        Returns:
            dict: A dictionary containing the prediction results.
                  e.g., {'prediction_label': 'amyloid', 'prediction_probability': 0.85, 'sequence': '...', 'error': None}
        """
        try:
            # --- 1. Load Data and Sequence ---
            sequence = load_sequence_from_pdb(pdb_path)
            if not sequence:
                return {"error": "Could not extract sequence from PDB file."}
            
            with open(json_path, 'r') as f:
                colabfold_data = json.load(f)
            plddt = np.array(colabfold_data['plddt'])
            pae = np.array(colabfold_data['pae'])
            seq_len = len(sequence)

            # --- 2. Feature Engineering ---
            # PLM Embedding
            embedding = get_protein_embedding(sequence, self.tokenizer, self.plm_model, self.device)
            
            # Scalar Features
            bio_feats = calculate_biochemical_features(sequence)
            biochem_features = [bio_feats['pI'], bio_feats['gravy'], bio_feats['aromaticity'], bio_feats['mol_weight']]
            
            plddt_scores = plddt[:seq_len]
            struct_summary_features = [
                np.mean(plddt_scores) if seq_len > 0 else 0,
                np.std(plddt_scores) if seq_len > 0 else 0,
                np.mean(plddt_scores < 70) if seq_len > 0 else 0,
                np.mean(pae[:seq_len, :seq_len]) if seq_len > 0 else 0
            ]
            sasa_feature = [calculate_sasa(pdb_path)]
            advanced_struct_features = [calculate_middle_plddt(plddt, seq_len), calculate_rog(pdb_path)]

            # --- 3. Prepare Data for Model Input ---
            # Pad sequence-based features
            slice_len = min(seq_len, MAX_LENGTH)
            padded_pae = np.zeros((MAX_LENGTH, MAX_LENGTH)); padded_pae[:slice_len, :slice_len] = pae[:slice_len, :slice_len]
            padded_plddt = np.zeros(MAX_LENGTH); padded_plddt[:slice_len] = plddt[:slice_len]
            padded_pae_row = np.zeros(MAX_LENGTH); padded_pae_row[:slice_len] = np.mean(pae[:slice_len, :slice_len], axis=1) if slice_len > 0 else 0
            padded_pae_col = np.zeros(MAX_LENGTH); padded_pae_col[:slice_len] = np.mean(pae[:slice_len, :slice_len], axis=0) if slice_len > 0 else 0

            # Scale scalar features
            scalar_features_combined = np.array(biochem_features + struct_summary_features + sasa_feature + advanced_struct_features).reshape(1, -1)
            scalar_features_scaled = self.scaler.transform(scalar_features_combined)

            # Create the input dictionary, ensuring a "batch size of 1" for each input
            X_pred_dict = {
                "pae_input": np.expand_dims(np.expand_dims(padded_pae, -1), 0),
                "plddt_input": np.expand_dims(np.expand_dims(padded_plddt, -1), 0),
                "embedding_input": np.expand_dims(embedding, 0),
                "pae_row_input": np.expand_dims(np.expand_dims(padded_pae_row, -1), 0),
                "pae_col_input": np.expand_dims(np.expand_dims(padded_pae_col, -1), 0),
                "length_input": np.array([seq_len]),
                "scalar_features_input": scalar_features_scaled
            }

            # --- 4. Run Prediction ---
            all_predictions = [model.predict(X_pred_dict, verbose=0) for model in self.ensemble_models]
            # Average predictions from the ensemble
            y_pred_proba = np.mean(np.hstack(all_predictions))
            
            # --- 5. Format Output ---
            prediction_label = "amyloid" if y_pred_proba > 0.5 else "non_amyloid"
            
            return {
                "prediction_label": prediction_label,
                "prediction_probability": float(y_pred_proba),
                "sequence": sequence,
                "error": None
            }

        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}


if __name__ == '__main__':
    # --- How to use the AmyloidPredictor class ---

    # DEFINE YOUR PATHS HERE
    # These must point to the directory with your .keras models and the specific scaler file
    MODEL_DIRECTORY = "champion_v8_ensemble_modelv3"
    SCALER_FILE_PATH = os.path.join(MODEL_DIRECTORY, "scalar_scaler_v8_ensemble.joblib")
    
    # Example usage requires creating dummy files, since we don't have the real ones.
    # In your web interface, these paths will come from the user's file uploads.
    print("\n--- EXAMPLE USAGE ---")
    print("NOTE: This example will fail if model/scaler paths are incorrect or if dummy files are not present.")
    
    # Create dummy files for demonstration purposes if they don't exist
    if not os.path.exists(MODEL_DIRECTORY):
        print(f"Warning: Model directory '{MODEL_DIRECTORY}' not found. The script will fail.")
        # You would need your actual model files here.
    if not os.path.exists(SCALER_FILE_PATH):
        print(f"Warning: Scaler file '{SCALER_FILE_PATH}' not found. The script will fail.")
        # You would need your actual scaler file here.
        
    # Create placeholder PDB and JSON for the script to run without error.
    # REPLACE THESE WITH YOUR ACTUAL TEST FILES
    DUMMY_PDB_PATH = "example.pdb"
    DUMMY_JSON_PATH = "example.json"
    
    if not os.path.exists(DUMMY_PDB_PATH):
        with open(DUMMY_PDB_PATH, "w") as f:
            f.write("ATOM      1  N   ALA A   1      27.222  16.142  28.534  1.00  0.00           N\n")
            f.write("ATOM      2  CA  ALA A   1      26.200  15.220  28.000  1.00  0.00           C\n")

    if not os.path.exists(DUMMY_JSON_PATH):
        dummy_data = {
            'plddt': [85.5, 90.1, 88.7], # Length should match sequence
            'pae': [[0.1, 2.3, 4.5], [2.4, 0.2, 3.1], [4.6, 3.2, 0.1]]
        }
        with open(DUMMY_JSON_PATH, "w") as f:
            json.dump(dummy_data, f)

    try:
        # 1. Initialize the predictor (loads all models, only needs to be done once)
        predictor = AmyloidPredictor(model_dir=MODEL_DIRECTORY, scaler_path=SCALER_FILE_PATH)

        # 2. Make a prediction on a new pair of files
        # In a web app, you would get these paths from the uploaded files
        result = predictor.predict(pdb_path=DUMMY_PDB_PATH, json_path=DUMMY_JSON_PATH)
        
        # 3. Print the result
        if result.get("error"):
            print(f"\nPrediction failed: {result['error']}")
        else:
            print("\n--- Prediction Result ---")
            print(f"  Sequence: {result['sequence']}")
            print(f"  Prediction: {result['prediction_label'].upper()}")
            print(f"  Confidence Score (Amyloid Probability): {result['prediction_probability']:.4f}")
            print("-------------------------")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: Could not initialize or run predictor. Please check your paths and files.")
        print(f"Details: {e}")