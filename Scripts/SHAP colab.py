# ============================================================
# Colab version of SHAP analysis script (GPU-ready)
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from collections import defaultdict

# --- Colab-specific setup -----------------------------------
from google.colab import drive
drive.mount('/content/drive')  # Uncomment if not already mounted

# ============================================================
# ANARCI
# ============================================================
from anarci import anarci

# PLM Dependencies
import torch
from transformers import AutoTokenizer, EsmModel

# Deep Learning and Data Processing Libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import regularizers, layers

# BioPython for PDB & Sequence Analysis
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Optional: sklearn for PCA + clustering
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except ImportError:
    PCA = None
    KMeans = None

# Optional: SciPy for t-tests
try:
    from scipy import stats
except ImportError:
    stats = None


# ---------------------------------------------------------------------
# Simple 95% CI helpers
# ---------------------------------------------------------------------
def mean_ci_95(arr, axis=0):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("mean_ci_95 got an empty array")

    n = arr.shape[axis]
    mean = np.mean(arr, axis=axis)

    if n <= 1:
        return mean, mean, mean

    se = np.std(arr, axis=axis, ddof=1) / np.sqrt(n)
    delta = 1.96 * se
    ci_low = mean - delta
    ci_high = mean + delta
    return mean, ci_low, ci_high


def mean_ci_95_flat(arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("mean_ci_95_flat got an empty array")

    n = arr.size
    mean = arr.mean()
    if n <= 1:
        return mean, mean, mean

    se = arr.std(ddof=1) / np.sqrt(n)
    delta = 1.96 * se
    return mean, mean - delta, mean + delta


# ---------------------------------------------------------------------
# SEM + paired t-test helper for ALL bar plots
# ---------------------------------------------------------------------
def summarize_across_models(values_matrix, labels, prefix, allow_nan=False, save_dir="."):
    values_matrix = np.asarray(values_matrix, dtype=float)
    n_models, n_cats = values_matrix.shape
    labels = list(labels)

    means = np.zeros(n_cats, dtype=float)
    sems = np.zeros(n_cats, dtype=float)

    for j in range(n_cats):
        col = values_matrix[:, j]
        if allow_nan:
            valid = ~np.isnan(col)
            col_valid = col[valid]
        else:
            col_valid = col
        if col_valid.size == 0:
            means[j] = np.nan
            sems[j] = np.nan
        elif col_valid.size == 1:
            means[j] = col_valid[0]
            sems[j] = 0.0
        else:
            means[j] = col_valid.mean()
            sems[j] = col_valid.std(ddof=1) / np.sqrt(col_valid.size)

    summary_df = pd.DataFrame({
        "Category": labels,
        "Mean": means,
        "SEM": sems,
    })
    summary_path = os.path.join(save_dir, f"{prefix}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    p_matrix = np.full((n_cats, n_cats), np.nan, dtype=float)
    if stats is not None and n_models > 1 and n_cats > 1:
        for i in range(n_cats):
            for j in range(i + 1, n_cats):
                col_i = values_matrix[:, i]
                col_j = values_matrix[:, j]
                if allow_nan:
                    valid = ~(np.isnan(col_i) | np.isnan(col_j))
                    col_i_use = col_i[valid]
                    col_j_use = col_j[valid]
                else:
                    col_i_use = col_i
                    col_j_use = col_j
                if col_i_use.size < 2:
                    p_val = np.nan
                else:
                    t_stat, p_val = stats.ttest_rel(col_i_use, col_j_use)
                p_matrix[i, j] = p_matrix[j, i] = p_val

        p_df = pd.DataFrame(p_matrix, index=labels, columns=labels)
        p_path = os.path.join(save_dir, f"{prefix}_pairwise_ttest_pvalues.csv")
        p_df.to_csv(p_path)
        print(f"Paired t-test p-values for '{prefix}' saved to {p_path}")
    else:
        print(f"No SciPy / insufficient models; skipping pairwise significance for '{prefix}'.")

    return means, sems, summary_df, p_matrix


def significance_label(p):
    """Disable significance stars in all plots."""
    return ""


# ---------------------------------------------------------------------
# Configuration (Colab / Google Drive)
# ---------------------------------------------------------------------
BASE_DIR = "/content/drive/MyDrive"

SAVE_DIR = os.path.join(BASE_DIR, "paper_model_all_versions_shap_oneexpl_entire")

MODEL_VERSION_DIRS = [
    os.path.join(BASE_DIR, f"paper_model_scalar_pathway_v1_minus5_stripped_80_20_seed{i}")
    for i in range(0, 12)
]

TEST_DATA_DIR = os.path.join(BASE_DIR, "entire")

FOLD_INDICES = [1, 2, 3, 4, 5, 6]

N_BACKGROUND_SAMPLES = 400
N_EXPLAIN_SAMPLES = 4860

MAX_LENGTH = 120
PLM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
EMBEDDING_DIM = 320

HMMER_PATH = "/usr/bin/"


# ---------------------------------------------------------------------
# MODEL DEFINITION (must match training)
# ---------------------------------------------------------------------
def create_model_with_features(max_length, embedding_dim, num_scalar_features):
    L2_REG = regularizers.l2(5e-4)

    pae_input = Input(shape=(max_length, max_length, 1), name="pae_input")
    plddt_input = Input(shape=(max_length, 1), name="plddt_input")
    embedding_input = Input(shape=(embedding_dim,), name="embedding_input")
    pae_row_input = Input(shape=(max_length, 1), name="pae_row_input")
    pae_col_input = Input(shape=(max_length, 1), name="pae_col_input")
    length_input = Input(shape=(1,), name="length_input")
    scalar_features_input = Input(shape=(num_scalar_features,), name="scalar_features_input")

    normalized_length = Lambda(lambda x: x / max_length, name='normalize_length')(length_input)

    # PAE 2D conv path
    x_pae = layers.Conv2D(16, (5, 5), activation='relu')(pae_input)
    x_pae = layers.MaxPooling2D((3, 3))(x_pae)
    x_pae = layers.BatchNormalization()(x_pae)
    x_pae = layers.Conv2D(32, (3, 3), activation='relu')(x_pae)
    x_pae = layers.MaxPooling2D((3, 3))(x_pae)
    x_pae = layers.Flatten()(x_pae)

    # pLDDT 1D conv path
    x_plddt = layers.Conv1D(16, 5, activation='relu')(plddt_input)
    x_plddt = layers.MaxPooling1D(3)(x_plddt)
    x_plddt = layers.BatchNormalization()(x_plddt)
    x_plddt = layers.Flatten()(x_plddt)

    # PLM embedding dense path
    x_seq = layers.Dense(64, activation='relu')(embedding_input)
    x_seq = layers.BatchNormalization()(x_seq)

    # PAE row mean
    x_pae_row = layers.Conv1D(8, 5, activation='relu')(pae_row_input)
    x_pae_row = layers.MaxPooling1D(3)(x_pae_row)
    x_pae_row = layers.BatchNormalization()(x_pae_row)
    x_pae_row = layers.Flatten()(x_pae_row)

    # PAE col mean
    x_pae_col = layers.Conv1D(8, 5, activation='relu')(pae_col_input)
    x_pae_col = layers.MaxPooling1D(3)(x_pae_col)
    x_pae_col = layers.BatchNormalization()(x_pae_col)
    x_pae_col = layers.Flatten()(x_pae_col)

    # Scalar features head
    x_scalar = layers.BatchNormalization(name='scalar_bn')(scalar_features_input)
    x_scalar = layers.Dense(16, activation='relu', name='scalar_dense_1')(x_scalar)
    x_scalar = layers.Dropout(0.2, name='scalar_dropout')(x_scalar)
    x_scalar = layers.Dense(8, activation='relu', name='scalar_dense_2')(x_scalar)

    combined = layers.concatenate(
        [x_pae, x_plddt, x_seq, x_pae_row, x_pae_col, normalized_length, x_scalar]
    )

    dense = layers.Dense(64, activation='relu', kernel_regularizer=L2_REG)(combined)
    dense = layers.Dropout(0.6)(dense)
    output = layers.Dense(1, activation='sigmoid', name="output")(dense)

    model = Model(
        inputs=[
            pae_input,
            plddt_input,
            embedding_input,
            pae_row_input,
            pae_col_input,
            length_input,
            scalar_features_input,
        ],
        outputs=output,
    )
    return model


# ---------------------------------------------------------------------
# FEATURE ENGINEERING & DATA PREP
# ---------------------------------------------------------------------
def calculate_rog(pdb_path: str) -> float:
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]

        atoms = list(model.get_atoms())
        if not atoms:
            return 0.0

        center_of_mass = sum(atom.coord for atom in atoms) / len(atoms)
        rog_sq_sum = sum(np.sum((atom.coord - center_of_mass) ** 2) for atom in atoms)
        rog = np.sqrt(rog_sq_sum / len(atoms))

        num_residues = len(list(model.get_residues()))
        return rog / np.sqrt(num_residues) if num_residues > 0 else 0.0
    except Exception:
        return 0.0


def calculate_biochemical_features(sequence: str) -> dict:
    try:
        clean_sequence = "".join([c for c in sequence if c in "ACDEFGHIKLMNPQRSTVWY"])
        analysed_seq = ProteinAnalysis(clean_sequence)
        return {
            'pI': analysed_seq.isoelectric_point(),
            'gravy': analysed_seq.gravy(),
            'aromaticity': analysed_seq.aromaticity(),
            'mol_weight': analysed_seq.molecular_weight(),
        }
    except Exception:
        return {'pI': 7.0, 'gravy': 0.0, 'aromaticity': 0.0, 'mol_weight': 12000.0}


def get_protein_embedding(sequence: str, tokenizer, plm_model, device):
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=1022,
    ).to(device)
    with torch.no_grad():
        outputs = plm_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()


def load_sequence_from_pdb(pdb_file):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        model = next(structure.get_models())
        chain = next(model.get_chains())
        seq = []
        for r in chain.get_residues():
            if not is_aa(r, standard=True):
                continue
            aa = protein_letters_3to1.get(r.get_resname().upper(), 'X')
            if aa != 'X':
                seq.append(aa)
        return "".join(seq)
    except Exception:
        return None


def prepare_test_data(data_dir, max_length, tokenizer, plm_model, device):
    base_lists = {
        "pae": [],
        "plddt": [],
        "embedding": [],
        "labels": [],
        "lengths": [],
        "pae_row": [],
        "pae_col": [],
    }
    feature_lists = {
        "biochem": [],
        "advanced_struct": [],
    }
    sequences_list = []

    print("Preparing test data for SHAP analysis (with minus-5 logic)...")

    for class_label, class_name in enumerate(["non_amyloid", "amyloid"]):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        protein_files = {}
        for f in os.listdir(class_dir):
            base_name = f.split('scores_rank')[0].split('unrelaxed_rank')[0]
            protein_files.setdefault(base_name, []).append(f)

        for base_name, files in tqdm(protein_files.items(), desc=f"Processing {class_name}"):
            json_file = next(
                (f for f in files if ('_rank_001' in f or '_rank_1' in f) and f.endswith('.json')),
                next((f for f in files if f.endswith('.json')), None),
            )
            pdb_file = next(
                (f for f in files if ('_rank_001' in f or '_rank_1' in f) and f.endswith('.pdb')),
                next((f for f in files if f.endswith('.pdb')), None),
            )
            if not json_file or not pdb_file:
                continue

            json_path = os.path.join(class_dir, json_file)
            pdb_path = os.path.join(class_dir, pdb_file)

            sequence = load_sequence_from_pdb(pdb_path)
            if not sequence:
                continue
            sequences_list.append(sequence)

            with open(json_path, 'r') as f:
                colabfold_data = json.load(f)
            plddt = np.array(colabfold_data['plddt'])
            pae = np.array(colabfold_data['pae'])

            seq_len = len(sequence)

            effective_seq_len = max(0, seq_len - 5)
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
                padded_pae_row[:slice_len] = np.mean(
                    pae_effective[:slice_len, :slice_len], axis=1
                )

            padded_pae_col = np.zeros(max_length)
            if slice_len > 0:
                padded_pae_col[:slice_len] = np.mean(
                    pae_effective[:slice_len, :slice_len], axis=0
                )

            base_lists["pae"].append(padded_pae)
            base_lists["plddt"].append(padded_plddt)
            base_lists["embedding"].append(
                get_protein_embedding(sequence, tokenizer, plm_model, device)
            )
            base_lists["labels"].append(class_label)
            base_lists["pae_row"].append(padded_pae_row)
            base_lists["pae_col"].append(padded_pae_col)

            bio_feats = calculate_biochemical_features(sequence)
            feature_lists["biochem"].append(
                [
                    bio_feats['pI'],
                    bio_feats['gravy'],
                    bio_feats['aromaticity'],
                    bio_feats['mol_weight'],
                ]
            )

            rog = calculate_rog(pdb_path)
            feature_lists["advanced_struct"].append([rog])

    for key in base_lists:
        base_lists[key] = np.array(base_lists[key])
    for key in feature_lists:
        feature_lists[key] = np.array(feature_lists[key])

    return base_lists, feature_lists, sequences_list


# ---------------------------------------------------------------------
# MAIN SHAP ANALYSIS (single explainer on full model)
# ---------------------------------------------------------------------
def run_shap_analysis():
    print(f"--- SCRIPT: SHAP Analysis for Model Interpretability (All Versions; minus-5 logic) ---")
    print(f"Loading Protein Language Model: {PLM_MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
    plm_model = EsmModel.from_pretrained(PLM_MODEL_NAME).to(device)
    plm_model.eval()

    base_test, features_test, sequences = prepare_test_data(
        TEST_DATA_DIR, MAX_LENGTH, tokenizer, plm_model, device
    )

    if len(base_test["labels"]) == 0:
        print("No test data found. Exiting.")
        return

    # scalar_features_all must match training: [4 biochem + 1 RoG] = 5
    scalar_features_all = np.hstack(
        [
            features_test["biochem"],
            features_test["advanced_struct"],
        ]
    )
    print("Scalar feature dim (SHAP):", scalar_features_all.shape[1])  # should be 5

    scalar_feature_names = [
        'pI',
        'GRAVY',
        'Aromaticity',
        'Mol. Weight',
        'Norm. RoG',
    ]

    # Helper: token-level embeddings for per-residue PLM attribution
    def get_token_embeddings_for_sequence(seq):
        inputs = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=1022,
        ).to(device)
        with torch.no_grad():
            outputs = plm_model(**inputs)
        tok_repr = outputs.last_hidden_state.squeeze(0)
        L_tokens, D = tok_repr.shape
        L_seq = len(seq)

        if L_tokens == L_seq + 2:
            tok_repr = tok_repr[1:-1]
        elif L_tokens >= L_seq:
            tok_repr = tok_repr[:L_seq]

        return tok_repr.cpu().numpy()

    # ------------------------------------------------------------
    # Decide how many samples to use for background and explanation
    # ------------------------------------------------------------
    total_samples = len(sequences)
    if total_samples < 2:
        print("Not enough samples for SHAP.")
        return

    n_bg = min(N_BACKGROUND_SAMPLES, total_samples // 3 if total_samples >= 3 else 1)
    n_explain = min(N_EXPLAIN_SAMPLES, total_samples - n_bg)

    if n_explain <= 0:
        n_bg = max(1, total_samples // 2)
        n_explain = total_samples - n_bg

    if n_bg + n_explain > total_samples:
        n_bg = max(1, total_samples // 3)
        n_explain = total_samples - n_bg

    if n_bg + n_explain > total_samples:
        n_bg = 1
        n_explain = total_samples - 1

    if total_samples > n_bg + n_explain:
        indices = np.random.choice(total_samples, n_bg + n_explain, replace=False)
    else:
        indices = np.arange(total_samples)

    background_indices = indices[:n_bg]
    explain_indices = indices[n_bg:]
    sequences_to_explain = [sequences[i] for i in explain_indices]

    print(f"\nUsing {n_bg} background samples and {n_explain} explanation samples "
          f"out of {total_samples} total sequences.")

    # -----------------------------------------------------------------
    # Aggregate SHAP across all versions (per-version, then across)
    # SINGLE EXPLAINER ON FULL MULTI-INPUT MODEL
    # -----------------------------------------------------------------
    version_shap_sums = {}
    version_fold_counts = defaultdict(int)
    processed_models = []
    X_test_for_plots = None

    for version_dir in MODEL_VERSION_DIRS:
        for fold in FOLD_INDICES:
            print(f"\n=== Processing version '{version_dir}', fold {fold} ===")
            model_path = os.path.join(version_dir, f"amyloid_champion_fold_{fold}.keras")
            scaler_path = os.path.join(version_dir, f"scalar_scaler_fold_{fold}.joblib")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                print(f"WARNING: Model or scaler not found for version '{version_dir}', fold {fold}, skipping.")
                continue

            scaler = joblib.load(scaler_path)
            num_scalar_features = scaler.n_features_in_

            if num_scalar_features != scalar_features_all.shape[1]:
                print(
                    f"WARNING: Scalar feature dimension mismatch in '{version_dir}', fold {fold} "
                    f"({num_scalar_features} vs {scalar_features_all.shape[1]}). Skipping."
                )
                continue

            original_model = create_model_with_features(MAX_LENGTH, EMBEDDING_DIM, num_scalar_features)
            original_model.load_weights(model_path)

            # Prepare scaled inputs for this model
            scalar_features_unscaled = scalar_features_all

            X_test = {
                "pae_input": np.expand_dims(base_test["pae"], -1),
                "plddt_input": np.expand_dims(base_test["plddt"], -1),
                "embedding_input": base_test["embedding"],
                "pae_row_input": np.expand_dims(base_test["pae_row"], -1),
                "pae_col_input": np.expand_dims(base_test["pae_col"], -1),
                "length_input": np.expand_dims(base_test["lengths"], -1),
                "scalar_features_input": scaler.transform(scalar_features_unscaled),
            }
            X_test_for_plots = X_test  # keep one copy for later plots

            input_names_in_order = [inp.name.split(':')[0] for inp in original_model.inputs]

            # Build background and explanation lists in the same order
            print("  Computing SHAP values for full multi-input model...")
            background_full = [X_test[name][background_indices] for name in input_names_in_order]
            explain_full = [X_test[name][explain_indices] for name in input_names_in_order]

            explainer = shap.GradientExplainer(original_model, background_full)
            shap_values_full = explainer.shap_values(explain_full)

            # Handle possible output structures from shap
            if isinstance(shap_values_full, list) and \
               len(shap_values_full) == len(input_names_in_order) and \
               not isinstance(shap_values_full[0], list):
                # [ per_input_array ]
                per_input_shap = shap_values_full
            elif isinstance(shap_values_full, list) and len(shap_values_full) == 1 and \
                 isinstance(shap_values_full[0], list) and \
                 len(shap_values_full[0]) == len(input_names_in_order):
                # [[ per_input_array ]]
                per_input_shap = shap_values_full[0]
            else:
                raise ValueError(
                    "Unexpected SHAP output structure from GradientExplainer for multi-input model."
                )

            fold_shap_values = {
                name: np.array(vals, dtype=np.float64)
                for name, vals in zip(input_names_in_order, per_input_shap)
            }

            # accumulate across folds within a version
            if version_dir not in version_shap_sums:
                version_shap_sums[version_dir] = {
                    k: v.copy().astype(np.float64) for k, v in fold_shap_values.items()
                }
            else:
                for k in version_shap_sums[version_dir]:
                    version_shap_sums[version_dir][k] += fold_shap_values[k].astype(np.float64)

            version_fold_counts[version_dir] += 1
            processed_models.append((version_dir, fold))

    if not version_shap_sums:
        print("No models were successfully processed. Exiting.")
        return

    shap_values_per_version = {}
    for version_dir, sums in version_shap_sums.items():
        count = version_fold_counts[version_dir]
        shap_values_per_version[version_dir] = {k: v / count for k, v in sums.items()}

    version_names = list(shap_values_per_version.keys())
    n_models = len(version_names)
    print(f"\nSHAP calculation complete. Aggregated over {n_models} model version(s):")
    for version_dir in version_names:
        print(f"  - {version_dir} (folds: {version_fold_counts[version_dir]})")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Average across versions for some global SHAP arrays
    input_names = list(next(iter(shap_values_per_version.values())).keys())
    shap_values_dict = {}
    for name in input_names:
        shap_values_dict[name] = np.mean(
            [shap_values_per_version[v][name] for v in version_names],
            axis=0
        )

    # -----------------------------------------------------------------
    # 3. Standard SHAP Visualizations
    # -----------------------------------------------------------------
    print("\nGenerating Standard SHAP plots with SEM across models and significance asterisks...")

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 24
    })

    # -----------------------------------------------------------------
    # Overall component importance
    # -----------------------------------------------------------------
    component_name_map = {
        'embedding_input': 'PLM Embedding',
        'pae_input': 'PAE Matrix',
        'plddt_input': 'pLDDT Vector',
        'scalar_features_input': 'Scalar Features',
        'pae_row_input': 'PAE Row Mean',
        'pae_col_input': 'PAE Col Mean',
        'length_input': 'Length',
    }

    components = [
        (internal, pretty)
        for internal, pretty in component_name_map.items()
        if internal in shap_values_dict
    ]
    n_comp = len(components)
    component_matrix = np.zeros((n_models, n_comp))

    for mi, version_dir in enumerate(version_names):
        v_shap = shap_values_per_version[version_dir]
        for ci, (internal, pretty) in enumerate(components):
            arr = np.abs(v_shap[internal])
            component_matrix[mi, ci] = np.mean(arr)

    comp_labels = [pretty for _, pretty in components]
    comp_means, comp_sems, comp_summary_df, comp_pmat = summarize_across_models(
        component_matrix, comp_labels, prefix="components", allow_nan=False, save_dir=SAVE_DIR
    )

    component_df = comp_summary_df.copy()
    component_df.rename(columns={"Mean": "Importance"}, inplace=True)
    component_df["Component"] = component_df["Category"]
    component_df = component_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    ref_cat_comp = component_df.loc[0, "Category"]
    ref_idx_comp = comp_labels.index(ref_cat_comp)
    max_x_val = (component_df["Importance"] + component_df["SEM"]).max()

    plt.figure(figsize=(10, 7))
    ax = sns.barplot(
        x='Importance',
        y='Component',
        data=component_df,
        palette='viridis',
        ci=None,
    )

    for i, row in component_df.iterrows():
        x = row['Importance']
        xerr = row['SEM']
        ax.errorbar(
            x=x,
            y=i,
            xerr=xerr,
            fmt='none',
            ecolor='black',
            elinewidth=1,
            capsize=3,
        )

        cat = row["Category"]
        if cat == ref_cat_comp:
            continue
        p_val = comp_pmat[comp_labels.index(cat), ref_idx_comp]
        stars = significance_label(p_val)
        if stars:
            ax.text(
                x + xerr + 0.02 * max_x_val,
                i,
                stars,
                ha='left',
                va='center',
                fontsize=12,
            )

    plt.title('Overall Feature Contribution by Architectural Component', fontsize=16)
    plt.xlabel('Mean Absolute SHAP Value (Impact on output)', fontsize=12)
    plt.ylabel('Model Input Component', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_overall_contribution.png"), dpi=300)
    plt.show()

    # -----------------------------------------------------------------
    # Scalar feature importance + directionality
    # -----------------------------------------------------------------
    scalar_matrix = np.zeros((n_models, len(scalar_feature_names)))
    for mi, version_dir in enumerate(version_names):
        scalar_shap_v = shap_values_per_version[version_dir]['scalar_features_input']
        scalar_abs = np.abs(scalar_shap_v)

        if scalar_abs.ndim == 3 and scalar_abs.shape[-1] == 1:
            vals = scalar_abs.mean(axis=(0, 2))
        elif scalar_abs.ndim == 2:
            vals = scalar_abs.mean(axis=0)
        else:
            vals = scalar_abs.reshape(scalar_abs.shape[0], -1).mean(axis=0)

        scalar_matrix[mi, :] = vals

    scalar_shap_values_avg = shap_values_dict['scalar_features_input']
    if scalar_shap_values_avg.ndim == 3 and scalar_shap_values_avg.shape[-1] == 1:
        scalar_shap_values_avg = scalar_shap_values_avg.squeeze(-1)

    scalar_explain_data = X_test_for_plots['scalar_features_input'][explain_indices]
    correlations = []
    for i in range(len(scalar_feature_names)):
        corr = np.corrcoef(
            scalar_explain_data[:, i].flatten(),
            scalar_shap_values_avg[:, i].flatten(),
        )[0, 1]
        correlations.append(corr)

    scalar_means, scalar_sems, scalar_summary_df, scalar_pmat = summarize_across_models(
        scalar_matrix,
        scalar_feature_names,
        prefix="scalar_features",
        allow_nan=False,
        save_dir=SAVE_DIR,
    )

    scalar_df = scalar_summary_df.copy()
    scalar_df.rename(columns={"Mean": "Importance"}, inplace=True)
    scalar_df["Feature"] = scalar_df["Category"]
    scalar_df["Correlation"] = correlations
    scalar_df = scalar_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    ref_cat_scalar = scalar_df.loc[0, "Category"]
    scalar_labels = scalar_summary_df["Category"].tolist()
    ref_idx_scalar = scalar_labels.index(ref_cat_scalar)
    max_x_val_scalar = (scalar_df["Importance"] + scalar_df["SEM"]).max()

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    norm = plt.Normalize(-1, 1)
    colors = [cmap(norm(c)) for c in scalar_df['Correlation']]

    sns.barplot(
        x='Importance',
        y='Feature',
        data=scalar_df,
        palette=colors,
        ax=ax,
        ci=None,
    )

    for i, row in scalar_df.iterrows():
        x = row['Importance']
        xerr = row['SEM']
        ax.errorbar(
            x=x,
            y=i,
            xerr=xerr,
            fmt='none',
            ecolor='black',
            elinewidth=1,
            capsize=3,
        )

        cat = row["Category"]
        if cat == ref_cat_scalar:
            continue
        p_val = scalar_pmat[scalar_labels.index(cat), ref_idx_scalar]
        stars = significance_label(p_val)
        if stars:
            ax.text(
                x + xerr + 0.02 * max_x_val_scalar,
                i,
                stars,
                ha='left',
                va='center',
                fontsize=12,
            )

    ax.set_title('Detailed Scalar Feature Importance and Effect', fontsize=16)
    ax.set_xlabel('Mean Absolute SHAP Value (Overall Impact)', fontsize=12)
    ax.set_ylabel('')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, aspect=40, pad=0.08)
    cbar.set_label('Correlation with SHAP Value', rotation=270, labelpad=15)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Negative', 'Neutral', 'Positive'])
    fig.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_scalar_directional_importance.png"), dpi=300)
    plt.show()

    # -----------------------------------------------------------------
    # PLM embedding SHAP: per-dimension importance + sparsity
    # -----------------------------------------------------------------
    one_version = next(iter(shap_values_per_version.values()))
    emb_example = np.array(one_version['embedding_input'])
    if emb_example.ndim == 1:
        emb_example = emb_example.reshape(1, -1)
    elif emb_example.ndim > 2:
        emb_example = emb_example.reshape(emb_example.shape[0], -1)
    eff_embed_dim = emb_example.shape[1]

    embed_dim_matrix = np.zeros((n_models, eff_embed_dim))
    for mi, version_dir in enumerate(version_names):
        emb_v = np.array(shap_values_per_version[version_dir]['embedding_input'])
        if emb_v.ndim == 1:
            emb_v = emb_v.reshape(1, -1)
        elif emb_v.ndim > 2:
            emb_v = emb_v.reshape(emb_v.shape[0], -1)
        embed_dim_matrix[mi, :] = np.mean(np.abs(emb_v), axis=0)

    mean_abs_embed_overall = embed_dim_matrix.mean(axis=0)

    if n_models > 1:
        se_embed = embed_dim_matrix.std(axis=0, ddof=1) / np.sqrt(n_models)
        delta_embed = 1.96 * se_embed
        ci_low_embed_model = mean_abs_embed_overall - delta_embed
        ci_high_embed_model = mean_abs_embed_overall + delta_embed
    else:
        ci_low_embed_model = ci_high_embed_model = mean_abs_embed_overall

    embed_overall_df = (
        pd.DataFrame(
            {
                'Dim': np.arange(1, eff_embed_dim + 1),
                'Importance': mean_abs_embed_overall,
            }
        )
        .sort_values('Importance', ascending=False)
    )

    TOP_N_EMBED_DIMS = 30
    top_embed_overall = (
        embed_overall_df.head(TOP_N_EMBED_DIMS)
        .sort_values('Dim', ascending=True)
    )
    top_dims = top_embed_overall['Dim'].values
    top_dims_zero = top_dims - 1

    embed_top_matrix = embed_dim_matrix[:, top_dims_zero]
    embed_labels = [f"Dim{d}" for d in top_dims]

    embed_means, embed_sems, embed_summary_df, embed_pmat = summarize_across_models(
        embed_top_matrix,
        embed_labels,
        prefix="embedding_topdims",
        allow_nan=False,
        save_dir=SAVE_DIR,
    )

    top_embed_df = embed_summary_df.copy()
    top_embed_df["Dim"] = [int(label.replace("Dim", "")) for label in top_embed_df["Category"]]
    top_embed_df.rename(columns={"Mean": "Importance"}, inplace=True)
    top_embed_df = top_embed_df.sort_values("Dim", ascending=True).reset_index(drop=True)

    ref_cat_embed = embed_summary_df.loc[embed_summary_df["Mean"].idxmax(), "Category"]
    embed_labels_full = embed_summary_df["Category"].tolist()
    ref_idx_embed = embed_labels_full.index(ref_cat_embed)
    max_y_embed = (top_embed_df["Importance"] + top_embed_df["SEM"]).max()

    plt.figure(figsize=(12, 4))
    ax = sns.barplot(
        x='Dim',
        y='Importance',
        data=top_embed_df,
        palette='mako',
        ci=None,
    )

    for i, row in top_embed_df.iterrows():
        bar = ax.patches[i]
        x_center = bar.get_x() + bar.get_width() / 2.0
        y = row['Importance']
        yerr = row['SEM']
        ax.errorbar(
            x=x_center,
            y=y,
            yerr=yerr,
            fmt='none',
            ecolor='black',
            elinewidth=1,
            capsize=3,
        )

        cat = row["Category"]
        if cat == ref_cat_embed:
            continue
        p_val = embed_pmat[embed_labels_full.index(cat), ref_idx_embed]
        stars = significance_label(p_val)
        if stars:
            ax.text(
                x_center,
                y + yerr + 0.02 * max_y_embed,
                stars,
                ha='center',
                va='bottom',
                fontsize=12,
            )

    plt.title(f'Top {TOP_N_EMBED_DIMS} PLM Embedding Dimensions by SHAP Importance', fontsize=16)
    plt.xlabel('Embedding Dimension', fontsize=12)
    plt.ylabel('Mean Absolute SHAP Value', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_plm_embedding_dims_top30.png"), dpi=300)
    plt.show()

    dims = np.arange(1, eff_embed_dim + 1)
    plt.figure(figsize=(12, 4))
    plt.plot(dims, mean_abs_embed_overall, lw=2)
    plt.fill_between(dims, ci_low_embed_model, ci_high_embed_model, alpha=0.2)
    plt.title('PLM Embedding Dimension-wise SHAP Importance', fontsize=16)
    plt.xlabel('Embedding Dimension', fontsize=12)
    plt.ylabel('Mean Absolute SHAP Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_plm_embedding_dims_full.png"), dpi=300)
    plt.show()

    # -----------------------------------------------------------------
    # Push PLM attribution back to residues (per-residue importance)
    # -----------------------------------------------------------------
    print("\nComputing per-residue PLM attribution for SHAP explanation set...")
    embedding_shap_explain = shap_values_dict['embedding_input']
    embedding_shap_explain = np.array(embedding_shap_explain)

    if embedding_shap_explain.ndim == 1:
        embedding_shap_explain = embedding_shap_explain.reshape(1, -1)
    elif embedding_shap_explain.ndim > 2:
        embedding_shap_explain = embedding_shap_explain.reshape(
            embedding_shap_explain.shape[0], -1
        )

    n_explain_local, emb_dim_local = embedding_shap_explain.shape

    print("Precomputing token embeddings for per-residue PLM analysis...")
    token_embs_list = []
    for seq in sequences_to_explain:
        token_embs_list.append(get_token_embeddings_for_sequence(seq))

    per_seq_plm_residue_importance = []

    for idx_in_explain, seq in enumerate(sequences_to_explain):
        shap_vec = embedding_shap_explain[idx_in_explain]
        w = np.abs(shap_vec)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

        token_embs = token_embs_list[idx_in_explain]

        if token_embs.shape[1] != emb_dim_local:
            D_use = min(token_embs.shape[1], emb_dim_local)
            token_embs_use = token_embs[:, :D_use]
            w_use = w[:D_use]
        else:
            token_embs_use = token_embs
            w_use = w

        contrib = np.abs(token_embs_use) * w_use[None, :]
        per_residue_score = contrib.sum(axis=1)
        per_seq_plm_residue_importance.append(per_residue_score)

    if per_seq_plm_residue_importance:
        max_len_obs = max(len(arr) for arr in per_seq_plm_residue_importance)
        max_plot_len = min(max_len_obs, MAX_LENGTH)

        n_seq_explain = len(per_seq_plm_residue_importance)
        pos_matrix = np.full((n_seq_explain, max_plot_len), np.nan, dtype=float)
        for i, arr in enumerate(per_seq_plm_residue_importance):
            L = min(len(arr), max_plot_len)
            pos_matrix[i, :L] = arr[:L]

        mean_pos = np.nanmean(pos_matrix, axis=0)
        n_eff = np.sum(~np.isnan(pos_matrix), axis=0)
        std_pos = np.nanstd(pos_matrix, axis=0, ddof=1)
        sem_pos = np.zeros_like(mean_pos)
        valid = n_eff > 1
        sem_pos[valid] = std_pos[valid] / np.sqrt(n_eff[valid])
        delta = 1.96 * sem_pos
        low_pos = mean_pos - delta
        high_pos = mean_pos + delta

        positions_res = np.arange(1, max_plot_len + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(positions_res, mean_pos, lw=2)
        plt.fill_between(positions_res, low_pos, high_pos, alpha=0.2)
        plt.title('PLM-derived Per-residue Positional Importance (N-terminal index)', fontsize=16)
        plt.xlabel('Residue Position (from N-terminus)', fontsize=12)
        plt.ylabel('Mean Weighted PLM Contribution', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "shap_plm_residue_positional_importance_nterm.png"), dpi=300)
        plt.show()

    # -----------------------------------------------------------------
    # Correlate top dims back to scalar features
    # -----------------------------------------------------------------
    embedding_all = base_test["embedding"]
    embed_explain_vals = embedding_all[explain_indices][:, top_dims_zero]
    scalar_explain_vals = scalar_features_all[explain_indices]

    corr_mat = np.zeros((TOP_N_EMBED_DIMS, len(scalar_feature_names)))
    for i, dim_idx in enumerate(top_dims_zero):
        for j in range(len(scalar_feature_names)):
            corr_mat[i, j] = np.corrcoef(
                embed_explain_vals[:, i].flatten(),
                scalar_explain_vals[:, j].flatten()
            )[0, 1]

    corr_df = pd.DataFrame(
        corr_mat,
        index=[f"Dim{d}" for d in top_dims],
        columns=scalar_feature_names,
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        corr_df,
        cmap='coolwarm',
        center=0.0,
        annot=False,
        cbar_kws={'label': 'Pearson r'},
    )
    plt.title('Correlation of Top PLM Embedding Dimensions with Scalar Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_plm_embedding_topdims_scalar_corr.png"), dpi=300)
    plt.show()

    # -----------------------------------------------------------------
    # PCA + clustering in top embedding subspace
    # -----------------------------------------------------------------
    if PCA is None or KMeans is None:
        print("scikit-learn not available, skipping PCA and clustering on PLM embeddings.")
    else:
        print("\nRunning PCA and KMeans in subspace of top embedding dimensions...")

        X_top = embed_explain_vals
        labels_explain = base_test["labels"][explain_indices]

        pca = PCA(n_components=2, random_state=0)
        X_pca = pca.fit_transform(X_top)

        plt.figure(figsize=(7, 6))
        for lab, name, col in [(0, 'Non-amyloid', 'tab:blue'),
                               (1, 'Amyloid', 'tab:orange')]:
            mask = labels_explain == lab
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                s=25,
                alpha=0.7,
                label=name,
                color=col,
            )
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA of Top PLM Embedding Dimensions (colored by label)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "shap_plm_embedding_pca_labels.png"), dpi=300)
        plt.show()

        km = KMeans(n_clusters=2, random_state=0, n_init=10)
        cluster_labels = km.fit_predict(X_top)

        plt.figure(figsize=(7, 6))
        for cl, col in [(0, 'tab:green'), (1, 'tab:red')]:
            mask = cluster_labels == cl
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                s=25,
                alpha=0.7,
                label=f'Cluster {cl}',
                color=col,
            )
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA of Top PLM Embedding Dimensions (colored by KMeans clusters)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "shap_plm_embedding_pca_clusters.png"), dpi=300)
        plt.show()

    # -----------------------------------------------------------------
    # Positional importance for pLDDT, PAE row, PAE col (non-IMGT)
    # -----------------------------------------------------------------
    plddt_shap_values = np.array(shap_values_dict['plddt_input']).squeeze()
    if plddt_shap_values.ndim == 1:
        plddt_shap_values = plddt_shap_values.reshape(1, -1)
    abs_plddt = np.abs(plddt_shap_values)
    mean_plddt_shap_per_pos, ci_low_plddt, ci_high_plddt = mean_ci_95(abs_plddt, axis=0)

    positions = np.arange(MAX_LENGTH)
    plt.figure(figsize=(12, 6))
    plt.plot(positions, mean_plddt_shap_per_pos, lw=2)
    plt.fill_between(positions, ci_low_plddt, ci_high_plddt, alpha=0.2)
    plt.title('pLDDT Positional Importance', fontsize=16)
    plt.xlabel('Residue Position', fontsize=12)
    plt.ylabel('Mean Absolute SHAP Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_plddt_positional_importance.png"), dpi=300)
    plt.show()

    pae_row_shap_values = np.array(shap_values_dict['pae_row_input']).squeeze()
    if pae_row_shap_values.ndim == 1:
        pae_row_shap_values = pae_row_shap_values.reshape(1, -1)
    abs_row = np.abs(pae_row_shap_values)
    mean_pae_row_shap_per_pos, ci_low_row, ci_high_row = mean_ci_95(abs_row, axis=0)

    positions = np.arange(MAX_LENGTH)
    plt.figure(figsize=(12, 6))
    plt.plot(positions, mean_pae_row_shap_per_pos, lw=2, color='green')
    plt.fill_between(positions, ci_low_row, ci_high_row, alpha=0.2, color='green')
    plt.title('PAE Row Mean Positional Importance', fontsize=16)
    plt.xlabel('Residue Position', fontsize=12)
    plt.ylabel('Mean Absolute SHAP Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_pae_row_importance.png"), dpi=300)
    plt.show()

    pae_col_shap_values = np.array(shap_values_dict['pae_col_input']).squeeze()
    if pae_col_shap_values.ndim == 1:
        pae_col_shap_values = pae_col_shap_values.reshape(1, -1)
    abs_col = np.abs(pae_col_shap_values)
    mean_pae_col_shap_per_pos, ci_low_col, ci_high_col = mean_ci_95(abs_col, axis=0)

    positions = np.arange(MAX_LENGTH)
    plt.figure(figsize=(12, 6))
    plt.plot(positions, mean_pae_col_shap_per_pos, lw=2, color='purple')
    plt.fill_between(positions, ci_low_col, ci_high_col, alpha=0.2, color='purple')
    plt.title('PAE Column Mean Positional Importance', fontsize=16)
    plt.xlabel('Residue Position', fontsize=12)
    plt.ylabel('Mean Absolute SHAP Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_pae_col_importance.png"), dpi=300)
    plt.show()

    # -----------------------------------------------------------------
    # 4. IMGT-ALIGNED SHAP ANALYSIS (unchanged logic)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60 + "\n      PART 4: IMGT-ALIGNED SHAP ANALYSIS\n" + "=" * 60)
    try:
        print("\nRunning ANARCI to get IMGT numbering...")
        seq_tuples = [(f"seq_{i}", seq) for i, seq in enumerate(sequences_to_explain)]
        anarci_results = anarci(seq_tuples, scheme="imgt", output=False, hmmerpath=HMMER_PATH)

        numbering_results, domain_results, _ = anarci_results
        anarci_maps = []

        for i in tqdm(range(len(sequences_to_explain)), desc="Parsing ANARCI results"):
            mapping = {}
            if i >= len(numbering_results):
                anarci_maps.append(mapping)
                continue

            seq_domains = numbering_results[i]
            if not seq_domains:
                anarci_maps.append(mapping)
                continue

            domain_numbering, start, end = seq_domains[0]

            chain_type = "?"
            if i < len(domain_results):
                chain_info_list = domain_results[i]
                if chain_info_list and isinstance(chain_info_list, list):
                    chain_type = chain_info_list[0].get("chain_type", "?")

            seq_idx = -1
            for numbered_residue in domain_numbering:
                if not (isinstance(numbered_residue, tuple) and len(numbered_residue) == 2):
                    continue
                pos_tuple, res = numbered_residue
                if not (isinstance(pos_tuple, tuple) and len(pos_tuple) == 2):
                    continue
                pos, ins = pos_tuple

                if res != '-':
                    seq_idx += 1
                    imgt_pos = f"{chain_type}{pos}{ins}".strip()
                    mapping[seq_idx] = imgt_pos

            anarci_maps.append(mapping)

        imgt_regions = {
            'FR1':  (1, 26), 'CDR1': (27, 38), 'FR2':  (39, 55),
            'CDR2': (56, 65), 'FR3':  (66, 104), 'CDR3': (105, 117),
            'FR4':  (118, 128),
        }
        MAX_IMGT_POS = 128

        def parse_imgt_label(label):
            if not label: return "?", 0, ""
            chain = label[0]
            num_part = label[1:]
            num_str = "".join(ch for ch in num_part if ch.isdigit())
            ins = "".join(ch for ch in num_part if ch.isalpha())
            num = int(num_str) if num_str else 0
            return chain, num, ins

        def aggregate_to_imgt_for_version(shap_matrix, maps):
            shap_matrix = np.array(shap_matrix).squeeze()
            if shap_matrix.ndim == 1: shap_matrix = shap_matrix.reshape(1, -1)
            aligned = defaultdict(list)
            for i, mapping in enumerate(maps):
                if i >= shap_matrix.shape[0]: break
                for orig_idx, imgt_pos in mapping.items():
                    if orig_idx < shap_matrix.shape[1]:
                        aligned[imgt_pos].append(shap_matrix[i, orig_idx])
            return aligned

        def build_numeric_profile_from_aligned(aligned_dict):
            chain_num_vals = defaultdict(lambda: defaultdict(list))
            for label, vals in aligned_dict.items():
                chain, num, ins = parse_imgt_label(label)
                if num <= 0: continue
                arr = np.abs(np.asarray(vals, dtype=float))
                if arr.size == 0: continue
                chain_num_vals[chain][num].extend(arr.tolist())
            profiles = {}
            for chain, num_dict in chain_num_vals.items():
                profiles[chain] = {num: float(np.mean(vs)) for num, vs in num_dict.items()}
            return profiles

        def build_profile_matrix(profiles_per_version, chain, max_pos=MAX_IMGT_POS):
            mat = np.full((n_models, max_pos), np.nan, dtype=float)
            for mi, version_dir in enumerate(version_names):
                prof_chain = profiles_per_version[version_dir].get(chain, {})
                for num, val in prof_chain.items():
                    if 1 <= num <= max_pos:
                        mat[mi, num - 1] = val
            return mat

        def mean_ci_across_models(mat):
            mat = np.asarray(mat, dtype=float)
            with np.errstate(all="ignore"):
                mean = np.nanmean(mat, axis=0)
                n_eff = np.sum(~np.isnan(mat), axis=0)
                std = np.nanstd(mat, axis=0, ddof=1)
                sem = np.zeros_like(mean)
                valid = n_eff > 1
                sem[valid] = std[valid] / np.sqrt(n_eff[valid])
                delta = 1.96 * sem
                low = mean - delta
                high = mean + delta
            return mean, low, high, sem

        def shade_regions(ax, use_color=True):
            for region, (start, end) in imgt_regions.items():
                if use_color:
                    color = 'salmon' if 'CDR' in region else 'lightblue'
                    ax.axvspan(start, end, facecolor=color, alpha=0.2, zorder=0)
                x_mid = 0.5 * (start + end)
                ax.text(x_mid, 0.98, region, transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=10, alpha=0.9)

        def configure_xaxis(ax, label_format="plain", chain=None):
            tick_nums = list(range(1, MAX_IMGT_POS + 1, 5))
            ax.set_xticks(tick_nums)
            if label_format == "chain" and chain is not None:
                labels = [f"{chain}{n}" for n in tick_nums]
            else:
                labels = [str(n) for n in tick_nums]
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_xlim(1, MAX_IMGT_POS)

        print("\n" + "-" * 50)
        print("Running Deep-Dive PLM Analyses (Scalars, AA-Specificity, Entropy, PAE)")
        print("-" * 50)

        # Directional PLM attribution
        print("Re-calculating PLM attribution to preserve directionality...")

        per_seq_plm_residue_signed = []
        embedding_shap_explain = shap_values_dict['embedding_input']
        embedding_shap_explain = np.array(embedding_shap_explain)
        if embedding_shap_explain.ndim == 1:
            embedding_shap_explain = embedding_shap_explain.reshape(1, -1)
        elif embedding_shap_explain.ndim > 2:
            embedding_shap_explain = embedding_shap_explain.reshape(embedding_shap_explain.shape[0], -1)

        for idx_in_explain, seq in enumerate(sequences_to_explain):
            shap_vec = embedding_shap_explain[idx_in_explain]
            token_embs = token_embs_list[idx_in_explain]
            D_v = shap_vec.shape[0]
            if token_embs.shape[1] != D_v:
                D_use = min(token_embs.shape[1], D_v)
                token_embs_use = token_embs[:, :D_use]
                shap_vec_use = shap_vec[:D_use]
            else:
                token_embs_use = token_embs
                shap_vec_use = shap_vec
            per_residue_score = np.dot(token_embs_use, shap_vec_use)
            per_seq_plm_residue_signed.append(per_residue_score)

        n_explain_plm = len(sequences_to_explain)

        dense_plm_imgt_K = np.full((n_explain_plm, MAX_IMGT_POS), np.nan)
        dense_plm_imgt_L = np.full((n_explain_plm, MAX_IMGT_POS), np.nan)
        dense_plm_imgt_All = np.full((n_explain_plm, MAX_IMGT_POS), np.nan)

        aa_order = list("ACDEFGHIKLMNPQRSTVWY")
        aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}

        dense_aa_idx_K = np.full((n_explain_plm, MAX_IMGT_POS), -1, dtype=int)
        dense_aa_idx_L = np.full((n_explain_plm, MAX_IMGT_POS), -1, dtype=int)
        dense_aa_idx_All = np.full((n_explain_plm, MAX_IMGT_POS), -1, dtype=int)

        for i, mapping in enumerate(anarci_maps):
            if i >= len(per_seq_plm_residue_signed): break
            seq_str = sequences_to_explain[i]
            raw_scores = per_seq_plm_residue_signed[i]

            for orig_idx, imgt_pos in mapping.items():
                if orig_idx >= len(raw_scores) or orig_idx >= len(seq_str): continue
                chain, num, ins = parse_imgt_label(imgt_pos)
                if num < 1 or num > MAX_IMGT_POS: continue

                val = raw_scores[orig_idx]
                aa_char = seq_str[orig_idx]
                aa_id = aa_to_idx.get(aa_char, -1)

                dense_plm_imgt_All[i, num-1] = val
                dense_aa_idx_All[i, num-1] = aa_id

                if chain == 'K':
                    dense_plm_imgt_K[i, num-1] = val
                    dense_aa_idx_K[i, num-1] = aa_id
                elif chain == 'L':
                    dense_plm_imgt_L[i, num-1] = val
                    dense_aa_idx_L[i, num-1] = aa_id

        print("Generating Analysis 1: IMGT-Position vs Scalar Feature Correlations...")
        scalars_explain = scalar_features_all[explain_indices]

        def plot_position_scalar_corr(dense_mat, chain_name):
            n_pos = dense_mat.shape[1]
            n_sc = scalars_explain.shape[1]
            corr_mat = np.zeros((n_pos, n_sc))
            for p in range(n_pos):
                col_p = dense_mat[:, p]
                mask_p = ~np.isnan(col_p)
                if np.sum(mask_p) < 5: continue
                for s in range(n_sc):
                    col_s = scalars_explain[:, s]
                    valid = mask_p & (~np.isnan(col_s))
                    if np.sum(valid) < 5: continue
                    c_val = np.corrcoef(col_p[valid], col_s[valid])[0, 1]
                    corr_mat[p, s] = c_val if not np.isnan(c_val) else 0

            plt.figure(figsize=(18, 6))
            sns.heatmap(corr_mat.T, cmap='seismic', center=0, vmin=-0.6, vmax=0.6,
                        yticklabels=scalar_feature_names, cbar_kws={'label': 'Pearson r'})
            plt.title(f'Correlation: PLM Directional Impact ({chain_name}) vs Global Scalars', fontsize=16)
            plt.xlabel('IMGT Position', fontsize=14)
            ax = plt.gca()
            x_ticks = np.arange(0, MAX_IMGT_POS, 5)
            ax.set_xticks(x_ticks + 0.5)
            ax.set_xticklabels(x_ticks + 1, rotation=90, fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"shap_plm_scalar_corr_{chain_name}.png"), dpi=300)
            plt.show()

        if np.sum(~np.isnan(dense_plm_imgt_All)) > 100:
            plot_position_scalar_corr(dense_plm_imgt_All, "All_Sequences")

        print("Generating Analysis 2: Specific Amino Acid Directional Impact...")

        def get_aa_means_matrix(dense_imp, dense_aa):
            mat = np.full((20, MAX_IMGT_POS), np.nan)
            for p in range(MAX_IMGT_POS):
                imp_col = dense_imp[:, p]
                aa_col = dense_aa[:, p]
                valid = (~np.isnan(imp_col)) & (aa_col != -1)
                if np.sum(valid) == 0: continue
                p_imps = imp_col[valid]
                p_aas = aa_col[valid]
                for aa_i in range(20):
                    mask_aa = (p_aas == aa_i)
                    if np.sum(mask_aa) > 0:
                        mat[aa_i, p] = np.mean(p_imps[mask_aa])
            return mat

        mat_K = get_aa_means_matrix(dense_plm_imgt_K, dense_aa_idx_K)
        mat_L = get_aa_means_matrix(dense_plm_imgt_L, dense_aa_idx_L)
        mat_All = get_aa_means_matrix(dense_plm_imgt_All, dense_aa_idx_All)

        all_vals = np.concatenate([mat_K.flatten(), mat_L.flatten(), mat_All.flatten()])
        all_vals = all_vals[~np.isnan(all_vals)]
        limit_val = np.max(np.abs(all_vals)) if len(all_vals) > 0 else 0.1

        def save_top_30_table(matrix, chain_name):
            rows = []
            for aa_i in range(20):
                for p in range(MAX_IMGT_POS):
                    val = matrix[aa_i, p]
                    if not np.isnan(val):
                        rows.append({
                            "Position_IMGT": p + 1,
                            "Amino_Acid": aa_order[aa_i],
                            "Impact_Score": val,
                            "Direction": "Amyloidogenic" if val > 0 else "Protective",
                            "Abs_Impact": abs(val)
                        })
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values("Abs_Impact", ascending=False).head(30)
                out_path = os.path.join(SAVE_DIR, f"shap_top30_amino_acids_{chain_name}.csv")
                df.to_csv(out_path, index=False)
                print(f"  -> Top 30 table saved to: {out_path}")

        def plot_aa_importance_heatmap(aa_heatmap, chain_name, vlim):
            plt.figure(figsize=(24, 10))
            cmap = "seismic"

            ax = sns.heatmap(aa_heatmap, mask=np.isnan(aa_heatmap), cmap=cmap, center=0,
                        vmin=-vlim, vmax=vlim,
                        yticklabels=list(aa_order),
                        cbar_kws={'label': 'Mean Directional Impact'})

            plt.title(f'Amino Acid Impact by Position ({chain_name})', fontsize=24, pad=20)
            plt.ylabel('Amino Acid', fontsize=18)
            plt.xlabel('IMGT Position', fontsize=18, labelpad=60)

            ax.tick_params(axis='y', labelsize=14)
            x_ticks = np.arange(0, MAX_IMGT_POS, 2)
            ax.set_xticks(x_ticks + 0.5)
            ax.set_xticklabels(x_ticks + 1, rotation=90, fontsize=14)

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Mean Directional Impact', size=16)

            for region, (start, end) in imgt_regions.items():
                ax.axvline(x=start-1, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
                ax.axvline(x=end, color='black', linestyle='-', linewidth=0.8, alpha=0.4)

                mid = (start + end) / 2.0 - 0.5
                font_w = 'bold' if 'CDR' in region else 'normal'
                ax.text(mid, -0.10, region, ha='center', va='top',
                        fontweight=font_w, fontsize=14, transform=ax.get_xaxis_transform())

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"shap_plm_aa_specificity_{chain_name}.png"), dpi=300)
            plt.show()

        if np.sum(~np.isnan(mat_K)) > 0:
            save_top_30_table(mat_K, "Kappa")
            plot_aa_importance_heatmap(mat_K, "Kappa", limit_val)
        if np.sum(~np.isnan(mat_L)) > 0:
            save_top_30_table(mat_L, "Lambda")
            plot_aa_importance_heatmap(mat_L, "Lambda", limit_val)
        if np.sum(~np.isnan(mat_All)) > 0:
            save_top_30_table(mat_All, "Combined")
            plot_aa_importance_heatmap(mat_All, "Combined", limit_val)

        print("Generating Analysis 2b: Total Model Impact (Sequence + Structure)...")

        plddt_shap = np.array(shap_values_dict['plddt_input']).squeeze()
        pae_row_shap = np.array(shap_values_dict['pae_row_input']).squeeze()
        pae_col_shap = np.array(shap_values_dict['pae_col_input']).squeeze()

        if plddt_shap.ndim == 1: plddt_shap = plddt_shap[None, :]
        if pae_row_shap.ndim == 1: pae_row_shap = pae_row_shap[None, :]
        if pae_col_shap.ndim == 1: pae_col_shap = pae_col_shap[None, :]

        dense_total_imgt_K = np.full((n_explain_plm, MAX_IMGT_POS), np.nan)
        dense_total_imgt_L = np.full((n_explain_plm, MAX_IMGT_POS), np.nan)
        dense_total_imgt_All = np.full((n_explain_plm, MAX_IMGT_POS), np.nan)

        for i, mapping in enumerate(anarci_maps):
            if i >= len(per_seq_plm_residue_signed): break

            plm_score = per_seq_plm_residue_signed[i]
            L_seq = len(plm_score)

            def get_struct_score(arr, idx, length):
                if idx >= len(arr): return np.zeros(length)
                row = arr[idx]
                return row[:length] if len(row) >= length else np.pad(row, (0, length-len(row)))

            s_plddt = get_struct_score(plddt_shap, i, L_seq)
            s_row = get_struct_score(pae_row_shap, i, L_seq)
            s_col = get_struct_score(pae_col_shap, i, L_seq)

            total_score = plm_score + s_plddt + s_row + s_col

            for orig_idx, imgt_pos in mapping.items():
                if orig_idx >= len(total_score): continue
                chain, num, ins = parse_imgt_label(imgt_pos)
                if num < 1 or num > MAX_IMGT_POS: continue

                val = total_score[orig_idx]
                dense_total_imgt_All[i, num-1] = val
                if chain == 'K':
                    dense_total_imgt_K[i, num-1] = val
                elif chain == 'L':
                    dense_total_imgt_L[i, num-1] = val

        mat_total_K = get_aa_means_matrix(dense_total_imgt_K, dense_aa_idx_K)
        mat_total_L = get_aa_means_matrix(dense_total_imgt_L, dense_aa_idx_L)
        mat_total_All = get_aa_means_matrix(dense_total_imgt_All, dense_aa_idx_All)

        all_vals_total = np.concatenate([mat_total_K.flatten(), mat_total_L.flatten(), mat_total_All.flatten()])
        all_vals_total = all_vals_total[~np.isnan(all_vals_total)]
        limit_val_total = np.max(np.abs(all_vals_total)) if len(all_vals_total) > 0 else 0.1

        def save_total_table(matrix, name):
            rows = []
            for aa_i in range(20):
                for p in range(MAX_IMGT_POS):
                    val = matrix[aa_i, p]
                    if not np.isnan(val):
                        rows.append({
                            "Position_IMGT": p + 1,
                            "Amino_Acid": aa_order[aa_i],
                            "Impact_Score": val,
                            "Direction": "Amyloidogenic" if val > 0 else "Protective",
                            "Abs_Impact": abs(val)
                        })
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values("Abs_Impact", ascending=False).head(30)
                out_path = os.path.join(SAVE_DIR, f"shap_top30_total_impact_{name}.csv")
                df.to_csv(out_path, index=False)
                print(f"  -> Top 30 Total Impact table saved for {name}")

        if np.sum(~np.isnan(mat_total_K)) > 0:
            print("  Plotting Total Impact (Kappa)...")
            save_total_table(mat_total_K, "Kappa")
            plot_aa_importance_heatmap(mat_total_K, "Kappa (Total Model Impact)", limit_val_total)

        if np.sum(~np.isnan(mat_total_L)) > 0:
            print("  Plotting Total Impact (Lambda)...")
            save_total_table(mat_total_L, "Lambda")
            plot_aa_importance_heatmap(mat_total_L, "Lambda (Total Model Impact)", limit_val_total)

        if np.sum(~np.isnan(mat_total_All)) > 0:
            print("  Plotting Total Impact (Combined)...")
            save_total_table(mat_total_All, "Combined")
            plot_aa_importance_heatmap(mat_total_All, "Combined (Total Model Impact)", limit_val_total)

        print("Generating Analysis 3: Sequence Entropy vs. Importance (Joint Plot)...")

        def calculate_entropy(aa_col):
            valid = aa_col[aa_col != -1]
            if len(valid) < 5: return np.nan
            _, counts = np.unique(valid, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs))

        def plot_entropy_vs_importance(dense_imp, dense_aa, chain_name):
            entropies, mean_imps, region_labels, pos_labels, region_simple = [], [], [], [], []

            for p in range(MAX_IMGT_POS):
                col_imp = np.abs(dense_imp[:, p])
                if np.sum(~np.isnan(col_imp)) < 5: continue
                mean_v = np.nanmean(col_imp)
                col_aa = dense_aa[:, p]
                ent = calculate_entropy(col_aa)
                if np.isnan(ent): continue

                entropies.append(ent)
                mean_imps.append(mean_v)
                pos_labels.append(p+1)
                r_lab = "Other"
                is_cdr = False
                for region, (start, end) in imgt_regions.items():
                    if start <= (p+1) <= end:
                        r_lab = region
                        is_cdr = 'CDR' in region
                        break
                region_labels.append(r_lab)
                region_simple.append('CDR' if is_cdr else 'FR')

            df_scatter = pd.DataFrame({
                'Entropy': entropies, 'Importance': mean_imps,
                'Region': region_labels, 'Type': region_simple, 'Position': pos_labels
            })

            g = sns.JointGrid(data=df_scatter, x='Entropy', y='Importance', height=10, ratio=4)
            sns.scatterplot(data=df_scatter, x='Entropy', y='Importance',
                            hue='Type', style='Type', s=120, alpha=0.7, ax=g.ax_joint)
            sns.kdeplot(data=df_scatter, x='Entropy', hue='Type', fill=True, legend=False, ax=g.ax_marg_x)
            sns.kdeplot(data=df_scatter, y='Importance', hue='Type', fill=True, legend=False, ax=g.ax_marg_y)

            g.ax_joint.set_xlim(left=0)

            top_pts = df_scatter.nlargest(10, 'Importance').sort_values('Entropy')
            offsets = [5, -10, 8, -8, 12, -12]
            for i, (_, row) in enumerate(top_pts.iterrows()):
                off_y = offsets[i % len(offsets)]
                g.ax_joint.annotate(f"{int(row['Position'])}",
                                    (row['Entropy'], row['Importance']),
                                    xytext=(0, off_y), textcoords='offset points',
                                    fontsize=11, fontweight='bold', color='black',
                                    ha='center', arrowprops=dict(arrowstyle='-', color='grey', alpha=0.5))

            g.fig.suptitle(f'Entropy vs. Model Impact ({chain_name})', y=1.02, fontsize=20)
            g.set_axis_labels('Sequence Entropy (Bits)', 'Mean Absolute Impact', fontsize=16)
            sns.move_legend(g.ax_joint, "upper right", title='Region Type')
            plt.savefig(os.path.join(SAVE_DIR, f"shap_plm_entropy_vs_importance_{chain_name}.png"), dpi=300, bbox_inches='tight')
            plt.show()

        if np.sum(~np.isnan(dense_plm_imgt_K)) > 50:
            plot_entropy_vs_importance(dense_plm_imgt_K, dense_aa_idx_K, "Kappa")
        if np.sum(~np.isnan(dense_plm_imgt_L)) > 50:
            plot_entropy_vs_importance(dense_plm_imgt_L, dense_aa_idx_L, "Lambda")
        if np.sum(~np.isnan(dense_plm_imgt_All)) > 50:
            plot_entropy_vs_importance(dense_plm_imgt_All, dense_aa_idx_All, "Combined")

        print("Generating Analysis 4: IMGT-Aligned PAE Matrix SHAP Heatmap...")

        pae_shap_values_raw = shap_values_dict['pae_input']
        if pae_shap_values_raw.ndim == 4:
            pae_shap_values_raw = pae_shap_values_raw.squeeze(-1)
        pae_shap_abs = np.abs(pae_shap_values_raw)

        pae_accum_K = np.zeros((MAX_IMGT_POS, MAX_IMGT_POS))
        pae_count_K = np.zeros((MAX_IMGT_POS, MAX_IMGT_POS))
        pae_accum_L = np.zeros((MAX_IMGT_POS, MAX_IMGT_POS))
        pae_count_L = np.zeros((MAX_IMGT_POS, MAX_IMGT_POS))
        pae_accum_All = np.zeros((MAX_IMGT_POS, MAX_IMGT_POS))
        pae_count_All = np.zeros((MAX_IMGT_POS, MAX_IMGT_POS))

        for i, mapping in enumerate(anarci_maps):
            if i >= pae_shap_abs.shape[0]: break
            if not mapping: continue
            first_key = next(iter(mapping))
            c_char, _, _ = parse_imgt_label(mapping[first_key])

            current_accum, current_count = None, None
            if c_char == 'K':
                current_accum, current_count = pae_accum_K, pae_count_K
            elif c_char == 'L':
                current_accum, current_count = pae_accum_L, pae_count_L

            all_accum, all_count = pae_accum_All, pae_count_All

            seq_len = pae_shap_abs.shape[1]
            trans = np.full(seq_len, -1, dtype=int)
            for orig, lbl in mapping.items():
                if orig < seq_len:
                    _, num, _ = parse_imgt_label(lbl)
                    if 1 <= num <= MAX_IMGT_POS: trans[orig] = num - 1

            valid_indices = np.where(trans != -1)[0]
            if len(valid_indices) < 2: continue
            sub_shap = pae_shap_abs[i][np.ix_(valid_indices, valid_indices)]
            imgt_indices = trans[valid_indices]

            for r_i, row_imgt in enumerate(imgt_indices):
                for c_i, col_imgt in enumerate(imgt_indices):
                    val = float(sub_shap[r_i, c_i])
                    all_accum[row_imgt, col_imgt] += val
                    all_count[row_imgt, col_imgt] += 1
                    if current_accum is not None:
                        current_accum[row_imgt, col_imgt] += val
                        current_count[row_imgt, col_imgt] += 1

        def plot_pae_heatmap(accum, count, chain_name):
            with np.errstate(invalid='ignore'):
                mean_mat = accum / count
            mean_mat[np.isnan(mean_mat)] = 0

            plt.figure(figsize=(14, 12))

            ax = sns.heatmap(mean_mat, cmap='viridis', square=True,
                       cbar_kws={'label': 'Mean |SHAP| on PAE Interaction'})

            ax.invert_yaxis()

            plt.title(f'IMGT-Aligned PAE Interaction Importance ({chain_name})', fontsize=22, pad=60)
            plt.xlabel('IMGT Position (Aligned On)', fontsize=18)
            plt.ylabel('IMGT Position (Residue Error)', fontsize=18)

            ticks = np.arange(0, MAX_IMGT_POS, 10)
            ax.set_xticks(ticks + 0.5)
            ax.set_xticklabels(ticks + 1, rotation=0, fontsize=14)
            ax.set_yticks(ticks + 0.5)
            ax.set_yticklabels(ticks + 1, rotation=0, fontsize=14)

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Mean |SHAP| on PAE Interaction', size=16)

            for region, (start, end) in imgt_regions.items():
                s_idx = start - 1
                e_idx = end

                ax.axvline(x=s_idx, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
                ax.axvline(x=e_idx, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
                ax.axhline(y=s_idx, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
                ax.axhline(y=e_idx, color='white', linestyle='--', linewidth=0.5, alpha=0.3)

                mid = (s_idx + e_idx) / 2
                font_w = 'bold' if 'CDR' in region else 'normal'

                ax.text(mid, MAX_IMGT_POS + 2, region, color='black', ha='center', va='bottom',
                        fontsize=12, fontweight=font_w)

                ax.text(MAX_IMGT_POS + 1.5, mid, region, color='black', ha='left', va='center',
                        fontsize=12, fontweight=font_w)

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"shap_pae_heatmap_imgt_{chain_name}.png"), dpi=300)
            plt.show()

        if np.sum(pae_count_K) > 0:
            plot_pae_heatmap(pae_accum_K, pae_count_K, "Kappa")
        if np.sum(pae_count_L) > 0:
            plot_pae_heatmap(pae_accum_L, pae_count_L, "Lambda")
        if np.sum(pae_count_All) > 0:
            plot_pae_heatmap(pae_accum_All, pae_count_All, "Combined")

        print("\nCalculating PLM Model-level Matrices for Line Plots & Stats...")

        plm_K_model_mat = np.full((n_models, MAX_IMGT_POS), np.nan, dtype=float)
        plm_L_model_mat = np.full((n_models, MAX_IMGT_POS), np.nan, dtype=float)

        positions_imgt = np.arange(1, MAX_IMGT_POS + 1)

        for mi, version_dir in enumerate(version_names):
            emb_shap_v = np.array(shap_values_per_version[version_dir]['embedding_input'])
            if emb_shap_v.ndim == 1: emb_shap_v = emb_shap_v.reshape(1, -1)
            elif emb_shap_v.ndim > 2: emb_shap_v = emb_shap_v.reshape(emb_shap_v.shape[0], -1)

            sum_K = np.zeros(MAX_IMGT_POS, dtype=float)
            cnt_K = np.zeros(MAX_IMGT_POS, dtype=int)
            sum_L = np.zeros(MAX_IMGT_POS, dtype=float)
            cnt_L = np.zeros(MAX_IMGT_POS, dtype=int)

            for i, mapping in enumerate(anarci_maps):
                if i >= len(emb_shap_v): break

                shap_vec = emb_shap_v[i]
                w = np.abs(shap_vec)
                if w.sum() > 0: w = w / w.sum()

                token_embs = token_embs_list[i]
                D_v = shap_vec.shape[0]
                if token_embs.shape[1] != D_v:
                    token_embs_use = token_embs[:, :D_v]
                    w_use = w[:D_v]
                else:
                    token_embs_use = token_embs
                    w_use = w

                contrib = np.abs(token_embs_use) * w_use[None, :]
                per_residue = contrib.sum(axis=1)

                for orig_idx, imgt_pos in mapping.items():
                    if orig_idx >= len(per_residue): continue
                    chain, num, ins = parse_imgt_label(imgt_pos)
                    if num < 1 or num > MAX_IMGT_POS: continue

                    if chain == 'K':
                        sum_K[num-1] += per_residue[orig_idx]
                        cnt_K[num-1] += 1
                    elif chain == 'L':
                        sum_L[num-1] += per_residue[orig_idx]
                        cnt_L[num-1] += 1

            with np.errstate(invalid='ignore'):
                plm_K_model_mat[mi, :] = sum_K / cnt_K
                plm_L_model_mat[mi, :] = sum_L / cnt_L

        def mean_ci_across_models_plm(mat):
            mat = np.asarray(mat, dtype=float)
            with np.errstate(all="ignore"):
                mean = np.nanmean(mat, axis=0)
                n_eff = np.sum(~np.isnan(mat), axis=0)
                std = np.nanstd(mat, axis=0, ddof=1)
                sem = np.zeros_like(mean)
                valid = n_eff > 1
                sem[valid] = std[valid] / np.sqrt(n_eff[valid])
                delta = 1.96 * sem
                low = mean - delta
                high = mean + delta
            return mean, low, high

        plm_K_mean_mod, plm_K_low_mod, plm_K_high_mod = mean_ci_across_models_plm(plm_K_model_mat)
        plm_L_mean_mod, plm_L_low_mod, plm_L_high_mod = mean_ci_across_models_plm(plm_L_model_mat)

        fig, ax = plt.subplots(figsize=(20, 7))

        if not np.all(np.isnan(plm_K_mean_mod)):
            maskK = ~np.isnan(plm_K_mean_mod)
            ax.plot(positions_imgt[maskK], plm_K_mean_mod[maskK],
                    color="blue", lw=2, label="Kappa")
            ax.fill_between(
                positions_imgt[maskK],
                plm_K_low_mod[maskK],
                plm_K_high_mod[maskK],
                color="blue", alpha=0.2,
            )

        if not np.all(np.isnan(plm_L_mean_mod)):
            maskL = ~np.isnan(plm_L_mean_mod)
            ax.plot(positions_imgt[maskL], plm_L_mean_mod[maskL],
                    color="orange", lw=2, label="Lambda")
            ax.fill_between(
                positions_imgt[maskL],
                plm_L_low_mod[maskL],
                plm_L_high_mod[maskL],
                color="orange", alpha=0.2,
            )

        shade_regions(ax)
        configure_xaxis(ax, label_format="plain")

        ax.set_title("PLM IMGT-Aligned Positional Importance (Kappa + Lambda overlay)", fontsize=20)
        ax.set_xlabel("IMGT Position", fontsize=16)
        ax.set_ylabel("Mean Weighted PLM Contribution (95% CI)", fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", fontsize=14)

        y_arrays = []
        if not np.all(np.isnan(plm_K_high_mod)): y_arrays.append(plm_K_high_mod[~np.isnan(plm_K_high_mod)])
        if not np.all(np.isnan(plm_L_high_mod)): y_arrays.append(plm_L_high_mod[~np.isnan(plm_L_high_mod)])
        if y_arrays:
            global_max = np.nanmax(np.concatenate(y_arrays))
            if np.isfinite(global_max) and global_max > 0:
                ax.set_ylim(0, global_max * 1.2)

        fig.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "shap_plm_imgt_overlay.png"), dpi=300)
        plt.show()

        print("\nAggregating SHAP values to IMGT coordinates per model (for positional curves)...")
        aligned_plddt_per_version = {}
        aligned_row_per_version = {}
        aligned_col_per_version = {}

        for version_dir in version_names:
            plddt_v = shap_values_per_version[version_dir]['plddt_input']
            row_v = shap_values_per_version[version_dir]['pae_row_input']
            col_v = shap_values_per_version[version_dir]['pae_col_input']

            aligned_plddt_per_version[version_dir] = aggregate_to_imgt_for_version(plddt_v, anarci_maps)
            aligned_row_per_version[version_dir] = aggregate_to_imgt_for_version(row_v, anarci_maps)
            aligned_col_per_version[version_dir] = aggregate_to_imgt_for_version(col_v, anarci_maps)

        plddt_profiles_per_version = {
            v: build_numeric_profile_from_aligned(aligned_plddt_per_version[v])
            for v in version_names
        }
        row_profiles_per_version = {
            v: build_numeric_profile_from_aligned(aligned_row_per_version[v])
            for v in version_names
        }
        col_profiles_per_version = {
            v: build_numeric_profile_from_aligned(aligned_col_per_version[v])
            for v in version_names
        }

        plddt_K_mat = build_profile_matrix(plddt_profiles_per_version, 'K')
        plddt_L_mat = build_profile_matrix(plddt_profiles_per_version, 'L')
        row_K_mat = build_profile_matrix(row_profiles_per_version, 'K')
        row_L_mat = build_profile_matrix(row_profiles_per_version, 'L')
        col_K_mat = build_profile_matrix(col_profiles_per_version, 'K')
        col_L_mat = build_profile_matrix(col_profiles_per_version, 'L')

        positions_imgt = np.arange(1, MAX_IMGT_POS + 1)

        plddt_K_mean, plddt_K_low, plddt_K_high, _ = mean_ci_across_models(plddt_K_mat)
        plddt_L_mean, plddt_L_low, plddt_L_high, _ = mean_ci_across_models(plddt_L_mat)
        row_K_mean, row_K_low, row_K_high, _ = mean_ci_across_models(row_K_mat)
        row_L_mean, row_L_low, row_L_high, _ = mean_ci_across_models(row_L_mat)
        col_K_mean, col_K_low, col_K_high, _ = mean_ci_across_models(col_K_mat)
        col_L_mean, col_L_low, col_L_high, _ = mean_ci_across_models(col_L_mat)

        def plot_overlay_with_ci(mean_K, low_K, high_K,
                                 mean_L, low_L, high_L,
                                 title, filename,
                                 colorK='blue', colorL='orange'):
            if np.all(np.isnan(mean_K)) and np.all(np.isnan(mean_L)):
                print(f"No data for {title}, skipping.")
                return

            fig, ax = plt.subplots(figsize=(20, 7))

            if not np.all(np.isnan(mean_K)):
                maskK = ~np.isnan(mean_K)
                ax.plot(positions_imgt[maskK], mean_K[maskK],
                        color=colorK, lw=2, label='Kappa')
                ax.fill_between(
                    positions_imgt[maskK],
                    low_K[maskK],
                    high_K[maskK],
                    color=colorK,
                    alpha=0.2,
                )

            if not np.all(np.isnan(mean_L)):
                maskL = ~np.isnan(mean_L)
                ax.plot(positions_imgt[maskL], mean_L[maskL],
                        color=colorL, lw=2, label='Lambda')
                ax.fill_between(
                    positions_imgt[maskL],
                    low_L[maskL],
                    high_L[maskL],
                    color=colorL,
                    alpha=0.2,
                )

            shade_regions(ax)
            configure_xaxis(ax, label_format="plain")

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('IMGT Position', fontsize=12)
            ax.set_ylabel('Mean per-model |SHAP| (95% CI)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper left')

            fig.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300)
            plt.show()

        def plot_chain_with_ci(chain, mean, low, high, title, color, filename):
            if np.all(np.isnan(mean)):
                print(f"No data for {title}, skipping.")
                return

            mask = ~np.isnan(mean)
            if not np.any(mask):
                print(f"No data (after masking) for {title}, skipping.")
                return

            fig, ax = plt.subplots(figsize=(20, 7))
            ax.plot(positions_imgt[mask], mean[mask], color=color, lw=2)
            ax.fill_between(
                positions_imgt[mask],
                low[mask],
                high[mask],
                color=color,
                alpha=0.2,
            )

            shade_regions(ax)
            configure_xaxis(ax, label_format="chain", chain=chain)

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('IMGT Position', fontsize=12)
            ax.set_ylabel('Mean per-model |SHAP| (95% CI)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)

            y_max_chain = np.nanmax(high[mask])
            if np.isfinite(y_max_chain) and y_max_chain > 0:
                ax.set_ylim(0, y_max_chain * 1.2)

            fig.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300)
            plt.show()

        print("\nGenerating IMGT-aligned positional plots with 95% CI across models...")

        plot_overlay_with_ci(
            plddt_K_mean, plddt_K_low, plddt_K_high,
            plddt_L_mean, plddt_L_low, plddt_L_high,
            'pLDDT IMGT-Aligned Positional Importance (Kappa + Lambda overlay)',
            'shap_imgt_plddt.png',
            colorK='blue', colorL='orange',
        )
        plot_overlay_with_ci(
            row_K_mean, row_K_low, row_K_high,
            row_L_mean, row_L_low, row_L_high,
            'PAE Row Mean IMGT-Aligned Positional Importance (Kappa + Lambda overlay)',
            'shap_imgt_pae_row.png',
            colorK='blue', colorL='orange',
        )
        plot_overlay_with_ci(
            col_K_mean, col_K_low, col_K_high,
            col_L_mean, col_L_low, col_L_high,
            'PAE Col Mean IMGT-Aligned Positional Importance (Kappa + Lambda overlay)',
            'shap_imgt_pae_col.png',
            colorK='blue', colorL='orange',
        )

        plot_chain_with_ci(
            'K', plddt_K_mean, plddt_K_low, plddt_K_high,
            'pLDDT IMGT-Aligned Positional Importance (Kappa only)',
            'blue', 'shap_imgt_plddt_K.png',
        )
        plot_chain_with_ci(
            'L', plddt_L_mean, plddt_L_low, plddt_L_high,
            'pLDDT IMGT-Aligned Positional Importance (Lambda only)',
            'blue', 'shap_imgt_plddt_L.png',
        )

        plot_chain_with_ci(
            'K', row_K_mean, row_K_low, row_K_high,
            'PAE Row Mean IMGT-Aligned Positional Importance (Kappa only)',
            'green', 'shap_imgt_pae_row_K.png',
        )
        plot_chain_with_ci(
            'L', row_L_mean, row_L_low, row_L_high,
            'PAE Row Mean IMGT-Aligned Positional Importance (Lambda only)',
            'green', 'shap_imgt_pae_row_L.png',
        )

        plot_chain_with_ci(
            'K', col_K_mean, col_K_low, col_K_high,
            'PAE Col Mean IMGT-Aligned Positional Importance (Kappa only)',
            'purple', 'shap_imgt_pae_col_K.png',
        )
        plot_chain_with_ci(
            'L', col_L_mean, col_L_low, col_L_high,
            'PAE Col Mean IMGT-Aligned Positional Importance (Lambda only)',
            'purple', 'shap_imgt_pae_col_L.png',
        )

        print("\nComputing region-level SHAP summaries (CDR vs FR) with SEM across models...")

        def compute_region_means_for_aligned(aligned_dict):
            region_vals = defaultdict(lambda: defaultdict(list))
            for pos_label, vals in aligned_dict.items():
                chain, num, ins = parse_imgt_label(pos_label)
                if num <= 0:
                    continue
                arr = np.abs(np.asarray(vals, dtype=float))
                if arr.size == 0:
                    continue
                for region, (start, end) in imgt_regions.items():
                    if start <= num <= end:
                        region_vals[chain][region].extend(arr.tolist())
            region_means = {}
            for chain, reg_dict in region_vals.items():
                region_means[chain] = {}
                for region, vals_list in reg_dict.items():
                    arr = np.asarray(vals_list, dtype=float)
                    if arr.size == 0:
                        continue
                    region_means[chain][region] = float(arr.mean())
            return region_means

        regions = list(imgt_regions.keys())

        def prepare_region_values(region_means_per_version):
            chains = sorted({
                chain
                for v in version_names
                for chain in region_means_per_version[v].keys()
            })
            out = {}
            for chain in chains:
                mat = np.full((n_models, len(regions)), np.nan, dtype=float)
                for mi, version_dir in enumerate(version_names):
                    reg_means_v = region_means_per_version[version_dir].get(chain, {})
                    for ri, region in enumerate(regions):
                        if region in reg_means_v:
                            mat[mi, ri] = reg_means_v[region]
                out[chain] = mat
            return out, chains

        def plot_region_bar(region_stats, title, filename, ylabel):
            if not region_stats:
                print(f"No data for {title}, skipping.")
                return

            chains = sorted(region_stats.keys())
            x = np.arange(len(regions))
            width = 0.35 if len(chains) > 1 else 0.6

            all_means = np.concatenate([region_stats[ch][0] for ch in chains])
            all_sems = np.concatenate([region_stats[ch][1] for ch in chains])
            y_max = np.nanmax(all_means + all_sems)

            fig, ax = plt.subplots(figsize=(9, 6))
            for i, chain in enumerate(chains):
                means, sems, pmat = region_stats[chain]
                offset = (i - (len(chains) - 1) / 2.0) * width
                bars = ax.bar(x + offset, means, width, label=f"{chain} chain", alpha=0.85)

                if np.all(np.isnan(means)):
                    ref_idx = None
                else:
                    ref_idx = int(np.nanargmax(means))

                for j, bar in enumerate(bars):
                    y = means[j]
                    yerr = sems[j]
                    if np.isnan(y):
                        continue
                    center_x = bar.get_x() + bar.get_width() / 2.0
                    ax.errorbar(
                        x=center_x,
                        y=y,
                        yerr=yerr,
                        fmt='none',
                        ecolor='black',
                        elinewidth=1,
                        capsize=3,
                    )
                    if ref_idx is None or j == ref_idx:
                        continue
                    p_val = pmat[j, ref_idx]
                    stars = significance_label(p_val)
                    if stars:
                        ax.text(
                            center_x,
                            y + (yerr if not np.isnan(yerr) else 0) + 0.02 * y_max,
                            stars,
                            ha='center',
                            va='bottom',
                            fontsize=12,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(regions)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            fig.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300)
            plt.show()

        region_means_plddt_per_version = {
            v: compute_region_means_for_aligned(aligned_plddt_per_version[v])
            for v in version_names
        }
        plddt_vals_per_chain, chains_plddt = prepare_region_values(region_means_plddt_per_version)
        region_stats_plddt = {}
        for chain in chains_plddt:
            mat = plddt_vals_per_chain[chain]
            means, sems, _, pmat = summarize_across_models(
                mat, regions, prefix=f"region_plddt_chain_{chain}",
                allow_nan=True, save_dir=SAVE_DIR
            )
            region_stats_plddt[chain] = (means, sems, pmat)

        plot_region_bar(
            region_stats_plddt,
            'Region-level Mean |SHAP| (pLDDT, IMGT regions)',
            'shap_imgt_region_plddt.png',
            ylabel='Mean per-model region |SHAP|',
        )

        region_means_row_per_version = {
            v: compute_region_means_for_aligned(aligned_row_per_version[v])
            for v in version_names
        }
        row_vals_per_chain, chains_row = prepare_region_values(region_means_row_per_version)
        region_stats_row = {}
        for chain in chains_row:
            mat = row_vals_per_chain[chain]
            means, sems, _, pmat = summarize_across_models(
                mat, regions, prefix=f"region_pae_row_chain_{chain}",
                allow_nan=True, save_dir=SAVE_DIR
            )
            region_stats_row[chain] = (means, sems, pmat)

        plot_region_bar(
            region_stats_row,
            'Region-level Mean |SHAP| (PAE Row Mean, IMGT regions)',
            'shap_imgt_region_pae_row.png',
            ylabel='Mean per-model region |SHAP|',
        )

        region_means_col_per_version = {
            v: compute_region_means_for_aligned(aligned_col_per_version[v])
            for v in version_names
        }
        col_vals_per_chain, chains_col = prepare_region_values(region_means_col_per_version)
        region_stats_col = {}
        for chain in chains_col:
            mat = col_vals_per_chain[chain]
            means, sems, _, pmat = summarize_across_models(
                mat, regions, prefix=f"region_pae_col_chain_{chain}",
                allow_nan=True, save_dir=SAVE_DIR
            )
            region_stats_col[chain] = (means, sems, pmat)

        plot_region_bar(
            region_stats_col,
            'Region-level Mean |SHAP| (PAE Col Mean, IMGT regions)',
            'shap_imgt_region_pae_col.png',
            ylabel='Mean per-model region |SHAP|',
        )

        print("\nComputing region-level PLM summaries (IMGT regions) with SEM across models...")

        def build_plm_region_matrix(chain_mat):
            mat = np.full((n_models, len(regions)), np.nan, dtype=float)
            for ri, region in enumerate(regions):
                start, end = imgt_regions[region]
                region_slice = chain_mat[:, start-1:end]
                with np.errstate(all="ignore"):
                    mat[:, ri] = np.nanmean(region_slice, axis=1)
            return mat

        plm_region_vals_per_chain = {}
        if not np.all(np.isnan(plm_K_model_mat)):
            plm_region_vals_per_chain['K'] = build_plm_region_matrix(plm_K_model_mat)
        if not np.all(np.isnan(plm_L_model_mat)):
            plm_region_vals_per_chain['L'] = build_plm_region_matrix(plm_L_model_mat)

        region_stats_plm = {}
        for chain, mat in plm_region_vals_per_chain.items():
            means, sems, _, pmat = summarize_across_models(
                mat,
                regions,
                prefix=f"region_plm_chain_{chain}",
                allow_nan=True,
                save_dir=SAVE_DIR,
            )
            region_stats_plm[chain] = (means, sems, pmat)

        if region_stats_plm:
            plot_region_bar(
                region_stats_plm,
                'Region-level Mean PLM Contribution (IMGT regions)',
                'shap_imgt_region_plm.png',
                ylabel='Mean per-model PLM contribution',
            )

    except Exception as e:
        print(f"\nAn error occurred during IMGT-aligned analysis: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping this part of the analysis.")

    print(f"\nAnalysis complete. Plots and stats saved to '{SAVE_DIR}' directory.")


if __name__ == "__main__":
    # ANARCI / HMMER sanity check
    try:
        from anarci import anarci as _anarci_test

        if not os.path.exists(os.path.join(HMMER_PATH, "hmmscan")):
            raise FileNotFoundError("hmmscan not found in HMMER_PATH")
    except (ImportError, FileNotFoundError) as e:
        print("=" * 60)
        print("ERROR: Could not find 'anarci' library or its 'hmmscan' dependency.")
        print(
            f"Please check that 'anarci' is installed in your environment and HMMER_PATH is correct: '{HMMER_PATH}'"
        )
        print(f"Underlying error: {e}")
        print("=" * 60)
        raise SystemExit(1)

    run_shap_analysis()