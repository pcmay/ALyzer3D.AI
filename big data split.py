import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import collections

# --- CONFIGURATION ---
# 1. List ALL source directories containing your raw data
SOURCE_DIRS = [
    "/Users/PeterMay/Downloads/amyloidosis/colabfold_combined",
    "/Users/PeterMay/Downloads/amyloidosis/colab_test",
    "/Users/PeterMay/Downloads/amyloidosis/colabfold_combined_new unique"
]

# 2. Define the destination for your new, clean data splits
SPLIT_DATA_DIR = "amyloid_data_split"

# 3. Define split ratios
TEST_SIZE = 0.15  # 15% of the data will be for the final test set
VALIDATION_SIZE = 0.10 # 10% of the original data will be for validation

# --- SCRIPT LOGIC ---

def collect_file_paths(source_dirs):
    """
    Scans source directories to find unique protein PDB and JSON files.
    It handles cases where multiple files exist for one protein by preferring rank 1.
    """
    protein_files = collections.defaultdict(lambda: {'pdb': None, 'json': None, 'class': None})

    print("Scanning source directories to find unique protein files...")

    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            print(f"Warning: Source directory not found, skipping: {source_dir}")
            continue
            
        for class_name in ["amyloid", "non_amyloid"]:
            class_dir = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                # Normalize name by removing colabfold suffixes to get a base ID
                base_name = filename.split('_scores_rank_')[0].split('_unrelaxed_rank_')[0].replace('.pdb', '').replace('.json', '')

                # Store the class for this protein
                protein_files[base_name]['class'] = class_name
                
                # Prioritize rank 1 files, but take any if rank 1 isn't found
                if filename.endswith(".pdb"):
                    current_pdb = protein_files[base_name].get('pdb')
                    if not current_pdb or ('_rank_001' in filename or '_rank_1' in filename):
                         protein_files[base_name]['pdb'] = os.path.join(class_dir, filename)

                elif filename.endswith(".json"):
                    current_json = protein_files[base_name].get('json')
                    if not current_json or ('_rank_001' in filename or '_rank_1' in filename):
                        protein_files[base_name]['json'] = os.path.join(class_dir, filename)

    # Filter out any entries that don't have both a PDB and a JSON file
    amyloid_list = []
    non_amyloid_list = []
    for base_name, files in protein_files.items():
        if files['pdb'] and files['json']:
            if files['class'] == 'amyloid':
                amyloid_list.append(files)
            else:
                non_amyloid_list.append(files)
                
    return amyloid_list, non_amyloid_list


def split_and_copy_files(amyloid_files, non_amyloid_files, dest_dir):
    """
    Splits the data into train, validation, and test sets and copies them
    to a new directory structure.
    """
    print("\nSplitting data and copying files...")
    
    # Clean and create destination directory
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    # Perform splitting for each class separately to maintain ratio
    for class_name, all_files in [("amyloid", amyloid_files), ("non_amyloid", non_amyloid_files)]:
        if not all_files:
            continue

        # First split: separate out the test set
        train_val_files, test_files = train_test_split(
            all_files,
            test_size=TEST_SIZE,
            random_state=42
        )

        # Second split: separate out the validation set from the remaining data
        # The new validation size must be recalculated relative to the remaining data
        val_size_recalculated = VALIDATION_SIZE / (1.0 - TEST_SIZE)
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_size_recalculated,
            random_state=42
        )

        # Copy files to the new structured directory
        for split_name, split_data in [("train", train_files), ("validation", val_files), ("test", test_files)]:
            split_path = os.path.join(dest_dir, split_name, class_name)
            os.makedirs(split_path, exist_ok=True)
            
            print(f"Copying {len(split_data)} {class_name} files to '{split_name}' set...")
            for file_group in tqdm(split_data, desc=f"Copying {class_name} {split_name} files"):
                shutil.copy(file_group['pdb'], split_path)
                shutil.copy(file_group['json'], split_path)

def main():
    """Main function to run the data preparation pipeline."""
    amyloid_files, non_amyloid_files = collect_file_paths(SOURCE_DIRS)

    total_amyloid = len(amyloid_files)
    total_non_amyloid = len(non_amyloid_files)
    total_files = total_amyloid + total_non_amyloid

    if total_files == 0:
        print("No files found. Please check your SOURCE_DIRS configuration.")
        return

    print(f"\nFound {total_files} unique protein entries in total.")
    print(f"  - Amyloid: {total_amyloid}")
    print(f"  - Non-Amyloid: {total_non_amyloid}")

    split_and_copy_files(amyloid_files, non_amyloid_files, SPLIT_DATA_DIR)

    print(f"\nData preparation complete.")
    print(f"New dataset is ready for training in: '{SPLIT_DATA_DIR}'")


if __name__ == "__main__":
    main()
