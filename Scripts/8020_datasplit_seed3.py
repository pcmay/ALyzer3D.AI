import os
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

# --------- CONFIGURE THESE PATHS ---------
BASE = "/Users/PeterMay/Downloads/amyloidosis"

DATASET_DIRS = {
    "colabfold_combined_Kopie": os.path.join(BASE, "colabfold_combined_Kopie"),
    "rev_colab_test": os.path.join(BASE, "rev_colab_test"),
    # If you still have a space in the folder name, either escape it or (better)
    # rename the folder to "rev_colabfold_combined_new_unique" and update here:
    "rev_colabfold_combined_new_unique": os.path.join(BASE, "rev_colabfold_combined_new unique"),
}

# Where to put the virtual split (symlinks)
COMBINED_ROOT = os.path.join(BASE, "combined_80_20_seed3")
TRAIN_ROOT = os.path.join(COMBINED_ROOT, "train")
TEST_ROOT = os.path.join(COMBINED_ROOT, "test")

CLASS_NAMES = ["non_amyloid", "amyloid"]


def collect_samples(dataset_name, dataset_root):
    """
    Collect one (json, pdb) pair per protein in a dataset/class,
    using the same base_name logic as your evaluation script.
    """
    samples = []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Class dir not found: {class_dir}")
            continue

        protein_files = defaultdict(list)
        for f in os.listdir(class_dir):
            # Match evaluation.py: base_name = ... split('_scores_rank_')[0] ...
            base = f.split('_scores_rank_')[0].split('_unrelaxed_rank_')[0]
            protein_files[base].append(f)

        for base, files in protein_files.items():
            # Prefer rank_001 / rank_1, fall back to any json/pdb
            json_file = next(
                (f for f in files if ('_rank_001' in f or '_rank_1' in f) and f.endswith('.json')),
                next((f for f in files if f.endswith('.json')), None)
            )
            pdb_file = next(
                (f for f in files if ('_rank_001' in f or '_rank_1' in f) and f.endswith('.pdb')),
                next((f for f in files if f.endswith('.pdb')), None)
            )

            if not json_file or not pdb_file:
                continue

            samples.append({
                "dataset": dataset_name,
                "class_name": class_name,
                "label": label_idx,
                "class_dir": class_dir,
                "json": json_file,
                "pdb": pdb_file,
                "base": base,
            })

    print(f"[INFO] {dataset_name}: collected {len(samples)} proteins")
    return samples


def make_symlink(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # Remove existing file/symlink if present
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    # 1) Collect all samples from all three datasets
    all_samples = []
    for dname, droot in DATASET_DIRS.items():
        if not os.path.isdir(droot):
            print(f"[WARN] Dataset root not found: {droot}")
            continue
        all_samples.extend(collect_samples(dname, droot))

    if not all_samples:
        print("No samples found - check your paths.")
        return

    print(f"\n[INFO] Total proteins collected across all datasets: {len(all_samples)}")

    # 2) Build stratification labels: class + dataset so each dataset/class combination
    #    is roughly preserved across train and test.
    idx = np.arange(len(all_samples))
    strata = np.array([f"{s['label']}_{s['dataset']}" for s in all_samples])

    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=3,
        stratify=strata
    )

    print(f"[INFO] Train size: {len(train_idx)}  |  Test size: {len(test_idx)}")

    # 3) Create symlinks for train and test splits
    def link_split(indices, out_root, split_name):
        counts = {c: 0 for c in CLASS_NAMES}
        for i in indices:
            s = all_samples[i]
            src_json = os.path.join(s["class_dir"], s["json"])
            src_pdb = os.path.join(s["class_dir"], s["pdb"])

            dst_class_dir = os.path.join(out_root, s["class_name"])
            dst_json = os.path.join(dst_class_dir, s["json"])
            dst_pdb = os.path.join(dst_class_dir, s["pdb"])

            make_symlink(src_json, dst_json)
            make_symlink(src_pdb, dst_pdb)

            counts[s["class_name"]] += 1

        print(f"[INFO] {split_name} split per class:")
        for c in CLASS_NAMES:
            print(f"   {c}: {counts[c]}")

    link_split(train_idx, TRAIN_ROOT, "TRAIN")
    link_split(test_idx, TEST_ROOT, "TEST")

    print("\n[DONE] Virtual 80/20 split created:")
    print(f"   Train dir: {TRAIN_ROOT}")
    print(f"   Test dir : {TEST_ROOT}")


if __name__ == "__main__":
    main()
