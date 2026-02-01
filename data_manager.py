"""
Data manager to harmonize CIFAKE, Fakeddit, and NewsCLIPpings datasets
into the MisinfoDataset format.

Output:
- harmonized_train_data.pkl: list of dicts with keys: text, image_path, label
- newscippings_genuine_seed.json: genuine NewsCLIPpings entries (for FAISS seeding)

Note: NewsCLIPpings preprocessing is commented out per request.
"""

import os
import json
import pickle
import random
from typing import List, Dict, Tuple


def set_seed(seed: int = 42) -> None:
    random.seed(seed)


def balanced_sample(items: List[Dict], per_label: int) -> List[Dict]:
    """
    Ensure a 50/50 split between label 0 and 1.
    """
    label0 = [x for x in items if x.get("label") == 0]
    label1 = [x for x in items if x.get("label") == 1]

    random.shuffle(label0)
    random.shuffle(label1)

    n = min(per_label, len(label0), len(label1))
    return label0[:n] + label1[:n]


def process_cifake(cifake_root: str, sample_per_label: int) -> List[Dict]:
    """
    CIFAKE structure:
    - train/REAL (label 0)
    - train/FAKE (label 1)
    Text is unavailable, so use placeholder text.
    """
    entries = []
    real_dir = os.path.join(cifake_root, "train", "REAL")
    fake_dir = os.path.join(cifake_root, "train", "FAKE")

    for label_dir, label in [(real_dir, 0), (fake_dir, 1)]:
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            fpath = os.path.join(label_dir, fname)
            if os.path.isfile(fpath):
                entries.append({
                    "text": "Unlabeled image",
                    "image_path": fpath,
                    "label": label
                })

    return balanced_sample(entries, sample_per_label)


def process_fakeddit(fakeddit_root: str, sample_per_label: int) -> List[Dict]:
    """
    Fakeddit:
    - Read multimodal_train.tsv
    - Map '2-class' to 0 (Clean) and 1 (Fake)
    - Verify image_path exists
    """
    entries = []
    tsv_path = os.path.join(fakeddit_root, "multimodal_train.tsv")
    if not os.path.isfile(tsv_path):
        return []

    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        idx_text = header.index("title") if "title" in header else None
        idx_label = header.index("2-class") if "2-class" in header else None
        idx_image = header.index("img_path") if "img_path" in header else None

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if idx_label is None or idx_image is None:
                continue

            label_raw = parts[idx_label]
            try:
                label_int = int(label_raw)
            except ValueError:
                continue

            label = 0 if label_int == 0 else 1
            image_rel = parts[idx_image]
            image_path = os.path.join(fakeddit_root, image_rel)
            if not os.path.isfile(image_path):
                continue

            text = parts[idx_text] if idx_text is not None else ""
            entries.append({
                "text": text,
                "image_path": image_path,
                "label": label
            })

    return balanced_sample(entries, sample_per_label)


# def process_newsclippings(newsc_root: str, sample_per_label: int) -> Tuple[List[Dict], List[Dict]]:
#     """
#     NewsCLIPpings:
#     - Read merged_annotations.json
#     - Map 'semantically_consistent' -> label 0
#     - Map 'random_mismatch' or 'adversarial_mismatch' -> label 1
#     - Return dataset entries and genuine-only seed entries
#     """
#     entries = []
#     genuine_seed = []
#     ann_path = os.path.join(newsc_root, "merged_annotations.json")
#     if not os.path.isfile(ann_path):
#         return [], []
#
#     with open(ann_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#
#     for item in data:
#         label_str = item.get("label", "")
#         if label_str == "semantically_consistent":
#             label = 0
#         elif label_str in ("random_mismatch", "adversarial_mismatch"):
#             label = 1
#         else:
#             continue
#
#         image_path = item.get("image_path")
#         if not image_path:
#             continue
#
#         # If paths are relative, resolve relative to dataset root
#         if not os.path.isabs(image_path):
#             image_path = os.path.join(newsc_root, image_path)
#
#         if not os.path.isfile(image_path):
#             continue
#
#         text = item.get("caption", "")
#         entry = {"text": text, "image_path": image_path, "label": label}
#         entries.append(entry)
#
#         if label == 0:
#             genuine_seed.append({"text": text, "image_path": image_path, "label": 0})
#
#     return balanced_sample(entries, sample_per_label), genuine_seed


def harmonize_datasets(
    cifake_root: str,
    fakeddit_root: str,
    newsc_root: str,
    sample_per_dataset: int = 5000
) -> Tuple[List[Dict], List[Dict]]:
    """
    Harmonize datasets into MisinfoDataset format.

    Args:
        cifake_root (str): Path to CIFAKE root
        fakeddit_root (str): Path to Fakeddit root
        newsc_root (str): Path to NewsCLIPpings root
        sample_per_dataset (int): Total samples per dataset (balanced 50/50)

    Returns:
        tuple: (harmonized_entries, newsclip_genuine_entries)
    """
    per_label = sample_per_dataset // 2

    cifake_entries = process_cifake(cifake_root, per_label)
    fakeddit_entries = process_fakeddit(fakeddit_root, per_label)

    # NewsCLIPpings preprocessing is commented out per request
    # newsclip_entries, newsclip_genuine = process_newsclippings(newsc_root, per_label)
    newsclip_entries = []
    newsclip_genuine = []

    harmonized = cifake_entries + fakeddit_entries + newsclip_entries
    random.shuffle(harmonized)

    return harmonized, newsclip_genuine


def save_outputs(harmonized: List[Dict], newsclip_genuine: List[Dict]) -> None:
    """
    Save harmonized data and NewsCLIPpings genuine seed.
    """
    with open("harmonized_train_data.pkl", "wb") as f:
        pickle.dump(harmonized, f)

    with open("newscippings_genuine_seed.json", "w", encoding="utf-8") as f:
        json.dump(newsclip_genuine, f, ensure_ascii=False, indent=2)


def main():
    set_seed(42)

    # Update these paths to your local dataset locations
    cifake_root = r"D:\ACM\data\archive"
    fakeddit_root = "./Fakeddit"
    newsc_root = "./NewsCLIPpings"

    harmonized, newsclip_genuine = harmonize_datasets(
        cifake_root=cifake_root,
        fakeddit_root=fakeddit_root,
        newsc_root=newsc_root,
        sample_per_dataset=5000
    )

    save_outputs(harmonized, newsclip_genuine)

    print(f"Total harmonized samples: {len(harmonized)}")
    print("Saved harmonized_train_data.pkl")
    print("Saved newscippings_genuine_seed.json")
    print("Note: NewsCLIPpings preprocessing is currently disabled in this script.")


if __name__ == "__main__":
    main()
