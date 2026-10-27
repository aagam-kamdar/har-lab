"""
download_and_train_scb.py
--------------------------
Downloads the closest publicly available classroom behavior dataset from
Roboflow Universe and fine-tunes YOLOv8n on it.

TARGET CLASSES (our 7):
  sitting  writing  typing  phone  sleeping  standing  discuss

SCB-Dataset3 original classes:
  hand-raising  reading  writing  phone  bowing-head  leaning-over-table

The Roboflow dataset "student-classroom-activity" has closer labels:
  hand raise  phone  sleep  study  discuss  typing

We remap everything to our 7 final labels in a post-processing step
(see action_classifier.py _SCB_MAP).

USAGE:
  1. Set your Roboflow API key below (free account at roboflow.com)
  2. python download_and_train_scb.py

OUTPUT:
  scb_training/
    data.yaml          ← dataset config
    runs/              ← YOLOv8 training output
  scb_model/
    best.pt            ← trained weights (copy this as ACTION_MODEL_PATH)
"""

from __future__ import annotations
import os
import sys
import shutil
import yaml

# ============================================================
# CONFIG — set your Roboflow API key here
# Get a free key at: https://app.roboflow.com/settings/api
# ============================================================
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

# Best public classroom behavior dataset on Roboflow Universe
# with discuss/discussion, writing, phone, sleeping, typing classes
ROBOFLOW_WORKSPACE = "student-activities-h65io"
ROBOFLOW_PROJECT   = "student-classroom-activity-action"
ROBOFLOW_VERSION   = 1

# Training config
BASE_MODEL   = "yolov8n.pt"      # lightest YOLOv8 — fastest to train
EPOCHS       = 30                 # increase to 50-100 for better accuracy
IMG_SIZE     = 640
BATCH_SIZE   = 16
OUTPUT_DIR   = "scb_training"
WEIGHTS_OUT  = "scb_model"

# Clean, honest 9-action taxonomy:
# No forced relabeling. Classes not in our scope (e.g. hand-raising) are discarded.
OUR_CLASSES = [
    "sitting",   # neutral attentive seated
    "reading",   # reading book/screen/notes (study)
    "writing",   # holding pen/pencil actively writing
    "typing",    # typing on laptop/keyboard
    "phone",     # holding or looking at smartphone
    "sleeping",  # head on desk / slumped asleep
    "standing",  # stationary upright posture
    "walking",   # dynamic locomotion across room
    "discuss",   # talking/discussing with peers
]

CLASS_REMAP = {
    # Genuine 1-to-1 matches:
    "writing":      "writing",
    "typing":       "typing",
    "reading":      "reading",
    "study":        "reading",
    "studying":     "reading",
    "phone":        "phone",
    "phone use":    "phone",
    "using-phone":  "phone",
    "using_phone":  "phone",
    "sleep":        "sleeping",
    "sleeping":     "sleeping",
    "standing":     "standing",
    "walking":      "walking",
    "discuss":      "discuss",
    "discussion":   "discuss",
    "talking":      "discuss",
    "sitting":      "sitting",
    "looking-ahead":"sitting",

    # Irrelevant or ambiguous classes -> DISCARDED (None):
    "hand raise":         None,
    "hand-raising":       None,
    "hand_raising":       None,
    "bowing-head":        None,
    "bowing_head":        None,
    "head-down":          None,
    "leaning-over-table": None,
    "leaning_over_table": None,
}


def check_api_key():
    if not ROBOFLOW_API_KEY:
        print("=" * 60)
        print("ERROR: Roboflow API key not set!")
        print()
        print("1. Create a FREE account at https://app.roboflow.com")
        print("2. Go to Settings → API Keys → Copy your key")
        print("3. Set it one of two ways:")
        print()
        print("   Option A — Environment variable:")
        print("   $env:ROBOFLOW_API_KEY = 'your_key_here'")
        print("   python download_and_train_scb.py")
        print()
        print("   Option B — Edit this file:")
        print("   ROBOFLOW_API_KEY = 'your_key_here'  (line 22)")
        print("=" * 60)
        sys.exit(1)


def download_dataset() -> str:
    """Download classroom behavior dataset from Roboflow Universe."""
    from roboflow import Roboflow

    print(f"\n[1/4] Connecting to Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    print(f"[1/4] Downloading dataset: {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}")
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov8", location=OUTPUT_DIR)

    print(f"[1/4] Dataset downloaded to: {dataset.location}")
    return dataset.location


def try_alternative_datasets(rf):
    """Try alternative public datasets if primary is unavailable."""
    alternatives = [
        # (workspace, project, version)
        ("studentactivity", "new-student-classroom-activity-3-hand-raise-phone-sleep", 2),
        ("classroom-wvpfq", "classroom-behavior", 1),
        ("scb-dataset", "student-behavior-detection", 1),
    ]
    for ws, proj, ver in alternatives:
        try:
            print(f"  Trying: {ws}/{proj} v{ver}...")
            dataset = rf.workspace(ws).project(proj).version(ver).download("yolov8", location=OUTPUT_DIR)
            print(f"  ✓ Downloaded from: {ws}/{proj}")
            return dataset.location
        except Exception as e:
            print(f"  ✗ {e}")
    return None


def remap_yaml(yaml_path: str) -> tuple[str, dict]:
    """
    Remap dataset class names to our 9 standard labels.
    Irrelevant classes (e.g. hand-raising) map to None and are discarded.
    """
    print(f"\n[2/4] Remapping class names in {yaml_path}...")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    orig_names = data.get("names", [])
    print(f"  Original dataset classes ({len(orig_names)}): {orig_names}")

    # The canonical list of classes for our model is OUR_CLASSES
    new_names = list(OUR_CLASSES)
    name_map_idx = {}   # old_idx -> new_idx (or None to discard)

    for i, name in enumerate(orig_names):
        nl = name.lower().strip()
        mapped = CLASS_REMAP.get(nl, None)

        if mapped is not None and mapped in new_names:
            name_map_idx[i] = new_names.index(mapped)
            print(f"    Keep: '{name}' -> '{mapped}' (class {name_map_idx[i]})")
        else:
            name_map_idx[i] = None
            print(f"    Drop: '{name}' -> DISCARDED")

    data["names"] = new_names
    data["nc"] = len(new_names)

    # Save updated yaml
    updated_path = yaml_path.replace(".yaml", "_remapped.yaml")
    with open(updated_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"  Saved remapped yaml: {updated_path}")
    print(f"  Final active classes ({len(new_names)}): {new_names}")
    return updated_path, name_map_idx


def remap_labels(dataset_location: str, name_map_idx: dict):
    """Update label .txt files: translate indices, discard dropped boxes."""
    print("\n[3/4] Updating label files with remapped class indices (discarding dropped classes)...")
    kept_count = 0
    dropped_count = 0
    file_count = 0

    for split in ["train", "valid", "test"]:
        labels_dir = os.path.join(dataset_location, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        for lf in os.listdir(labels_dir):
            if not lf.endswith(".txt"):
                continue
            fpath = os.path.join(labels_dir, lf)
            lines = open(fpath).readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_cls = int(parts[0])
                    new_cls = name_map_idx.get(old_cls, None)
                    if new_cls is not None:
                        new_lines.append(f"{new_cls} " + " ".join(parts[1:]) + "\n")
                        kept_count += 1
                    else:
                        dropped_count += 1
            with open(fpath, "w") as f:
                f.writelines(new_lines)
            file_count += 1

    print(f"  Processed {file_count} label files: kept {kept_count} boxes, cleanly discarded {dropped_count} unwanted boxes.")


def train_model(yaml_path: str):
    """Fine-tune YOLOv8n on the downloaded dataset."""
    from ultralytics import YOLO

    print(f"\n[4/4] Starting YOLOv8n fine-tuning...")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Image size : {IMG_SIZE}")
    print(f"  Batch      : {BATCH_SIZE}")
    print(f"  Data YAML  : {yaml_path}")
    print()

    model = YOLO(BASE_MODEL)
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=OUTPUT_DIR,
        name="scb_run",
        exist_ok=True,
        verbose=True,
        patience=10,
        # Regularization for small dataset
        dropout=0.1,
        weight_decay=0.0005,
        augment=True,
        degrees=10.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
    )

    # Copy best weights to scb_model/
    os.makedirs(WEIGHTS_OUT, exist_ok=True)
    best_pt = os.path.join(OUTPUT_DIR, "scb_run", "weights", "best.pt")
    if os.path.exists(best_pt):
        dst = os.path.join(WEIGHTS_OUT, "scb_action_best.pt")
        shutil.copy2(best_pt, dst)
        print(f"\n{'='*60}")
        print(f"✅ Training complete!")
        print(f"   Best weights → {dst}")
        print(f"\n   To use in dashboard, set:")
        print(f"   $env:ACTION_MODEL_PATH = '{os.path.abspath(dst)}'")
        print(f"   python dashboard.py")
        print(f"{'='*60}")
        return dst
    else:
        print("Warning: best.pt not found. Check training output.")
        return None


def check_existing_dataset() -> bool:
    """Check if dataset already exists locally in scb_training/."""
    if os.path.isdir(OUTPUT_DIR):
        yaml_candidates = [
            os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
            if f.endswith(".yaml") and not f.endswith("_remapped.yaml")
        ]
        if yaml_candidates:
            print(f"✓ Found existing dataset in '{OUTPUT_DIR}' with YAML: {yaml_candidates[0]}")
            return True
    return False


def main():
    # If user already downloaded dataset manually to scb_training/
    if check_existing_dataset():
        print("\n[1/4] Using existing local dataset in scb_training/ folder...")
        dataset_loc = OUTPUT_DIR
    else:
        check_api_key()
        # Step 1: Download
        try:
            dataset_loc = download_dataset()
        except Exception as e:
            print(f"\nPrimary dataset failed: {e}")
            print("Trying alternative datasets...")
            from roboflow import Roboflow
            rf = Roboflow(api_key=ROBOFLOW_API_KEY)
            dataset_loc = try_alternative_datasets(rf)
            if dataset_loc is None:
                print("\n" + "=" * 65)
                print("❌ WHY THIS HAPPENED:")
                print("Roboflow requires public Universe datasets to be 'Forked' into")
                print("your own workspace first before your API key can access them.")
                print()
                print("EASIEST 1-MINUTE FIX (Option A - Fork & Download via Code):")
                print("1. Open in browser: https://universe.roboflow.com/project-yzut0/student-classroom-behaviour")
                print("2. Click 'Fork Dataset' (top right) -> copy into your workspace")
                print("3. Click 'Download Dataset' -> select 'YOLOv8' -> 'show download code'")
                print("4. Copy the workspace name & project name from the snippet into lines 45-46 of this script:")
                print("      ROBOFLOW_WORKSPACE = 'your-workspace-name'")
                print("      ROBOFLOW_PROJECT   = 'your-project-name'")
                print()
                print("EVEN EASIER FIX (Option B - Manual Zip Download):")
                print("1. On the Universe page, click 'Download Dataset' -> select 'YOLOv8' -> 'download zip to computer'")
                print(f"2. Extract the zip contents directly into: {os.path.abspath(OUTPUT_DIR)}")
                print("3. Re-run: python download_and_train_scb.py (it will auto-detect and train!)")
                print("=" * 65)
                sys.exit(1)

    # Find yaml file
    yaml_files = [f for f in os.listdir(dataset_loc) if f.endswith(".yaml") and "data" in f.lower()]
    if not yaml_files:
        yaml_files = [f for f in os.listdir(dataset_loc) if f.endswith(".yaml")]
    if not yaml_files:
        print(f"❌ No YAML found in {dataset_loc}")
        sys.exit(1)

    yaml_path = os.path.join(dataset_loc, yaml_files[0])

    # Step 2: Remap class names
    updated_yaml, name_map_idx = remap_yaml(yaml_path)

    # Step 3: Remap label files
    remap_labels(dataset_loc, name_map_idx)

    # Step 4: Train
    weights_path = train_model(updated_yaml)
    return weights_path


if __name__ == "__main__":
    main()
