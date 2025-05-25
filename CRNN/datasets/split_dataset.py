import os
import random
import shutil

# ——— GLOBAL CONFIGURATION ———
SOURCE_DIR = "v4_data_!/trainval"   # contains your combined train+val images\ nTRAIN_DIR  = "v4_data_!/train"       # destination for training split
TRAIN_DIR  = "v4_data_!/train"       # destination for training split
VAL_DIR    = "v4_data_!/val"         # destination for validation split
SEED       = 300188                   # fixed seed for reproducibility
VALID_RATIO = 0.2                     # 8:2 split ratio (20% val)
# ————————————————————————

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def is_image_file(fname: str) -> bool:
    """Return True if file has a supported image extension."""
    return fname.lower().endswith(IMAGE_EXTENSIONS)


def main():
    random.seed(SEED)

    # Create output directories if they don't exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    # List all image files in SOURCE_DIR
    all_files = [f for f in os.listdir(SOURCE_DIR) if is_image_file(f)]
    total = len(all_files)
    val_count = int(total * VALID_RATIO)

    # Filter out files with parentheses for validation candidates
    candidates = [f for f in all_files if '(' not in f and ')' not in f]
    if val_count > len(candidates):
        print(f"⚠️  Requested {val_count} val images but only {len(candidates)} without parentheses; reducing to {len(candidates)}.")
        val_count = len(candidates)

    # Randomly sample validation set
    val_set = set(random.sample(candidates, val_count))
    print(f"Total images: {total}. Using {len(val_set)} for validation, {total - len(val_set)} for training.")

    # Copy files to train/val
    for fname in all_files:
        src = os.path.join(SOURCE_DIR, fname)
        dst_dir = VAL_DIR if fname in val_set else TRAIN_DIR
        dst = os.path.join(dst_dir, fname)
        shutil.copy2(src, dst)

    print("✅ Dataset split complete.")


if __name__ == "__main__":
    main()