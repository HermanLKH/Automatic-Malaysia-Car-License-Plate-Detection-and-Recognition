# =============================
# File: create_lmdb_dataset.py
# =============================
"""
A simple script to create an LMDB dataset from images + ground-truth file.
Each entry is stored as:
  image-00000001 -> raw image bytes
  label-00000001 -> UTF-8 text label
  num-samples   -> total number of valid samples
"""
import os
import lmdb
import cv2
import numpy as np

# === Configuration ===
INPUT_PATH   = '../datasets-workspace/v4_data_!/test/'           # path where images are stored
GT_FILE      = '../datasets-workspace/v4_data_!/test/gt.txt'     # tab-separated file: "relative/path.jpg\tLABEL"
OUTPUT_PATH  = 'v4_lmdb_data_!/test'         # output LMDB directory
CHECK_VALID  = True              # verify images are decodable
MAP_SIZE     = 1_000_000_000     # 1GB
WRITE_FREQ   = 1000              # flush to disk every X samples


def check_image_valid(image_bytes):
    """
    Returns False if image_bytes cannot be decoded to a non-empty image.
    """
    if image_bytes is None:
        return False
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape
    return (h > 0 and w > 0)


def write_cache(env, cache_dict):
    """
    Write all key/value pairs in cache_dict into the LMDB environment.
    """
    with env.begin(write=True) as txn:
        for key, value in cache_dict.items():
            txn.put(key, value)


def create_lmdb_dataset(input_path, gt_file, output_path, check_valid=True):
    """
    Walk through the GT file, read each image/label pair, validate, and store into LMDB.
    """
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=MAP_SIZE)
    cache = {}
    count = 1

    # Read all lines from GT file
    with open(gt_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    total = len(lines)
    for idx, line in enumerate(lines, start=1):
        try:
            rel_path, label = line.split('\t')
        except ValueError:
            print(f"Skipping invalid GT line {idx}: {line}")
            continue
        img_path = os.path.join(input_path, rel_path)

        if not os.path.exists(img_path):
            print(f"[Warning] Image not found: {img_path}")
            continue

        with open(img_path, 'rb') as img_file:
            img_bytes = img_file.read()

        if check_valid and not check_image_valid(img_bytes):
            print(f"[Warning] Invalid image: {img_path}")
            continue

        # LMDB keys: image-%09d, label-%09d
        img_key   = f'image-{count:09d}'.encode('utf-8')
        label_key = f'label-{count:09d}'.encode('utf-8')
        cache[img_key]   = img_bytes
        cache[label_key] = label.encode('utf-8')

        # periodically write cache
        if count % WRITE_FREQ == 0:
            write_cache(env, cache)
            cache.clear()
            print(f"Written {count}/{total} samples...")
        count += 1

    # final write
    num_samples = count - 1
    cache[b'num-samples'] = str(num_samples).encode('utf-8')
    write_cache(env, cache)
    print(f"==> LMDB dataset created with {num_samples} samples at '{output_path}'")


if __name__ == '__main__':
    create_lmdb_dataset(INPUT_PATH, GT_FILE, OUTPUT_PATH, check_valid=CHECK_VALID)