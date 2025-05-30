# ─── CONFIG ────────────────────────────────────────────────────────────
TRAIN_IMG_DIR  = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL_DIR  = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMG_DIR    = os.path.join(DATASET_DIR, "val",   "images")
VAL_LBL_DIR    = os.path.join(DATASET_DIR, "val",   "labels")
TEST_IMG_DIR   = os.path.join(DATASET_DIR, "test",  "images")
TEST_LBL_DIR   = os.path.join(DATASET_DIR, "test",  "labels")

# fractions for the split
VAL_FRAC  = 0.2
TEST_FRAC = 0.1

random.seed(RANDOM_SEED)

# 1. make sure output dirs exist
for d in (VAL_IMG_DIR, VAL_LBL_DIR, TEST_IMG_DIR, TEST_LBL_DIR):
    os.makedirs(d, exist_ok=True)

# 2. list all images in the original train folder
all_imgs = glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")) \
         + glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png"))

random.shuffle(all_imgs)
n = len(all_imgs)
n_val  = int(VAL_FRAC  * n)
n_test = int(TEST_FRAC * n)

val_imgs  = all_imgs[:n_val]
test_imgs = all_imgs[n_val:n_val + n_test]

# 3. move val
for img_path in val_imgs:
    fname = os.path.basename(img_path)
    stem  = os.path.splitext(fname)[0]
    lbl   = stem + ".txt"

    shutil.move(img_path, os.path.join(VAL_IMG_DIR, fname))
    lbl_src = os.path.join(TRAIN_LBL_DIR, lbl)
    if os.path.exists(lbl_src):
        shutil.move(lbl_src, os.path.join(VAL_LBL_DIR, lbl))

# 4. report
remain = len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*")))
print("Split complete:")
print(f"  train: {remain} images")
print(f"  val:   {len(val_imgs)} images")
print(f"  test:  {len(test_imgs)} images")