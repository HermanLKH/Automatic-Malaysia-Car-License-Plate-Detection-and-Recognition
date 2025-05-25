#!/usr/bin/env python3
import os
import glob
from PIL import Image

# ——— CONFIGURATION ———
# Set these to the paths you need:
INPUT_DIR  = "Vehicle-License-Plate-Detection-8/test"   # contains "images/" and "labels/"
OUTPUT_DIR = "v4_data_add_on_only/test"  # will be created if it doesn't exist
# ————————————————————

IMG_DIR   = os.path.join(INPUT_DIR, "images")
LBL_DIR   = os.path.join(INPUT_DIR, "labels")

def find_image_path(base_name):
    """Look for an image with this base name in IMG_DIR, any common extension."""
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(IMG_DIR, base_name + ext)
        if os.path.isfile(p):
            return p, ext
    # fallback: glob
    matches = glob.glob(os.path.join(IMG_DIR, base_name + ".*"))
    return (matches[0], os.path.splitext(matches[0])[1]) if matches else (None, None)

def crop_class_zero():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for lbl_path in glob.glob(os.path.join(LBL_DIR, "*.txt")):
        base = os.path.splitext(os.path.basename(lbl_path))[0]
        img_path, img_ext = find_image_path(base)
        if not img_path:
            print(f"⚠️  No image found for label {lbl_path!r}, skipping.")
            continue

        img = Image.open(img_path)
        W, H = img.size

        count = 0
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x_c, y_c, w_n, h_n = parts
                if cls_id != "0":
                    continue

                # YOLO format: normalized center + width/height
                x_c, y_c, w_n, h_n = map(float, (x_c, y_c, w_n, h_n))
                box_w, box_h = w_n * W, h_n * H
                x1 = int((x_c * W) - box_w / 2)
                y1 = int((y_c * H) - box_h / 2)
                x2 = int(x1 + box_w)
                y2 = int(y1 + box_h)

                # clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                crop = img.crop((x1, y1, x2, y2))
                out_name = f"{base}_{count}{img_ext}"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                crop.save(out_path)
                count += 1

        if count == 0:
            print(f"ℹ️  No class 0 boxes in {base}, nothing saved.")
        else:
            print(f"✅ Saved {count} crop(s) from {base} → {OUTPUT_DIR}")

if __name__ == "__main__":
    crop_class_zero()
