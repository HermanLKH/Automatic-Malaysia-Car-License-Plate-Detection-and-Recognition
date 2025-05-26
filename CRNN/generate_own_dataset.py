#!/usr/bin/env python
"""
generate_own_dataset.py
───────────────────────
Produce a synthetic Malaysian JPJ‑style license‑plate dataset and store it
directly in LMDB format for CRNN‑style OCR training.

Outputs
-------
generated_lmdb_data/train   ← 80 % of samples, with on‑the‑fly augmentations
generated_lmdb_data/val     ← 20 % of samples, clean (no augmentation)

Use `--count` to set the total number of images, and `--two_line_ratio`
to decide what share use the stacked two‑line layout.
"""

import random, os, io, argparse, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
import lmdb

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETERS THAT DEFINE PLATE APPEARANCE
# ──────────────────────────────────────────────────────────────────────────────
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS  = "0123456789"
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",    # Linux
    "/Library/Fonts/Arial Bold.ttf",                           # macOS pre-Catalina
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",       # macOS Catalina+
    "C:\\Windows\\Fonts\\arialbd.ttf",                         # Windows
    "C:\\Windows\\Fonts\\Arial Bold.ttf",                      # alternate Win
]

FONT_FALLBACK = None
for p in FONT_CANDIDATES:
    if os.path.exists(p):
        FONT_FALLBACK = p
        break
if FONT_FALLBACK is None:
    print("⚠️  Warning: no TTF font found, using PIL default bitmap font.")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  TEXT GENERATOR
# ──────────────────────────────────────────────────────────────────────────────
def generate_plate_text() -> str:
    """Return a random JPJ‑style plate string, eg 'ABC1234' or 'WD512A'."""
    prefix = "".join(random.choices(LETTERS, k=random.randint(1, 3)))
    number = "".join(random.choices(DIGITS,  k=random.randint(1, 4)))
    suffix = random.choice(LETTERS) if random.random() < 0.5 else ""
    return prefix + number + suffix

# ──────────────────────────────────────────────────────────────────────────────
# 2.  RENDERER  (single‑line OR two‑line)
# ──────────────────────────────────────────────────────────────────────────────
def render_plate_text(text: str,
                      size: tuple[int,int]=(240,80),
                      two_line: bool=False,
                      font_path: str|None=None) -> Image.Image:
    """Return a PIL image with `text` rendered on a black plate."""
    W,H = size
    font_path = font_path or FONT_FALLBACK
    img  = Image.new("RGB", (W,H), "black")
    draw = ImageDraw.Draw(img)

    # helper to find max font size that fits
    def fit_font(txt, target_h, target_w):
        if FONT_FALLBACK is None:
            return ImageFont.load_default()
        fs   = target_h
        font = ImageFont.truetype(FONT_FALLBACK, fs)
        while fs>1 and (draw.textbbox((0,0),txt,font=font)[2] > target_w or
                        draw.textbbox((0,0),txt,font=font)[3] > target_h):
            fs -= 1
            font = ImageFont.truetype(font_path, fs)
        return font

    if two_line:
        # split letters(top) vs digits+suffix(bottom)
        letters = "".join([c for c in text if c.isalpha() and text.index(c)<len(text.rstrip(DIGITS))])
        rest    = text[len(letters):]
        top_font  = fit_font(letters,  int(H*0.45), int(W*0.9))
        bot_font  = fit_font(rest,     int(H*0.45), int(W*0.9))
        gap       = int(0.1*H)
        # vertical centering
        top_bbox  = draw.textbbox((0,0), letters, font=top_font)
        bot_bbox  = draw.textbbox((0,0), rest,    font=bot_font)
        total_h   = (top_bbox[3]-top_bbox[1]) + gap + (bot_bbox[3]-bot_bbox[1])
        start_y   = (H-total_h)//2
        # draw two rows
        top_x = (W - (top_bbox[2]-top_bbox[0]))//2
        bot_x = (W - (bot_bbox[2]-bot_bbox[0]))//2
        draw.text((top_x, start_y),                letters, font=top_font, fill="white")
        draw.text((bot_x, start_y+(top_bbox[3]-top_bbox[1])+gap),
                  rest, font=bot_font, fill="white")
    else:
        # ── single‐line: extra gap at letter↔digit transitions ──
        font = fit_font(text, int(H * 0.8), int(W * 0.9))

        # find indices where we want a bigger gap
        transitions = {
            i for i in range(len(text) - 1)
            if (text[i].isalpha() and text[i+1].isdigit())
            or (text[i].isdigit() and text[i+1].isalpha())
        }

        # measure each glyph
        widths = []
        for ch in text:
            bb = draw.textbbox((0, 0), ch, font=font)
            widths.append(bb[2] - bb[0])

        # measure a reference "0" for gap sizing
        bb0 = draw.textbbox((0, 0), "0", font=font)
        ref_w = bb0[2] - bb0[0]
        normal_gap = int(ref_w * 0.1)   # 10% of "0"
        big_gap    = int(ref_w * 0.3)   # 30% of "0"

        # total width including gaps
        total_w = sum(widths) + sum(
            (big_gap if i in transitions else normal_gap)
            for i in range(len(text)-1)
        )

        # vertical centering via full text bbox
        tb = draw.textbbox((0, 0), text, font=font)
        text_h = tb[3] - tb[1]
        x = (W - total_w) // 2
        y = (H - text_h) // 2

        # draw each char with its gap
        for i, ch in enumerate(text):
            draw.text((x, y), ch, font=font, fill="white")
            x += widths[i]
            if i < len(text)-1:
                x += big_gap if i in transitions else normal_gap

    return img

# ──────────────────────────────────────────────────────────────────────────────
# 3.  AUGMENTATION (lightweight)
# ──────────────────────────────────────────────────────────────────────────────
def augment_image(img: Image.Image) -> Image.Image:
    if random.random()<0.5:
        img = img.rotate(random.uniform(-5,5), fillcolor="black")
    if random.random()<0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7,1.3))
    if random.random()<0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.5)))
    if random.random()<0.3:
        arr = np.array(img)
        h,w,_ = arr.shape
        n_pix = int(0.01*h*w)
        ys = np.random.randint(0,h,size=n_pix)
        xs = np.random.randint(0,w,size=n_pix)
        arr[ys,xs] = np.random.choice([0,255], size=(n_pix,3))
        img = Image.fromarray(arr)
    return img

# ──────────────────────────────────────────────────────────────────────────────
# 4.  LMDB WRITER
# ──────────────────────────────────────────────────────────────────────────────
def write_lmdb(db_path: str,
               count: int,
               two_line_ratio: float,
               augment: bool):
    os.makedirs(db_path, exist_ok=True)
    env = lmdb.open(db_path, map_size=1<<30)  # 1 GB
    with env.begin(write=True) as txn:
        for i in range(1, count+1):
            label = generate_plate_text()
            two   = random.random()<two_line_ratio
            # random plate size
            if two:
                W,H = random.choice([160,180,200,220]), random.choice([80,90,100,110])
            else:
                W,H = random.choice([200,240,260,300]), random.choice([60,70,80,90])
            img = render_plate_text(label, size=(W,H), two_line=two)
            if augment: img = augment_image(img)
            buf = io.BytesIO(); img.save(buf,'PNG'); bytes_im = buf.getvalue()
            txn.put(f"image-{i:09d}".encode(), bytes_im)
            txn.put(f"label-{i:09d}".encode(), label.encode())
        txn.put(b'num-samples', str(count).encode())
    env.sync(); env.close()

# ──────────────────────────────────────────────────────────────────────────────
# 5.  PREVIEW SOME SAMPLES
# ──────────────────────────────────────────────────────────────────────────────
def preview(lmdb_dir:str, k:int=5):
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        total = int(txn.get(b'num-samples'))
        for idx in random.sample(range(1,total+1), min(k,total)):
            img = Image.open(io.BytesIO(txn.get(f"image-{idx:09d}".encode())))
            lbl = txn.get(f"label-{idx:09d}".encode()).decode()
            print("Label:", lbl); img.show()
    env.close()

# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN  (argument parsing and dataset split)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000,
                        help="Total images to generate (train+val).")
    parser.add_argument("--two_line_ratio", type=float, default=0.4,
                        help="Fraction rendered as stacked two‑line plates.")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train split size (remainder is validation).")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--preview", action="store_true",
                        help="After creation, preview few samples.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path("generated_lmdb_data")
    train_dir = out_root / "train"
    val_dir   = out_root / "val"
    # wipe old dirs
    for d in (train_dir, val_dir):
        if d.exists(): shutil.rmtree(d)

    n_train = int(args.count * args.train_ratio)
    n_val   = args.count - n_train
    print(f"Generating {n_train} train  +  {n_val} val  (total {args.count})")

    write_lmdb(str(train_dir), n_train, args.two_line_ratio, augment=True)
    write_lmdb(str(val_dir),   n_val,   args.two_line_ratio, augment=False)

    print("✅ Finished. LMDBs written to", out_root.resolve())
    if args.preview:
        print("\nShowing a few random TRAIN samples …")
        preview(str(train_dir))
        print("\nShowing a few random VAL samples …")
        preview(str(val_dir))

if __name__ == "__main__":
    main()
