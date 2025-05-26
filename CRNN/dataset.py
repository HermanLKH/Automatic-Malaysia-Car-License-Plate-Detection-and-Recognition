# =============================
# File: dataset.py
# =============================
"""
Dataset reader for LMDB text-recognition.
Provides LmdbDataset and AlignCollate for PyTorch.
"""
import os
import io
import re
import lmdb
import sys
import math
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import CHARACTERS, MAX_LABEL_LENGTH, IMG_HEIGHT, IMG_WIDTH

# === Configuration ===
PADDING          = False    # if True, keep aspect ratio with padding


class LmdbDataset(Dataset):
    """
    Reads (image, label) pairs from an LMDB database.
    Automatically filters out entries with labels longer than MAX_LABEL_LENGTH
    or containing characters outside CHARACTERS.
    """
    def __init__(self, lmdb_path):
        self.env = lmdb.open(
            lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        if not self.env:
            print(f"Cannot open LMDB at {lmdb_path}")
            sys.exit(1)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get(b'num-samples'))
            # build filtered index list
            self.indices = []
            for i in range(1, self.nSamples + 1):
                label = txn.get(f'label-{i:09d}'.encode()).decode('utf-8')
                if len(label) > MAX_LABEL_LENGTH:
                    continue
                if re.search(f'[^{CHARACTERS}]', label):
                    continue
                self.indices.append(i)
        print(f"Loaded {len(self.indices)} valid samples from {lmdb_path}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        with self.env.begin(write=False) as txn:
            label = txn.get(f'label-{real_idx:09d}'.encode()).decode('utf-8')
            img_bytes = txn.get(f'image-{real_idx:09d}'.encode())

        # load image
        buf = io.BytesIO(img_bytes)
        img = Image.open(buf).convert('L')  # grayscale

        return img, label


class AlignCollate:
    """
    Resize + normalize (and optionally pad) for a batch of PIL images.
    Returns a tensor batch of shape [B, C, H, W] and list of labels.
    """
    def __init__(self, imgH=IMG_HEIGHT, imgW=IMG_WIDTH, keep_ratio_with_pad=PADDING):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.toTensor = transforms.ToTensor()

    def __call__(self, batch):
        images, labels = zip(*batch)
        processed = []

        if self.keep_ratio_with_pad:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            # pad each image to (C, imgH, imgW)
            for img in images:
                w, h = img.size
                ratio = w / float(h)
                new_w = min(self.imgW, math.ceil(self.imgH * ratio))
                resized = img.resize((new_w, self.imgH), Image.BICUBIC)
                tensor = transform(resized)
                # pad to full width
                pad = torch.zeros(1, self.imgH, self.imgW - new_w)
                tensor = torch.cat([tensor, pad], dim=2)
                processed.append(tensor)
        else:
            # simple resize
            transform = transforms.Compose([
                transforms.Resize((self.imgH, self.imgW), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            processed = [transform(img) for img in images]

        import torch
        image_batch = torch.stack(processed, 0)
        return image_batch, labels
