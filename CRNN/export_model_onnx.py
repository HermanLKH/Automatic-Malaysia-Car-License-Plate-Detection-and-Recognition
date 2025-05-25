#!/usr/bin/env python3
import torch
from torch import nn
from config import (
    IMG_HEIGHT, IMG_WIDTH,
    INPUT_CHANNEL, OUTPUT_CHANNEL,
    HIDDEN_SIZE, NUM_FIDUCIAL,
    CHARACTERS, MAX_LABEL_LENGTH
)
from model import CRNN
from utils import AttnLabelConverter

# ────────────────────────────────────────────────────────────────
#  Paths
# ────────────────────────────────────────────────────────────────
PTH_PATH  = "trained_model/best_attention_crnn_!_augment_8513.pth"
ONNX_PATH = "crnn_attention.onnx"

# ────────────────────────────────────────────────────────────────
#  1) Load your trained CRNN
# ────────────────────────────────────────────────────────────────
device    = torch.device("cpu")
converter = AttnLabelConverter(CHARACTERS)
num_classes = len(converter.character)  # includes [GO] & [s]

crnn = CRNN(
    IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNEL,
    OUTPUT_CHANNEL, HIDDEN_SIZE, num_classes,
    use_attention=True, num_fiducial=NUM_FIDUCIAL
).to(device)
crnn.load_state_dict(torch.load(PTH_PATH, map_location=device))
crnn.eval()

# ────────────────────────────────────────────────────────────────
#  2) Build an inference wrapper (TPS→CNN→mean-pool→BiLSTM→Attention)
# ────────────────────────────────────────────────────────────────
class CRNNInference(nn.Module):
    def __init__(self, backbone: CRNN, go_idx: int, max_len: int):
        super().__init__()
        self.trans   = backbone.Transformation
        self.cnn     = backbone.FeatureExtraction
        self.bilstm  = backbone.SequenceModeling
        self.attn    = backbone.Prediction
        self.go_idx  = go_idx
        self.max_len = max_len

    def forward(self, image: torch.Tensor):
        # 1) rectify image
        x = self.trans(image)
        # 2) extract CNN features
        feat = self.cnn(x)
        # 3) collapse height dimension by mean pooling
        seq  = feat.permute(0,3,2,1).mean(dim=2)
        # 4) BiLSTM encoding
        ctx  = self.bilstm(seq)
        B    = ctx.size(0)
        # 5) Attention decoding starting from [GO]
        init = torch.full((B,), self.go_idx, dtype=torch.long, device=image.device)
        logits = self.attn(
            batch_H           = ctx,
            text              = init,
            is_train          = False,
            batch_max_length  = self.max_len
        )
        return logits

inference_model = CRNNInference(
    crnn,
    go_idx=converter.dict["[GO]"],
    max_len=MAX_LABEL_LENGTH
).to(device)
inference_model.eval()

# ────────────────────────────────────────────────────────────────
#  3) Export to ONNX (with opset >=16 for grid_sampler support)
# ────────────────────────────────────────────────────────────────
dummy = torch.randn(1, INPUT_CHANNEL, IMG_HEIGHT, IMG_WIDTH, device=device)
torch.onnx.export(
    inference_model,
    dummy,
    ONNX_PATH,
    input_names   = ["image"],
    output_names  = ["logits"],
    dynamic_axes  = {
        "image":  {0: "batch"},
        "logits": {0: "batch", 1: "time"},
    },
    opset_version = 16,
    do_constant_folding = True,
)
print(f"✅ Successfully exported full CRNN+Attention to {ONNX_PATH} with opset 16")
