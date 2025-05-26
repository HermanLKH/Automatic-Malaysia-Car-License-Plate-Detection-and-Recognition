# model.py
import torch.nn as nn
from config import MAX_LABEL_LENGTH
from transformation import TPS_SpatialTransformerNetwork
from feature_extraction import ResNetFeatureExtractor
from sequence_modeling  import BidirectionalLSTM
from prediction         import Attention

class CRNN(nn.Module):
    def __init__(self,
                 imgH, imgW, input_channel,
                 output_channel, hidden_size,
                 num_classes,
                 use_attention=False,
                 num_fiducial=20):
        super().__init__()
        # 1) STN
        self.Transformation    = TPS_SpatialTransformerNetwork(
            F=num_fiducial,
            I_size=(imgH, imgW),
            I_r_size=(imgH, imgW),
            I_channel_num=input_channel
        )

        # 2) ResNet feature extractor
        self.FeatureExtraction      = ResNetFeatureExtractor()
        self.FeatureExtraction_out  = output_channel

        # 3) pool height→1
        self.AdaptiveAvgPool        = nn.AdaptiveAvgPool2d((None, 1))
        # self.AdaptiveAvgPool        = nn.AdaptiveAvgPool2d((2, None)) # (height stays “None”, width→2)


        # 4) two‐layer BiLSTM
        self.SequenceModeling       = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size,    hidden_size, hidden_size),
        )
        self.SequenceModeling_out   = hidden_size

        # 5) prediction head: CTC or Attention
        self.use_attention = use_attention
        if not use_attention:
            # CTC
            self.Prediction = nn.Linear(self.SequenceModeling_out, num_classes)
        else:
            # Attention
            self.Prediction = Attention(
                input_size       = self.SequenceModeling_out,
                hidden_size      = hidden_size,
                num_classes      = num_classes
            )

    def forward(self, input, text=None, is_train=True, batch_max_length=MAX_LABEL_LENGTH):
        # 1) rectify
        x = self.Transformation(input)

        # 2) cnn features
        visual_feature = self.FeatureExtraction(x)  # [B,C,H,W]

        # 3) pool → [B,W,C]
        vf = self.AdaptiveAvgPool(visual_feature.permute(0,3,1,2))
        vf = vf.squeeze(3)

        # 3) Adaptive average-pool to height 2
        # pooled = self.AdaptiveAvgPool(visual_feature)    # [B, C, 2, W′]

        # #  → make it a sequence: [B, W′, 2, C] then [B, 2*W′, C]
        # pooled = pooled.permute(0, 3, 2, 1)              # (B, W′, 2, C)
        # B, Wp, R, C = pooled.size()                      # R should be 2
        # vf = pooled.contiguous().view(B, Wp * R, C)      # (B, 2·W′, C)

        # 4) BiLSTM → [B,W,hidden]
        contextual = self.SequenceModeling(vf)

        # 5) head
        if not self.use_attention:
            # CTC: returns [B, T, num_classes]
            return self.Prediction(contextual.contiguous())
        else:
            # Attention: returns [B, S, num_classes]
            return self.Prediction(
                batch_H           = contextual.contiguous(),
                text              = text,
                is_train          = is_train,
                batch_max_length  = batch_max_length
            )
