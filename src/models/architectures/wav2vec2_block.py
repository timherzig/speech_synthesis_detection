import json
import torch
import torch.nn as nn

from src.models.architectures.wav2vec2.model import Model


class wav2vec2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wav2vec2 = Model(self.config, None, None)

        if self.config.model.wav2vec2_pretrained is not None:
            self.wav2vec2.load_state_dict(
                torch.load(self.config.model.wav2vec2_pretrained, map_location="cpu")
            )

    def forward(self, x):
        x, labels = x
        x = Model(x)
        return (x, labels)
