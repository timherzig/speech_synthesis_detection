import torch

import torch.nn as nn
import torch.nn.functional as F

from transformers import ConvNextV2Config, ConvNextV2Model, AutoImageProcessor


class convnextv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if isinstance(self.config.model.convnextv2.pretrained, str):
            self.model = ConvNextV2Model.from_pretrained(
                self.config.model.convnextv2.pretrained
            )
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.config.model.convnextv2.pretrained, do_normalize=False
            )
        else:
            self.configuration = ConvNextV2Config(**self.config.model.convnextv2)
            self.model = ConvNextV2Model(self.configuration)

    def forward(self, x):
        device = x.device
        x = x.expand(-1, self.model.config.num_channels, -1, -1)
        x = self.image_processor(x, return_tensors="pt").to(device)
        outputs = self.model(**x)
        return outputs.last_hidden_state
