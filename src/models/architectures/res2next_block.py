import timm
import torch

import torch.nn as nn
import torch.nn.functional as F


class res2next(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = timm.create_model(
            self.config.model.res2next.name,
            pretrained=self.config.model.res2next.pretrained,
            num_classes=self.config.model.res2next.num_classes,
            in_chans=self.config.model.res2next.in_channels,
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.repeat(1, 1, 3, 1)
        return self.model(x)
