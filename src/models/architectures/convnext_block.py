import timm
import torch

import torch.nn as nn
import torch.nn.functional as F


class convnext(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if isinstance(self.config.model.convnext.pretrained, str):
            self.model = timm.create_model(
                self.config.model.convnext.pretrained,
                pretrained=True,
                num_classes=2,
                global_pool="avg",
            )
        else:
            raise NotImplementedError

        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=True
        )

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        return self.model(x)
