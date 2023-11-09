import torch

import torch.nn as nn

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
                self.config.model.convnextv2.pretrained,
                do_normalize=False,
                do_rescale=False,
            )
        else:
            self.configuration = ConvNextV2Config(**self.config.model.convnextv2)
            self.model = ConvNextV2Model(self.configuration)

        self.fc1 = nn.Linear(
            in_features=768 * 7 * 7,
            out_features=128,
        )
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=64,
        )
        self.out = nn.Linear(
            in_features=64,
            out_features=2,
        )
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.view(x.size(0), -1)
        x -= x.min(1, keepdim=True)[0]
        x /= x.max(1, keepdim=True)[0]
        x = x.view(bs, c, h, w)
        device = x.device

        x = x.expand(-1, self.model.config.num_channels, -1, -1)
        x = self.image_processor(x, return_tensors="pt").to(device)
        outputs = self.model(**x)
        x = torch.flatten(outputs.last_hidden_state, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        x = self.activation(x)
        return x
