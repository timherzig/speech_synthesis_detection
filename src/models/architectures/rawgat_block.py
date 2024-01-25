import yaml
import torch
import torch.nn as nn

from src.models.architectures.rawgat.model import RawGAT_ST


class rawgat(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config

        with open(config.model.rawgat_config, "r") as f_yaml:
            self.rawgat_config = yaml.safe_load(f_yaml)

        self.rawgat = RawGAT_ST(self.rawgat_config["model"], device)

        if config.model.rawgat_weights is not None:
            self.rawgat.load_state_dict(
                torch.load(config.model.rawgat_weights, map_location="cpu")
            )

    def forward(self, x):
        x, labels = x
        x = self.rawgat(x)
        return (x, labels)
