import json
import torch
import torch.nn as nn

from src.models.architectures.aasist.models.AASIST import Model


class aasist(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        with open(config.model.aasist_config, "r") as f_json:
            self.aasist_config = json.loads(f_json.read())

        self.aasist = Model(self.aasist_config["model_config"])

        if config.model.aasist_weights is not None:
            self.aasist.load_state_dict(
                torch.load(config.model.aasist_weights, map_location="cpu")
            )

    def forward(self, x):
        x, labels = x
        _, x = self.aasist(x)
        return (x, labels)
