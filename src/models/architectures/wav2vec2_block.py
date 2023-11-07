import torch

import torch.nn as nn

from transformers import Wav2Vec2Model, AutoProcessor


class wav2vec2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.model.wav2vec2.pretrained is not False:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model.wav2vec2.pretrained
            )
            self.model = Wav2Vec2Model.from_pretrained(
                self.config.model.wav2vec2.pretrained
            )
        else:
            NotImplementedError

        out_shape = 768 * 319
        self.fc1 = nn.Linear(in_features=out_shape, out_features=int(out_shape / 128))
        self.fc2 = nn.Linear(
            in_features=int(out_shape / 128), out_features=int(out_shape / 256)
        )
        self.out = nn.Linear(in_features=int(out_shape / 256), out_features=2)

    def forward(self, x):
        x = torch.squeeze(x, 1)
        outputs = self.model(input_values=x)
        x = self.fc1(torch.flatten(outputs.last_hidden_state, start_dim=1))
        x = self.fc2(x)
        x = self.out(x)
        return x
