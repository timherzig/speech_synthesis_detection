import torch.nn as nn
import torch.nn.functional as F

# from src.models.architectures.res2next_block import res2next
# from src.models.architectures.convnext_block import convnext
# from src.models.architectures.convnextv2_block import convnextv2
from src.models.architectures.resnet_block import resnet
from src.models.architectures.inception_block import inception
from src.models.architectures.wav2vec2_block import wav2vec2


class first_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.model.dim

        if self.dim == 1:
            self.shared_block_1_conv = nn.Conv1d(
                in_channels=config.model.conv1.in_channels,
                out_channels=config.model.conv1.out_channels,
                kernel_size=config.model.conv1.kernel_size,
                padding=config.model.conv1.padding,
                bias=config.model.conv1.bias,
            )
            self.shared_block_1_bn = nn.BatchNorm1d(config.model.bn1.num_features)
        elif self.dim == 2:
            self.shared_block_1_conv = nn.Conv2d(
                in_channels=config.model.conv1.in_channels,
                out_channels=config.model.conv1.out_channels,
                kernel_size=config.model.conv1.kernel_size,
                padding=config.model.conv1.padding,
                bias=config.model.conv1.bias,
            )
            self.shared_block_1_bn = nn.BatchNorm2d(config.model.bn1.num_features)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.shared_block_1_conv(x)
        x = self.shared_block_1_bn(x)
        return x


class last_layers(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(
            in_features=config.model.fc1.in_features,
            out_features=config.model.fc1.out_features,
        )
        self.fc2 = nn.Linear(
            in_features=config.model.fc2.in_features,
            out_features=config.model.fc2.out_features,
        )
        self.out = nn.Linear(
            in_features=config.model.out.in_features,
            out_features=config.model.out.out_features,
        )

        if config.model.activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif config.model.activation == "logsoftmax":
            self.activation = nn.LogSoftmax(dim=1)
        else:  # default
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return self.activation(x)


class build_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()

        if self.config.model.shared_first_last:
            self.layers.append(first_layer(self.config))

        if self.config.model.architecture == "resnet":
            self.layers.append(resnet(self.config))
        elif self.config.model.architecture == "inception":
            self.layers.append(inception(self.config))
        # elif self.config.model.architecture == "res2next":
        #     self.layers.append(res2next(self.config))
        # elif self.config.model.architecture == "convnextv2":
        #     self.layers.append(convnextv2(self.config))
        # elif self.config.model.architecture == "convnext":
        #     self.layers.append(convnext(self.config))
        elif self.config.model.architecture == "wav2vec2":
            self.layers.append(wav2vec2(self.config))
        else:
            raise NotImplementedError

        if self.config.model.shared_first_last:
            self.layers.append(last_layers(self.config))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
