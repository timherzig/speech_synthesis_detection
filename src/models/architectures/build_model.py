import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures.resnet_block import resnet
from src.models.architectures.inception_block import inception
from src.models.architectures.aasist_block import aasist

from src.utils.loss import AMSoftmax, OCSoftmax


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
        x, labels = x
        x = self.shared_block_1_conv(x)
        x = self.shared_block_1_bn(x)
        return (x, labels)


class last_layers(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_type = config.model.activation

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
        elif config.model.activation == "am_softmax":
            self.activation = AMSoftmax(
                num_classes=config.model.out.out_features,
                enc_dim=config.model.out.out_features,
            )
        elif config.model.activation == "oc_softmax":
            self.activation = OCSoftmax(
                feat_dim=config.model.out.out_features,
            )
        else:  # default
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x, labels = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        if self.activation_type == "am_softmax" or self.activation_type == "oc_softmax":
            return (self.activation(x, labels), labels)
        else:
            return (self.activation(x), labels)


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
        elif self.config.model.architecture == "aasist":
            self.layers.append(aasist(self.config))
        else:
            raise NotImplementedError

        if self.config.model.shared_first_last:
            self.layers.append(last_layers(self.config))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x, labels = x
        return self.model((x, labels))
