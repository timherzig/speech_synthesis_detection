import torch
import torch.nn as nn
import torch.nn.functional as F


class RNBlock1D(nn.Module):
    def __init__(self, config, in_channels=None, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=config.model.resnet.conv_kernel_size,
            padding=config.model.resnet.padding,
            dilation=config.model.resnet.dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            bias=False,
            kernel_size=config.model.resnet.conv_kernel_size,
            padding=config.model.resnet.padding,
            dilation=config.model.resnet.dilation,
        )
        self.conv3 = nn.Conv1d(
            out_channels,
            out_channels,
            bias=False,
            kernel_size=config.model.resnet.conv_kernel_size,
            padding=config.model.resnet.padding,
            dilation=config.model.resnet.dilation,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.nin = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)

        out += self.nin(residual)
        out = F.relu(self.bn3(out))
        return out


class RNBlock2D(nn.Module):
    def __init__(self, config, in_channels=None, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=config.model.resnet.conv_kernel_size,
            padding=config.model.resnet.padding,
            dilation=config.model.resnet.dilation,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            bias=False,
            kernel_size=config.model.resnet.conv_kernel_size,
            padding=config.model.resnet.padding,
            dilation=config.model.resnet.dilation,
        )
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            bias=False,
            kernel_size=config.model.resnet.conv_kernel_size,
            padding=config.model.resnet.padding,
            dilation=config.model.resnet.dilation,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.nin = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)

        out += self.nin(residual)
        out = F.relu(self.bn3(out))
        return out


class resnet_w_dilation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList()
        self.max_pool_ks = []

        for idx, block in enumerate(config.model.resnet.blocks[:-1]):
            if config.model.dim == 1:
                self.layers.append(
                    RNBlock1D(config, block.in_channels, block.out_channels)
                )
                self.max_pool_ks.append(block.max_pool_ks)
            elif config.model.dim == 2:
                self.layers.append(
                    RNBlock2D(config, block.in_channels, block.out_channels)
                )
                self.max_pool_ks.append(block.max_pool_ks)
            else:
                raise NotImplementedError

        self.last_block = None
        self.last_avg_pool_ks = None
        if config.model.dim == 1:
            self.last_block = RNBlock1D(
                config,
                config.model.resnet.blocks[-1].in_channels,
                config.model.resnet.blocks[-1].out_channels,
            )
            self.last_avg_pool_ks = config.model.resnet.blocks[-1].max_pool_ks
        elif config.model.dim == 2:
            self.last_block = RNBlock1D(
                config,
                config.model.resnet.blocks[-1].in_channels,
                config.model.resnet.blocks[-1].out_channels,
            )
            self.last_avg_pool_ks = config.model.resnet.blocks[-1].max_pool_ks
        else:
            raise NotImplementedError

    def forward(self, x):
        x, labels = x
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if self.config.model.dim == 1:
                x = F.max_pool1d(x, kernel_size=self.max_pool_ks[idx])
            elif self.config.model.dim == 2:
                x = F.max_pool2d(x, kernel_size=self.max_pool_ks[idx])

        x = self.last_block(x)
        if self.config.model.dim == 1:
            x = F.max_pool1d(
                x, kernel_size=x.shape[-1]
            )  # kernel_size=self.last_avg_pool_ks)
        elif self.config.model.dim == 2:
            x = F.avg_pool2d(x, kernel_size=self.last_avg_pool_ks)

        return (torch.flatten(x, start_dim=1), labels)
