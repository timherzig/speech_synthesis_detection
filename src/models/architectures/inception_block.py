import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConv1D(nn.Module):
    def __init__(self, blocks, in_channels, out_channels):
        super().__init__()

        out_channels = int(out_channels / len(blocks))

        self.conv_blocks = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=block.kernel_size,
                    dilation=block.dilation,
                    padding=block.padding,
                )
                for block in blocks
            ]
        )
        self.bn_blocks = nn.ModuleList(
            [nn.BatchNorm1d(out_channels) for _ in range(len(blocks))]
        )

    def forward(self, x):
        out = torch.cat(
            [
                F.relu(self.bn_blocks[i](self.conv_blocks[i](x)))
                for i in range(len(self.conv_blocks))
            ],
            dim=1,
        )
        return out


class inception(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList()
        self.max_pool_ks = []

        for idx, block in enumerate(config.model.inception.blocks[:-1]):
            if config.model.dim == 1:
                self.layers.append(
                    DilatedConv1D(
                        block.blocks,
                        in_channels=block.in_channels,
                        out_channels=block.out_channels,
                    )
                )
                self.max_pool_ks.append(block.max_pool_ks)
            else:
                raise NotImplementedError

        last_block = config.model.inception.blocks[-1]
        if config.model.dim == 1:
            self.layers.append(
                DilatedConv1D(
                    blocks=last_block.blocks,
                    in_channels=last_block.in_channels,
                    out_channels=last_block.out_channels,
                )
            )
            self.max_pool_ks.append(last_block.max_pool_ks * len(last_block.blocks))
        else:
            raise NotImplementedError

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if self.config.model.dim == 1:
                x = F.max_pool1d(x, kernel_size=self.max_pool_ks[idx])
            elif self.config.model.dim == 2:
                x = F.max_pool2d(x, kernel_size=self.max_pool_ks[idx])

        return torch.flatten(x, start_dim=1)
