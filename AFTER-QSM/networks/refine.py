import torch
import torch.nn as nn


class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(DenseLayer, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        return torch.cat([x, self.relu(self.conv(x))], 1)


class SqueezeAndExcitation(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(SqueezeAndExcitation, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool3d(1)

        self.excitation = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Linear(out_channels, in_channels)
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv3d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):

        scale = self.squeeze(x)

        scale = self.excitation(scale.permute([0, 2, 3, 4, 1]))

        return scale.permute([0, 4, 1, 2, 3]) * self.relu(self.conv(x))


class ResBlocks(nn.Module):

    def __init__(self, in_channels, increment, num_layers, use_dense):
        super(ResBlocks, self).__init__()

        block = DenseLayer if use_dense else SqueezeAndExcitation

        self.CNNs = nn.Sequential(
            *[block(in_channels + increment * i, increment) for i in range(num_layers)])\
            if use_dense else nn.Sequential(*[block(in_channels, in_channels) for _ in range(num_layers)])

        self.output = nn.Conv3d(in_channels + increment * num_layers, increment, 1, 1, 0)\
            if use_dense else nn.Conv3d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):

        return x + self.output(self.CNNs(x))


class Refinement(nn.Module):

    def __init__(self, num_channels=128, base=64, num_blocks=4, num_layers=3, use_dense=True):
        
        super(Refinement, self).__init__()

        self.feature_extraction1 = nn.Conv3d(1, base, 3, 1, 1)
        self.feature_extraction2 = nn.Conv3d(base, base, 3, 1, 1)

        self.blocks = nn.ModuleList([ResBlocks(base, base, num_layers, use_dense) for _ in range(num_blocks)])

        self.feature_fusion_global = nn.Sequential(
            nn.Conv3d(base * num_blocks, base, 3, 1, 1),
            nn.Conv3d(base, base, 3, 1, 1)
        )

        self.conv = nn.Conv3d(base, num_channels, 3, 1, 1)

        self.output = nn.Conv3d(num_channels, 1, 1, 1, 0)

    def forward(self, x):

        shortcut = self.feature_extraction1(x)

        feature = self.feature_extraction2(shortcut)

        skips = []

        for block in self.blocks:

            feature = block(feature)

            skips.append(feature)

        x = self.feature_fusion_global(torch.cat(skips, 1)) + shortcut

        x = self.conv(x)

        return self.output(x)
