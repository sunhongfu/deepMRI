import torch
import torch.nn as nn


class Unet(nn.Module):

    class Encoder(nn.Module):

        def __init__(self, in_channel, out_channel):

            super(Unet.Encoder, self).__init__()

            self._input = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

            self._output = nn.Sequential(
                nn.Conv3d(out_channel, out_channel, 3, 1, 1),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

        def forward(self, feature_map):

            mid = self._input(feature_map)
            res = self._output(mid)
            return res

    class Decoder(nn.Module):

        def __init__(self, in_channel, out_channel):

            super(Unet.Decoder, self).__init__()

            self._input = nn.Sequential(
                nn.ConvTranspose3d(in_channel, out_channel, 2, stride=2),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

            self._mid = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

            self._output = nn.Sequential(
                nn.Conv3d(out_channel, out_channel, 3, 1, 1),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

        def forward(self, feature_map, skip):

            x = self._input(feature_map)

            mid = self._mid(torch.cat([x, skip], dim=1))

            res = self._output(mid)

            return res

    def __init__(self, depth, base, init=1):

        super(Unet, self).__init__()

        self._input = Unet.Encoder(init, base)

        self._encoders = nn.ModuleList([nn.Sequential(nn.MaxPool3d(2),
                                        Unet.Encoder(base * 2 ** i, base * 2 ** (i + 1)))
                                        for i in range(depth)])

        self._decoders = nn.ModuleList([Unet.Decoder(base * 2 ** i, base * 2 ** (i - 1))
                                        for i in range(depth, 0, -1)])

        self._output = nn.Conv3d(base, init, 1, 1, 0)

    def forward(self, x):

        skips = []

        in_encoder = self._input(x)
        skips.append(in_encoder)

        for encoder in self._encoders:

            in_encoder = encoder(in_encoder)
            skips.append(in_encoder)

        in_decoder = in_encoder

        skips.pop()

        for decoder in self._decoders:
            in_decoder = decoder(in_decoder, skips.pop())

        return self._output(in_decoder)
