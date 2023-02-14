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
        self.depth = depth
        self._input = Unet.Encoder(init, base)
        self._encoders = nn.ModuleList([nn.Sequential(nn.MaxPool3d(2),
                                        Unet.Encoder(base * 2 ** i, base * 2 ** (i + 1)))
                                        for i in range(depth)])
        self._decoders = nn.ModuleList([Unet.Decoder(base * 2 ** i, base * 2 ** (i - 1))
                                        for i in range(depth, 0, -1)])
        self._output = nn.Conv3d(base, init, 1, 1, 0)

    def forward(self, x):
        skips = []
        inEncoder = self._input(x)
        skips.append(inEncoder)

        for encoder in self._encoders:
            inEncoder = encoder(inEncoder)
            skips.append(inEncoder)

        inDecoder = inEncoder
        skips.pop()
        skips.reverse()

        for decoder, skip in zip(self._decoders, skips):
            inDecoder = decoder(inDecoder, skip)

        return self._output(inDecoder)

    # def forward(self, x):
    #     print(-1)
    #     skips = []
    #     inEncoder = self._input(x)
    #     skips.append(inEncoder)
    #     print(0)
    #
    #     for index, encoder in enumerate(self._encoders):
    #         inEncoder = encoder(inEncoder)
    #         skips.append(inEncoder)
    #         print(index)
    #
    #     inDecoder = inEncoder
    #     skips.pop()
    #
    #     inDecoder = self.decoder_0(inDecoder, skips.pop())
    #     inDecoder = self.decoder_1(inDecoder, skips.pop())
    #     inDecoder = self.decoder_2(inDecoder, skips.pop())
    #
    #     skips.reverse()
    #     for index, (decoder, skip) in enumerate(zip(self._decoders, skips)):
    #         inDecoder = decoder(inDecoder, skip)
    #         print(-index)
    #
    #     return self._output(inDecoder)


class UnetNoTranspose(nn.Module):

    class Encoder(nn.Module):

        def __init__(self, in_channel, out_channel):
            super(UnetNoTranspose.Encoder, self).__init__()
            self._input = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
            self._output = nn.Sequential(
                nn.Conv3d(out_channel, out_channel, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

        def forward(self, feature_map):
            mid = self._input(feature_map)
            res = self._output(mid)
            return res

    class Decoder(nn.Module):

        def __init__(self, in_channel, out_channel):
            super(UnetNoTranspose.Decoder, self).__init__()
            self._input = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(in_channel, out_channel, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
            self._mid = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
            self._output = nn.Sequential(
                nn.Conv3d(out_channel, out_channel, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

        def forward(self, feature_map, skip):
            x = self._input(feature_map)
            mid = self._mid(torch.cat([x, skip], dim=1))
            res = self._output(mid)
            return res

    def __init__(self, depth, base):
        super(UnetNoTranspose, self).__init__()
        self.depth = depth
        self._input = UnetNoTranspose.Encoder(1, base)
        self._encoders = nn.ModuleList([nn.Sequential(nn.MaxPool3d(2),
                                                      UnetNoTranspose.Encoder(base * 2 ** i, base * 2 ** (i + 1)))
                                        for i in range(depth)])
        self._decoders = nn.ModuleList([UnetNoTranspose.Decoder(base * 2 ** i, base * 2 ** (i - 1))
                                        for i in range(depth, 0, -1)])
        self._output = nn.Conv3d(base, 1, 1, 1, 0)

    def forward(self, x):
        skips = []
        inEncoder = self._input(x)
        skips.append(inEncoder)

        for encoder in self._encoders:
            inEncoder = encoder(inEncoder)
            skips.append(inEncoder)

        inDecoder = inEncoder
        skips.pop()
        skips.reverse()

        for decoder, skip in zip(self._decoders, skips):
            inDecoder = decoder(inDecoder, skip)

        return self._output(inDecoder)


class DeepQSM(nn.Module):

    class Encoder(nn.Module):

        def __init__(self, in_channel, out_channel):
            super(DeepQSM.Encoder, self).__init__()
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
            skip = self._input(feature_map)
            res = self._output(skip)
            return skip, res

    class Decoder(nn.Module):

        def __init__(self, in_channel, out_channel):
            super(DeepQSM.Decoder, self).__init__()
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

    def __init__(self, depth, base):
        super(DeepQSM, self).__init__()
        self.depth = depth
        self._input = DeepQSM.Encoder(1, base)
        self._encoders = nn.ModuleList([nn.Sequential(nn.MaxPool3d(2),
                                                      DeepQSM.Encoder(base * 2 ** i, base * 2 ** (i + 1))
                                                      )
                                        for i in range(depth)])
        self._decoders = nn.ModuleList([DeepQSM.Decoder(base * 2 ** i, base * 2 ** (i - 1))
                                        for i in range(depth, 0, -1)])
        self._output = nn.Conv3d(base, 1, 1, 1, 0)

    def forward(self, x):
        skips = []
        skip, inEncoder = self._input(x)
        skips.append(skip)

        for encoder in self._encoders:
            skip, inEncoder = encoder(inEncoder)
            skips.append(skip)

        inDecoder = inEncoder
        skips.pop()
        skips.reverse()

        decoders = [inDecoder]
        for decoder, skip in zip(self._decoders, skips):
            inDecoder = decoder(inDecoder, skip)
            decoders.append(inDecoder)
        #
        # return self._output(inDecoder) * mask
        return decoders, self._output(inDecoder)


class LargeKernelUnet(nn.Module):

    class Encoder(nn.Module):

        def __init__(self, in_channel, out_channel):
            super(LargeKernelUnet.Encoder, self).__init__()
            self._input = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 5, 1, 2),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
            self._output = nn.Sequential(
                nn.Conv3d(out_channel, out_channel, 5, 1, 2),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

        def forward(self, feature_map):
            mid = self._input(feature_map)
            res = self._output(mid)
            return res

    class Decoder(nn.Module):

        def __init__(self, in_channel, out_channel):
            super(LargeKernelUnet.Decoder, self).__init__()
            self._input = nn.Sequential(
                nn.ConvTranspose3d(in_channel, out_channel, 2, stride=2),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
            self._mid = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 5, 1, 2),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
            self._output = nn.Sequential(
                nn.Conv3d(out_channel, out_channel, 5, 1, 2),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

        def forward(self, feature_map, skip):
            x = self._input(feature_map)
            mid = self._mid(torch.cat([x, skip], dim=1))
            res = self._output(mid)
            return res

    def __init__(self, depth, base):
        super(LargeKernelUnet, self).__init__()
        self.depth = depth
        self._input = LargeKernelUnet.Encoder(1, base)
        self._encoders = nn.ModuleList([nn.Sequential(nn.MaxPool3d(2),
                                        LargeKernelUnet.Encoder(base * 2 ** i, base * 2 ** (i + 1)))
                                        for i in range(depth)])
        self._decoders = nn.ModuleList([LargeKernelUnet.Decoder(base * 2 ** i, base * 2 ** (i - 1))
                                        for i in range(depth, 0, -1)])
        self._output = nn.Conv3d(base, 1, 1, 1, 0)

    def forward(self, x):
        skips = []
        inEncoder = self._input(x)
        skips.append(inEncoder)

        for encoder in self._encoders:
            inEncoder = encoder(inEncoder)
            skips.append(inEncoder)

        inDecoder = inEncoder
        skips.pop()
        skips.reverse()

        for decoder, skip in zip(self._decoders, skips):
            inDecoder = decoder(inDecoder, skip)

        return self._output(inDecoder)
