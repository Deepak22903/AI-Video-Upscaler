import torch
import torch.nn as nn

# A compact SR network inspired by SRResNet / RRDB-lite but much smaller
# Inputs: 3xHxW, Outputs: 3x(2xH)x(2xW) or configurable scale

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out * 0.1 + x

class LightweightESRGAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_feats=48, num_blocks=6, scale=2):
        super().__init__()
        self.scale = scale
        self.entry = nn.Conv2d(in_channels, num_feats, 3, 1, 1)

        body = [ResidualBlock(num_feats) for _ in range(num_blocks)]
        self.body = nn.Sequential(*body)
        self.mid = nn.Conv2d(num_feats, num_feats, 3, 1, 1)

        # Upsampler using pixelshuffle
        upsampler = []
        for _ in range(int(scale // 2) if scale % 2 == 0 else 1):
            upsampler += [nn.Conv2d(num_feats, num_feats * 4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(inplace=True)]
        if scale == 3:
            upsampler = [nn.Conv2d(num_feats, num_feats * 9, 3, 1, 1), nn.PixelShuffle(3), nn.ReLU(inplace=True)]

        self.upsampler = nn.Sequential(*upsampler)
        self.exit = nn.Conv2d(num_feats, out_channels, 3, 1, 1)

    def forward(self, x):
        fea = self.entry(x)
        out = self.body(fea)
        out = self.mid(out) + fea
        out = self.upsampler(out)
        out = self.exit(out)
        return out


if __name__ == "__main__":
    # simple smoke test
    net = LightweightESRGAN()
    x = torch.randn(1,3,64,64)
    y = net(x)
    print(y.shape)  # should be (1,3,128,128) for scale=2
