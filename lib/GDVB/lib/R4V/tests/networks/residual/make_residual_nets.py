import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, downsample=False):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
        self.bn_1 = nn.BatchNorm2d(16)

        out_size = 16 if downsample else 3
        self.conv_2 = nn.Conv2d(16, out_size, (3, 3), padding=(1, 1))
        self.bn_2 = nn.BatchNorm2d(out_size)

        self.downsample = downsample
        if self.downsample:
            self.ds_conv = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
            self.ds_bn = nn.BatchNorm2d(16)

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.ds_conv(residual)
            residual = self.ds_bn(residual)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)

        return x + residual


dummy_input = torch.ones((1, 3, 224, 224))

res1 = Residual()
torch.onnx.export(res1, dummy_input, "residual_identity.onnx")

res2 = Residual(downsample=True)
torch.onnx.export(res2, dummy_input, "residual_downsample.onnx")
