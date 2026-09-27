import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=in_channels,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyMaskPredictor(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=16,
                 channels=(16, 32, 64, 96)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, base_channels, kernel_size=3, stride=2),
            DepthwiseSeparableBlock(base_channels, c1, stride=1),
        )
        self.enc2 = DepthwiseSeparableBlock(c1, c2, stride=2)
        self.enc3 = DepthwiseSeparableBlock(c2, c3, stride=2)
        self.enc4 = DepthwiseSeparableBlock(c3, c4, stride=2)
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableBlock(c4, c4, stride=1),
            DepthwiseSeparableBlock(c4, c4, stride=1),
        )
        self.dec3 = nn.Sequential(
            ConvNormAct(c4 + c3, c3, kernel_size=3),
            DepthwiseSeparableBlock(c3, c3, stride=1),
        )
        self.dec2 = nn.Sequential(
            ConvNormAct(c3 + c2, c2, kernel_size=3),
            DepthwiseSeparableBlock(c2, c2, stride=1),
        )
        self.low_res_head = nn.Sequential(
            ConvNormAct(c2, c1, kernel_size=3),
            nn.Conv2d(c1, 1, kernel_size=1),
        )
        self.apply(utils.weight_init)

    def forward(self, image):
        input_hw = image.shape[-2:]
        feat1 = self.stem(image)  # 112
        feat2 = self.enc2(feat1)  # 56
        feat3 = self.enc3(feat2)  # 28
        feat4 = self.enc4(feat3)  # 14
        bottleneck = self.bottleneck(feat4)

        up3 = F.interpolate(bottleneck,
                            size=feat3.shape[-2:],
                            mode='bilinear',
                            align_corners=False)
        up3 = self.dec3(torch.cat([up3, feat3], dim=1))

        up2 = F.interpolate(up3,
                            size=feat2.shape[-2:],
                            mode='bilinear',
                            align_corners=False)
        up2 = self.dec2(torch.cat([up2, feat2], dim=1))

        low_res_logits = self.low_res_head(up2)
        logits = F.interpolate(low_res_logits,
                               size=input_hw,
                               mode='bilinear',
                               align_corners=False)
        return logits
