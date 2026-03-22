"""
WoundCNN – custom CNN backbone, built entirely from scratch.
No pretrained weights, no timm, no torchvision models.

Architecture:
  Stem:    Conv(3→64, 7×7, s2) → BN → ReLU → MaxPool
  Stage 1: ResBlock×3  64→64
  Stage 2: ResBlock×3  64→128, stride=2
  Stage 3: ResBlock×3  128→256, stride=2
  Stage 4: ResBlock×3  256→512, stride=2
  Pool:    AdaptiveAvgPool2d(1) → flatten → (B, 512)
"""

import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            ) if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        return F.relu(
            self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)), inplace=True)))
            + self.downsample(x),
            inplace=True,
        )


def _make_stage(in_ch, out_ch, num_blocks, stride=1):
    blocks = [ResBlock(in_ch, out_ch, stride=stride)]
    for _ in range(1, num_blocks):
        blocks.append(ResBlock(out_ch, out_ch))
    return nn.Sequential(*blocks)


class WoundCNN(nn.Module):
    """
    Custom CNN backbone trained from scratch.
    Input:  (B, 3, 224, 224)
    Output: (B, 512)
    """
    FEAT_DIM = 512

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = _make_stage(64,  64,  3, stride=1)
        self.stage2 = _make_stage(64,  128, 3, stride=2)
        self.stage3 = _make_stage(128, 256, 3, stride=2)
        self.stage4 = _make_stage(256, 512, 3, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.pool(x).flatten(1)  # (B, 512)
