import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _norm(num_channels: int) -> nn.Module:
    num_groups = min(32, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _norm(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                _norm(out_channels),
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.double_conv(x)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CorrectorNet(nn.Module):
    def __init__(self, in_channels=101, out_channels=60, hidden_dim=256):
        super().__init__()
        bilinear = True
        self.inc = ResBlock(in_channels, hidden_dim)
        self.down1 = Down(hidden_dim, hidden_dim * 2)
        self.down2 = Down(hidden_dim * 2, hidden_dim * 4)
        self.down3 = Down(hidden_dim * 4, hidden_dim * 8)
        self.up1 = Up(hidden_dim * 8 + hidden_dim * 4, hidden_dim * 4, bilinear)
        self.up2 = Up(hidden_dim * 4 + hidden_dim * 2, hidden_dim * 2, bilinear)
        self.up3 = Up(hidden_dim * 2 + hidden_dim, hidden_dim, bilinear)
        self.outc = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        self.checkpoint_blocks = False
        nn.init.constant_(self.outc.weight, 0.0)
        nn.init.constant_(self.outc.bias, 0.0)

    def forward(self, x):
        def apply(module, *values):
            if self.checkpoint_blocks and torch.is_grad_enabled():
                return checkpoint(module, *values, use_reentrant=False)
            return module(*values)
        x1 = apply(self.inc, x)
        x2 = apply(self.down1, x1)
        x3 = apply(self.down2, x2)
        x4 = apply(self.down3, x3)
        x = apply(self.up1, x4, x3)
        x = apply(self.up2, x, x2)
        x = apply(self.up3, x, x1)
        return self.outc(x)
