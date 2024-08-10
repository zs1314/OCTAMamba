import torch
from torch import nn


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DualAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=7):
        super(DualAttentionModule, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)
        self.act=nn.SiLU()
    def forward(self, x):
        # Apply Channel Attention Module
        x_out_cam = self.channel_attention(x)
        x_out_sam=self.spatial_attention(x)
        x_out=x_out_sam+x_out_cam
        x_out=x*x_out
        x_out=self.act(x_out)
        x_out=x_out*x
        x_out=x_out+x

        return x_out


# Testing the Dual Attention Module
if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 32, 32)  # Example input
    dam = DualAttentionModule(64)
    output = dam(input_tensor)
    print(output.shape)  # Should be the same as input shape
