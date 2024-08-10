import torch
import torch.nn as nn
import torch.nn.functional as F

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class MultiScaleConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvModule, self).__init__()
        middim=out_channels*4

        self.bnrelu1=BNPReLU(nIn=in_channels)

        # 1x1 convolutions
        self.conv1x1_1 = nn.Conv2d(in_channels, middim, kernel_size=1, dilation=1)

        # Convolutions with different kernel sizes and dilations
        self.conv1x3 = nn.Conv2d(middim, middim, kernel_size=(1, 3), dilation=1, padding=(0, 1),groups=middim)
        self.conv3x1 = nn.Conv2d(middim, middim, kernel_size=(3, 1), dilation=1, padding=(1, 0),groups=middim)
        self.conv3x3_1 = nn.Conv2d(middim, middim, kernel_size=3, dilation=3, padding=3,groups=middim)

        self.conv1x5 = nn.Conv2d(middim, middim, kernel_size=(1, 5), dilation=1, padding=(0, 2),groups=middim)
        self.conv5x1 = nn.Conv2d(middim, middim, kernel_size=(5, 1), dilation=1, padding=(2, 0),groups=middim)
        self.conv3x3_2 = nn.Conv2d(middim, middim, kernel_size=3, dilation=5, padding=5,groups=middim)

        self.conv1x7 = nn.Conv2d(middim, middim, kernel_size=(1, 7), dilation=1, padding=(0, 3),groups=middim)
        self.conv7x1 = nn.Conv2d(middim, middim, kernel_size=(7, 1), dilation=1, padding=(3, 0),groups=middim)
        self.conv3x3_3 = nn.Conv2d(middim, middim, kernel_size=3, dilation=7, padding=7,groups=middim)
        self.eca_1=eca_layer(middim)
        self.eca_2=eca_layer(middim)

        # Final 3x3 convolution
        self.conv3x3_final = nn.Conv2d(middim * 3, out_channels, kernel_size=3, dilation=1, padding=1,stride=1)

    def forward(self, x):
        # residual=self.conv1x1_0(x)
        # First path
        residual = x
        x=self.bnrelu1(x)

        x1 = self.conv1x1_1(x)
        # print("x1:", x1.shape)
        x_eca_1=self.eca_1(x1)
        # print("eca_1:", x_eca_1.shape)
        x_eca_2=self.eca_2(x1)
        # print("eca_2:", x_eca_2.shape)
        # print(x1.shape)
        # Second path

        x2 = self.conv1x3(x1)
        x2 = self.conv3x1(x2)
        x2 = self.conv3x3_1(x2)
        # print("x2:",x2.shape)
        # Third path

        x3 = self.conv1x5(x1)
        x3 = self.conv5x1(x3)
        x3 = self.conv3x3_2(x3)
        # print("x3:", x3.shape)
        # Fourth path

        x4 = self.conv1x7(x1)
        x4 = self.conv7x1(x4)
        x4 = self.conv3x3_3(x4)
        # print("x4:", x4.shape)
        x_branch1=x2+x_eca_1+x3
        # print("branch1:", x_branch1.shape)
        x_branch2=x3+x_eca_2+x4
        # print("branch_2:", x_branch2.shape)
        # Concatenate paths
        out = torch.cat([x_branch2, x_branch1,x1], dim=1)

        # Final 3x3 convolution
        out = self.conv3x3_final(out)

        return out



if __name__ == '__main__':
    import torch
    import time
    from fvcore.nn import FlopCountAnalysis

    # 测试Flops和参数量大小
    from ptflops import get_model_complexity_info

    input = torch.randn(1, 128, 224, 224).to('cuda')
    model = MultiScaleConvModule(in_channels=128,out_channels=256).to('cuda')
    out=model(input)
    print(out.shape)
    macs, params = get_model_complexity_info(model, (128, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print(f"Total FLOPs: {macs}")
    print(f"Total params: {params}")

    # throughput, latency = measure_latency(input_tensor, our_model)
    # print(throughput)
    # print(latency)

    from thop import profile
    from thop import clever_format

    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops,params)
