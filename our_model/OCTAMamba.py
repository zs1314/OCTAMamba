import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import time
from functools import partial
from typing import Optional, Callable
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
from our_model.MDR import MultiScaleConvModule
from our_model.DAM import DualAttentionModule
from our_model.wtconv2d import *

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.dual_att = DualAttentionModule(in_channels=self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        z = z.permute(0, 3, 1, 2).contiguous()  # B C H W
        z = self.dual_att(z)  # B C H W
        z = z.permute(0, 2, 3, 1).contiguous()
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class AvgPoolingChannel(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class MaxPoolingChannel(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)


class SEAttention(nn.Module):
    def __init__(self, channel=3, reduction=3):
        super().__init__()
        # 池化层，将每一个通道的宽和高都变为 1 (平均值)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # 先降低维
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # 再升维
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # y是权重
        return x * y.expand_as(x)


# Quad Stream Efficient Mining Embedding
class QSEME(nn.Module):
    def __init__(self, out_c):
        super().__init__()

        self.out_c = out_c
        self.se = SEAttention(channel=out_c)
        self.maxpool = MaxPoolingChannel()
        self.avgpool = AvgPoolingChannel()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )

        self.wtconv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )

        self.wtconv2 = nn.Sequential(
            WTConv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )

        self.wtconv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
        )
        # self.out_conv=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.out_c, kernel_size=1),
            nn.BatchNorm2d(num_features=self.out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.init_conv(x)  # BCHW

        x1, x2, x3, x4 = x.chunk(4, dim=1)
        # branch1_maxpool
        chaneel_1_max_pool = self.maxpool(x1)
        desired_size = (x1.size(2), x1.size(3))
        channel_1_max_pool_out = F.interpolate(chaneel_1_max_pool, size=desired_size, mode='bilinear',
                                               align_corners=False)

        # branch2_avgpool
        channel_2_avg_pool = self.avgpool(x2)
        desired_size = (x2.size(2), x2.size(3))
        channel_2_avg_pool_out = F.interpolate(channel_2_avg_pool, size=desired_size, mode='bilinear',
                                               align_corners=False)

        # branch3_Wtconv
        channel_3_1 = self.wtconv1(x3)
        channel_3_2 = self.wtconv2(channel_3_1)
        channel_3_3_out = self.wtconv3(channel_3_2)

        # branch4_residual
        channel_4 = x4

        output = torch.cat((channel_1_max_pool_out, channel_2_avg_pool_out, channel_3_3_out, channel_4), dim=1)
        output = self.out_conv(output)
        output = self.se(output)
        return output


# AttentionGate
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# DAVSSM
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.hidden_dim = hidden_dim

    def forward(self, input: torch.Tensor):
        x_mamba = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x_mamba


# OCTA-Mamba Block
class OCTAMambaBlock(nn.Module):
    def __init__(self, in_c, out_c, ):
        super().__init__()
        self.in_c = in_c
        self.conv = MultiScaleConvModule(in_channels=in_c, out_channels=out_c)
        self.ln = nn.LayerNorm(out_c)
        self.act = nn.GELU()
        self.block = VSSBlock(hidden_dim=out_c)
        self.scale = nn.Parameter(torch.ones(1))
        self.residual_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding="same")

    def forward(self, x):
        skip = x
        skip = self.residual_conv(skip)
        x = self.conv(x)

        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.block(x)
        x = x.permute(0, 3, 1, 2)  # B C H W

        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.act(self.ln(x))
        x = x.permute(0, 3, 1, 2)  # B C H W
        return x + skip * self.scale


# Encoder Block
class EncoderBlock(nn.Module):
    """Encoding then downsampling"""

    def __init__(self, in_c, out_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.octamamba = OCTAMambaBlock(in_c, out_c)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.octamamba(x)
        skip = self.act(self.bn(x))
        x = self.down(skip)
        return x, skip


### Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.attGate = Attention_block(F_g=in_c, F_l=skip_c, F_int=skip_c // 2)

        self.bn2 = nn.BatchNorm2d(in_c + skip_c, out_c)
        self.octamamba = OCTAMambaBlock(in_c + skip_c, out_c)
        self.act = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attGate(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn2(x))
        x = self.octamamba(x)
        return x


### **Model**
class OCTAMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.qseme = QSEME(out_c=16)

        """Encoder"""
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)

        """Decoder"""
        self.d3 = DecoderBlock(128, 128, 64)
        self.d2 = DecoderBlock(64, 64, 32)
        self.d1 = DecoderBlock(32, 32, 16)

        """Final"""
        self.conv_out = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Pre-Component"""
        x = self.qseme(x)

        """Encoder"""
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)

        """Decoder"""
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)

        """Final"""
        x = self.conv_out(x)
        x = self.sigmoid(x)
        return x


# 参数计算
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import torch
    import time
    from fvcore.nn import FlopCountAnalysis

    img = torch.randn(1, 1, 224, 224).to('cuda')
    model = OCTAMamba().to('cuda')
    # out = our_model(img)
    # print(out.shape)

    # 测试Flops和参数量大小
    from ptflops import get_model_complexity_info

    model = OCTAMamba().to('cuda')
    macs, params = get_model_complexity_info(model, (1, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print(f"Total FLOPs: {macs}")
    print(f"Total params: {params}")
    print(count_parameters(model) // 1e3)

    from thop import profile
    from thop import clever_format

    input = torch.randn(1, 1, 224, 224).to('cuda')
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
