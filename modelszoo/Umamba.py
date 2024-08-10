"FBSNet: A Fast Bilateral Symmetrical Network for Real-Time Semantic Segmentation"
""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary


from torch.autograd import Variable
import time
import math
import copy
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count, parameter_count_table

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from torch import Tensor

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1

    print("导入")
except:
    pass


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


def selective_scan_flop_jit(inputs, outputs):
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    # for i in inputs:
    #     print(i.debugName())
    #     print(i.type().sizes())
    assert inputs[0].debugName().startswith("u")  # (B, D, L)
    assert inputs[2].debugName().startswith("A")  # (D, N)
    assert inputs[3].debugName().startswith("B")  # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("D")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("delta")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(self.args)


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn * s_attn  # [B, N, C]
        return ori_x * attn


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchMerging2DConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm, out_dim=None):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)

        return x



class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = self.norm(x)

        return x


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
            dropout=0.5,
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
        if self.expand is None:
            self.expand = 2

        if 'constrain_ss2d_expand' in kwargs and kwargs['constrain_ss2d_expand']:
            if d_model >= 384:
                self.expand = 1

        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.no_act_branch = False
        if self.no_act_branch:
            self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        else:
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

        self.bi_scan = kwargs['bi_scan'] if 'bi_scan' in kwargs else None
        self.merge_attn = kwargs['merge_attn'] if 'merge_attn' in kwargs else None
        # self.merge_attn_ratio = kwargs['merge_attn_ratio'] if 'merge_attn_ratio' in kwargs and isinstance(kwargs['merge_attn_ratio'], float) else None

        if self.merge_attn:
            self.attn = BiAttn(self.d_inner, act_ratio=.2)

        if not self.bi_scan:
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
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
            del self.dt_projs

            self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
            self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        else:
            self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            )
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
            del self.x_proj

            self.dt_projs = (
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
            )
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
            del self.dt_projs

            self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
            self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.forward_core = self.forward_corev2
        # self.forward_core = self.forward_corev0
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.dual_att=DualAttentionModule(in_channels=self.d_inner)
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

    # pixmamba seamamba
    def forward_corev2(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W

        if self.bi_scan:
            K = 2
        else:
            K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)

        if self.bi_scan:
            if self.bi_scan == 'xs':
                xs = torch.stack([x.view(B, -1, L), torch.flip(x.view(B, -1, L), dims=[-1])], dim=1)
            else:
                xs = x_hwwh
        else:
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        D, N = self.A_logs.shape

        out_y = []
        for i in range(K):
            yi = self.selective_scan(
                xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                delta_bias=dt_projs_bias.view(K, -1)[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)

        if self.merge_attn:
            out_y = [rearrange(self.attn(rearrange(out, 'b d l -> b l d')), 'b l d -> b d l') for out in out_y]

        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        if self.bi_scan:
            if self.bi_scan == 'xs':
                inv_y = torch.flip(out_y[:, 1], dims=[-1]).view(B, -1, L)
                y = out_y[:, 0] + inv_y
            else:
                wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
                # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
                y = out_y[:, 0] + wh_y

            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y).to(x.dtype)

        return y

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
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev0_seq(self, x: torch.Tensor):
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

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

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

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.view(B, K, -1, L)  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        Ds = self.Ds.view(-1)  # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1)  # (k * d)

        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        residual = x
        B, H, W, C = x.shape
        # print("x",x.shape)
        if self.no_act_branch:
            x = self.in_proj(x)
        else:
            xz = self.in_proj(x)
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        z = z.permute(0, 3, 1, 2).contiguous()  # B C H W
        z=self.dual_att(z)  # B C H W
        z = z.permute(0, 2, 3, 1).contiguous()


        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y = self.forward_core(x)
        y = self.out_norm(y)
        if not self.no_act_branch:
            y = y *z

        # out = self.SRFFN(out)
        # # out=self.effn(out)
        # out = out.permute(0, 2, 3, 1).contiguous()  # B H W C
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out + residual


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=2):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上



def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups
    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, height, width, groups, channels_per_group)

    x = torch.transpose(x, 3, 4).contiguous()

    # flatten
    x = x.view(batch_size, height, width, -1)

    return x


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = SS2D(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):

        x = x.permute(0, 2, 3, 1).contiguous()  # B H W C
        # x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=-1)
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        # x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        # x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        # x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        # x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=-1)

        x_mamba = self.norm(x_mamba)
        x_mamba=self.proj(x_mamba)
        x_mamba = x_mamba.permute(0, 3, 1, 2).contiguous()  # B C H W
        return x_mamba


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


class UltraLight_VM_UNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=1, c_list=[32, 64, 128, 256, 512,1024],
                 split_att='fc', bridge=False):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            PVMLayer(c_list[0], c_list[1]),
        )
        self.encoder3 = nn.Sequential(
            PVMLayer(c_list[1], c_list[2]),
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        self.decoder4 = nn.Sequential(
            PVMLayer(c_list[2], c_list[1]),
        )
        self.decoder5 = nn.Sequential(
            PVMLayer(c_list[1], c_list[0]),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2
        # print(t1.shape)
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import torch
    import time
    from fvcore.nn import FlopCountAnalysis

    # 测试Flops和参数量大小
    from ptflops import get_model_complexity_info

    input = torch.randn(1, 1, 224, 224).to('cuda')
    model = UltraLight_VM_UNet(num_classes=1,input_channels=1).to('cuda')
    # out=our_model(input)
    # print("out:",out.shape)


    macs, params = get_model_complexity_info(model, (1, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print(f"Total FLOPs: {macs}")
    print(f"Total params: {params}")
    print(count_parameters(model)//1e3)
    # throughput, latency = measure_latency(input_tensor, our_model)
    # print(throughput)
    # print(latency)

    from thop import profile
    from thop import clever_format

    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops,params)
