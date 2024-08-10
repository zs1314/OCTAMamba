import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
from collections import OrderedDict
import re
import math
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch.distributions.uniform import Uniform
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
from collections import OrderedDict
import re
import math
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch.distributions.uniform import Uniform


############ channel att #######
class ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


'''
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
'''


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
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x) + x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.eca = eca_layer(out_features, 3)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.view(B, C, H, W)
        x = self.eca(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x)
        return x


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        '''
        B,h,n1,n2 = attn.shape
        W1 = torch.mean(attn, 1, True)
        W = torch.sigmoid(W1)
        attn = attn * W
        '''
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            # attn = attn + attn.transpose(-2, -1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        ##### channel dimension
        '''
        if mask is  None:
            attn_c = (k.transpose(-2, -1) @ q)  
            attn_c = self.softmax(attn_c)
            x_c = (v @ attn_c).transpose(1, 2).reshape(B_, N, C)  # channel dimension
            x = x + x_c
        '''
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        #### channel ##
        # self.C_Att = ChannelBlock(dim=dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,drop_path=drop_path,norm_layer=nn.LayerNorm)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
            slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (
            slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # Channl
        # x = self.C_Att(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        # nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)
        return x


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


'''
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=[4], in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // 4, img_size[1] // 4] # only for flops calculation
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution

        self.in_chans = in_chans
        self.embed_dim = embed_dim


        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = 4
            padding = (ps - 4) // 2
            self.projs.append(nn.Conv2d(in_chans, dim, kernel_size=ps, stride=stride, padding=padding))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x).flatten(2).transpose(1, 2)
            xs.append(tx)  # B Ph*Pw C
        x = torch.cat(xs, dim=2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging(nn.Module):

    def __init__(self, dim, patch_size=[2,4], norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size
        self.norm = norm_layer(dim)

        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = 2 * dim // 2 ** i
            else:
                out_dim = 2 * dim // 2 ** (i + 1)
            stride = 2
            padding = (ps - stride) // 2
            #padding = math.ceil(((ps-1)*dilations[i] + 1 - stride) / 2)
            self.reductions.append(nn.Sequential(nn.Conv2d(dim, out_dim, kernel_size=ps, stride=stride, padding=padding)))

    def forward(self, x):
        B, L, C = x.shape
        x = self.norm(x)
        H = int(np.sqrt(L))
        W = int(np.sqrt(L))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x)
            xs.append(tmp_x)
        x = torch.cat(xs, dim=1)
        return x
'''


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=[4], in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // 4, img_size[1] // 4]  # only for flops calculation
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.eca = eca_layer(embed_dim, 3)

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = 2
            padding = (ps - stride) // 2
            self.projs.append(nn.Conv2d(in_chans, dim, kernel_size=ps, stride=2, padding=padding))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x)
            xs.append(tx)
        x = torch.cat(xs, dim=1)
        # print(x.shape)
        x = self.eca(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):

    def __init__(self, dim, patch_size=[2, 4], norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size
        self.norm = norm_layer(dim)
        self.eca = eca_layer(dim, 3)
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = 2 * dim // 2 ** i
            else:
                out_dim = 2 * dim // 2 ** (i + 1)
            stride = 2
            padding = (ps - stride) // 2
            # padding = math.ceil(((ps-1)*dilations[i] + 1 - stride) / 2)
            self.reductions.append(
                nn.Sequential(nn.Conv2d(dim, out_dim, kernel_size=ps, stride=stride, padding=padding)))

    def forward(self, x):
        B, L, C = x.shape
        x = self.norm(x)
        H = int(np.sqrt(L))
        W = int(np.sqrt(L))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x)
            xs.append(tmp_x)
        x = torch.cat(xs, dim=1)
        x = self.eca(x)
        return x


#####  seg
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Seg_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Seg_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Seg_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Seg_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


dpr = [x.item() for x in torch.linspace(0, 0.1, 6)]
num_heads = [8, 5, 2]
sr_ratios = [2, 4, 8]
curs = [0, 2, 4]


class Seg_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, cur, num_heads, sr_ratios):
        super(Seg_Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.Seg_block = nn.ModuleList(
            [Block(dim=2 * out_channels, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                   drop=0.0, attn_drop=0.0, drop_path=dpr[cur + i], norm_layer=nn.LayerNorm, sr_ratio=sr_ratios) for i
             in range(2)])
        self.norm = nn.LayerNorm(out_channels)
        self.Reduce = nn.Linear(2 * out_channels, out_channels, bias=True)

    def Seq2Img(self, x):
        B, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = x.flatten(2).transpose(1, 2)

        for blk in self.Seg_block:
            x = blk(x)
        x = self.Reduce(x)
        x = self.norm(x)
        x = self.Seq2Img(x)
        return x

class Res34_Swin_MS(nn.Module):  ###  low resolution + Multi-scale
    def __init__(self, image_size, block, layers, num_classes, zero_init_residual=False, groups=1, width_per_group=64):
        super(Res34_Swin_MS, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.sigmoid=nn.Sigmoid()
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.swin_layers = nn.ModuleList()
        embed_dim = 64
        self.num_layers = 4
        self.image_size = image_size
        depths = [2, 2, 2, 2]
        num_heads = [2, 4, 8, 16]
        window_size = self.image_size // 16
        self.mlp_ratio = 4.0
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        patches_resolution = [self.image_size // 2, self.image_size // 2]

        patch_size = [2, 4, 8, 16]
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
        self.MS2 = PatchMerging(64)
        self.MS3 = PatchMerging(128)
        self.MS4 = PatchMerging(256)

        for i_layer in range(self.num_layers):
            swin_layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                    input_resolution=(
                                    patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                                    depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size,
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False)
            self.swin_layers.append(swin_layer)

        channels = [64, 128, 256, 512]
        self.decode4 = Decoder(channels[3], channels[2])
        self.decode3 = Decoder(channels[2], channels[1])
        self.decode2 = Decoder(channels[1], channels[0])
        self.decode0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                     nn.Conv2d(channels[0], num_classes, kernel_size=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                            norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        encoder = []
        ms1 = self.patch_embed(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # print(x.shape,ms1.shape)
        x = x.flatten(2).transpose(1, 2)
        x = x + ms1

        x = self.swin_layers[0](x)
        B, L, C = x.shape
        ms2 = self.MS2(x)
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0, 3, 1, 2)
        encoder.append(x)

        x = self.layer2(x)
        x = x + ms2
        x = x.flatten(2).transpose(1, 2)
        x = self.swin_layers[1](x)
        B, L, C = x.shape
        ms3 = self.MS3(x)
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0, 3, 1, 2)
        encoder.append(x)

        x = self.layer3(x)
        x = x + ms3
        x = x.flatten(2).transpose(1, 2)
        x = self.swin_layers[2](x)
        B, L, C = x.shape
        ms4 = self.MS4(x)
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0, 3, 1, 2)
        encoder.append(x)

        x = self.layer4(x)
        x = x + ms4
        x = x.flatten(2).transpose(1, 2)
        x = self.swin_layers[3](x)
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0, 3, 1, 2)
        encoder.append(x)

        d4 = self.decode4(encoder[3], encoder[2])
        d3 = self.decode3(d4, encoder[1])
        d2 = self.decode2(d3, encoder[0])
        out = self.decode0(d2)
        out=self.sigmoid(out)
        return out


def res34_swin_MS(image_size, num_class):
    return Res34_Swin_MS(image_size, BasicBlock, [3, 4, 6, 3], num_classes=num_class)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    import torch
    import time
    from fvcore.nn import FlopCountAnalysis

    img = torch.randn(1, 1, 224, 224).to('cuda')
    model = res34_swin_MS(image_size=224,num_class=1).to('cuda')
    # out = our_model(img)
    # print(out.shape)

    # 测试Flops和参数量大小
    from ptflops import get_model_complexity_info

    model = res34_swin_MS(image_size=224,num_class=1).to('cuda')
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