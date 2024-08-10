import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class SelfAtten(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        if (len(self.reduction_ratio) == 4):
            self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[3], reduction_ratio[3])
            self.sr1 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr2 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        elif (len(self.reduction_ratio) == 3):
            self.sr0 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr1 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if (len(self.reduction_ratio) == 4):
            tem0 = x[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
            tem1 = x[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem2 = x[:, 4704:5684, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem3 = x[:, 5684:6076, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))

        if (len(self.reduction_ratio) == 3):
            tem0 = x[:, :1568, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem1 = x[:, 1568:2548, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem2 = x[:, 2548:2940, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, tem2], -2))

        return reduce_out


class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio  # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class LocalEnhance_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.local_pos = DWConv(dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)
        local_v = v.permute(0, 2, 1, 3).reshape(B, N, C)
        local_pos = self.local_pos(local_v).reshape(B, -1, self.head, C // self.head).permute(0, 2, 1, 3)
        x_atten = ((attn_score @ v) + local_pos).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MixD_FFN(nn.Module):
    def __init__(self, c1, c2, fuse_mode="add"):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1) if fuse_mode == "add" else nn.Linear(c2 * 2, c1)
        self.fuse_mode = fuse_mode

    def forward(self, x):
        ax = self.dwconv(self.fc1(x), H, W)
        fuse = self.act(ax + self.fc1(x)) if self.fuse_mode == "add" else self.act(torch.cat([ax, self.fc1(x)], 2))
        out = self.fc2(ax)
        return out


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W


class TransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx


class FuseTransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, fuse_mode="add"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixD_FFN(dim, int(dim * 4), fuse_mode)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class MiT(nn.Module):
    def __init__(self, image_size, dims, layers, token_mlp='mix_skip'):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(image_size // 4, patch_sizes[1], strides[1], padding_sizes[1],
                                                   dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(image_size // 8, patch_sizes[2], strides[2], padding_sizes[2],
                                                   dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(image_size // 16, patch_sizes[3], strides[3], padding_sizes[3],
                                                   dims[2], dims[3])

        # transformer encoder
        self.block1 = nn.ModuleList([
            TransformerBlock(dims[0], heads[0], reduction_ratios[0], token_mlp)
            for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList([
            TransformerBlock(dims[1], heads[1], reduction_ratios[1], token_mlp)
            for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList([
            TransformerBlock(dims[2], heads[2], reduction_ratios[2], token_mlp)
            for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList([
            TransformerBlock(dims[3], heads[3], reduction_ratios[3], token_mlp)
            for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])

        # self.head = nn.Linear(dims[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class FuseMiT(nn.Module):
    def __init__(self, image_size, dims, layers, fuse_mode='add'):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(image_size // 4, patch_sizes[1], strides[1], padding_sizes[1],
                                                   dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(image_size // 8, patch_sizes[2], strides[2], padding_sizes[2],
                                                   dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(image_size // 16, patch_sizes[3], strides[3], padding_sizes[3],
                                                   dims[2], dims[3])

        # transformer encoder
        self.block1 = nn.ModuleList([
            FuseTransformerBlock(dims[0], heads[0], reduction_ratios[0], fuse_mode)
            for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList([
            FuseTransformerBlock(dims[1], heads[1], reduction_ratios[1], fuse_mode)
            for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList([
            FuseTransformerBlock(dims[2], heads[2], reduction_ratios[2], fuse_mode)
            for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList([
            FuseTransformerBlock(dims[3], heads[3], reduction_ratios[3], fuse_mode)
            for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])

        # self.head = nn.Linear(dims[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class Decoder(nn.Module):
    def __init__(self, dims, embed_dim, num_classes):
        super().__init__()

        self.linear_c1 = MLP(dims[0], embed_dim)
        self.linear_c2 = MLP(dims[1], embed_dim)
        self.linear_c3 = MLP(dims[2], embed_dim)
        self.linear_c4 = MLP(dims[3], embed_dim)

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim, 1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

        self.conv_seg = nn.Conv2d(128, num_classes, 1)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = inputs
        n = c1.shape[0]
        c1f = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        c2f = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        c2f = F.interpolate(c2f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c3f = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        c3f = F.interpolate(c3f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c4f = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        c4f = F.interpolate(c4f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c = self.linear_fuse(torch.cat([c4f, c3f, c2f, c1f], dim=1))
        c = self.dropout(c)
        return self.linear_pred(c)


segformer_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2], 256],  # [channel dimensions, num encoder layers, embed dim]
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2], 256],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3], 768],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3], 768],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3], 768],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3], 768]
}


class SegFormer(nn.Module):
    def __init__(self, model_name: str = 'B0', num_classes: int = 19, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in segformer_settings.keys(), f"SegFormer model name should be in {list(segformer_settings.keys())}"
        dims, layers, embed_dim = segformer_settings[model_name]

        self.backbone = MiT(image_size, dims, layers)
        self.decode_head = Decoder(dims, embed_dim, num_classes)

    def init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_outs = self.backbone(x)
        return self.decode_head(encoder_outs)
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class SegU_decoder(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, n_class=9, norm_layer=nn.LayerNorm,
                 is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
            # self.last_layer = None

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out


class BridgeLayer_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims, dims * 4)
        self.mixffn2 = MixFFN_skip(dims * 2, dims * 8)
        self.mixffn3 = MixFFN_skip(dims * 5, dims * 20)
        self.mixffn4 = MixFFN_skip(dims * 8, dims * 32)

    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem1 = tx[:, :3136, :].reshape(B, -1, C)
        tem2 = tx[:, 3136:4704, :].reshape(B, -1, C * 2)
        tem3 = tx[:, 4704:5684, :].reshape(B, -1, C * 5)
        tem4 = tx[:, 5684:6076, :].reshape(B, -1, C * 8)

        m1f = self.mixffn1(tem1, 56, 56).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)


        return t1


class BridgeLayer_3(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        # self.mixffn1 = MixFFN(dims,dims*4)
        self.mixffn2 = MixFFN(dims * 2, dims * 8)
        self.mixffn3 = MixFFN(dims * 5, dims * 20)
        self.mixffn4 = MixFFN(dims * 8, dims * 32)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c2f, c3f, c4f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        # tem1 = tx[:,:3136,:].reshape(B, -1, C)
        tem2 = tx[:, :1568, :].reshape(B, -1, C * 2)
        tem3 = tx[:, 1568:2548, :].reshape(B, -1, C * 5)
        tem4 = tx[:, 2548:2940, :].reshape(B, -1, C * 8)

        # m1f = self.mixffn1(tem1, 56, 56).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m2f, m3f, m4f], -2)

        tx2 = tx1 + t1

        return tx2


class BridegeBlock_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_4(dims, head, reduction_ratios)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B, _, C = bridge4.shape
        outs = []

        sk1 = bridge4[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
        sk2 = bridge4[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
        sk3 = bridge4[:, 4704:5684, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
        sk4 = bridge4[:, 5684:6076, :].reshape(B, 7, 7, C * 8).permute(0, 3, 1, 2)

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class BridegeBlock_3(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_3(dims, head, reduction_ratios)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        if (type(x) == list):
            # print("-----1-----")
            outs.append(x[0])
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B, _, C = bridge4.shape

        # sk1 = bridge2[:,:3136,:].reshape(B, 56, 56, C).permute(0,3,1,2)
        sk2 = bridge4[:, :1568, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
        sk3 = bridge4[:, 1568:2548, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
        sk4 = bridge4[:, 2548:2940, :].reshape(B, 7, 7, C * 8).permute(0, 3, 1, 2)

        # outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, token_mlp_mode, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
            # self.last_layer = None

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out


class MISSFormer(nn.Module):
    def __init__(self, num_classes=9, token_mlp_mode="mix_skip", encoder_pretrained=True):
        super().__init__()

        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        d_base_feat_size = 7  # 16 for 512 inputsize   7for 224
        in_out_chan = [[32, 64], [144, 128], [288, 320], [512, 512]]

        dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]
        self.backbone = MiT(320, dims, layers, token_mlp_mode)

        self.reduction_ratios = [1, 2, 4, 8]
        self.bridge = BridegeBlock_4(64, 1, self.reduction_ratios)

        self.decoder_3 = MyDecoderLayer((d_base_feat_size, d_base_feat_size), in_out_chan[3], heads[3],
                                        reduction_ratios[3], token_mlp_mode, n_class=num_classes)
        self.decoder_2 = MyDecoderLayer((d_base_feat_size * 2, d_base_feat_size * 2), in_out_chan[2], heads[2],
                                        reduction_ratios[2], token_mlp_mode, n_class=num_classes)
        self.decoder_1 = MyDecoderLayer((d_base_feat_size * 4, d_base_feat_size * 4), in_out_chan[1], heads[1],
                                        reduction_ratios[1], token_mlp_mode, n_class=num_classes)
        self.decoder_0 = MyDecoderLayer((d_base_feat_size * 8, d_base_feat_size * 8), in_out_chan[0], heads[0],
                                        reduction_ratios[0], token_mlp_mode, n_class=num_classes, is_last=True)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        encoder = self.backbone(x)
        bridge = self.bridge(encoder)  # list

        b, c, _, _ = bridge[3].shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        # ---------------Decoder-------------------------
        # print("stage3-----")
        tmp_3 = self.decoder_3(bridge[3].permute(0, 2, 3, 1).view(b, -1, c))
        # print("stage2-----")
        tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0, 2, 3, 1))
        # print("stage1-----")
        tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0, 2, 3, 1))
        # print("stage0-----")
        tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0, 2, 3, 1))
        tmp_0=self.sigmoid(tmp_0)
        return tmp_0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    import torch
    import time
    from fvcore.nn import FlopCountAnalysis

    img = torch.randn(1, 1, 224, 224).to('cuda')
    model = MISSFormer(num_classes=1).to('cuda')
    out = model(img)
    print(out.shape)

    # 测试Flops和参数量大小
    from ptflops import get_model_complexity_info

    model = MISSFormer(num_classes=1).to('cuda')
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