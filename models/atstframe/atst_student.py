import math
import warnings
from functools import partial
import torch
from torch import nn
from .audio_transformer import *
from torchvision.models.resnet import BasicBlock
from ..conformer.encoder import ConformerBlock

class StudentSED(nn.Module):
    def __init__(self, nprompt=0, spec_h=64, spec_w=1001, patch_w=16, patch_h=16, pos_type="cut", in_chans=1,
                 num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.spec_w = spec_w
        self.spec_h = spec_h
        self.embed_dim = embed_dim
        self.patch_w = patch_w
        self.patch_h = patch_h

        self.pos_type = pos_type

        self.patch_embed = PatchEmbed_v2(patch_h, patch_w, embed_dim)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # hack
        self.nprompt = nprompt
        if self.nprompt > 0:
            self.prompt_embed = nn.Parameter(torch.zeros(1, self.nprompt, self.embed_dim))
            trunc_normal_(self.prompt_embed, std=.02)

        num_patches = get_num_patches(spec_h, spec_w, patch_h, patch_w)
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.conformer_blocks = nn.Sequential(*[
            ConformerBlock(encoder_dim = embed_dim,num_attention_heads = num_heads,feed_forward_expansion_factor=int(mlp_ratio),
                           feed_forward_dropout_p=drop_rate,attention_dropout_p=attn_drop_rate) for _ in range(depth)
        ])

        self.norm_frame = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.mask_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x, mask_index, length, mask=True):
        B, nc, h, w = x.shape
        mel_patches, x, patch_length = self.patch_embed(x, length)  # patch linear embedding
        B, T, C = x.shape

        if (mask_index is not None) and mask:
            mask_index_expand = mask_index.unsqueeze(2).expand(B, T, self.embed_dim).float()
            x = (1 - mask_index_expand) * x + mask_index_expand * self.mask_embed.expand(B, T, C)

        # add positional encoding to each token
        if self.pos_type == "cut":
            pos = self.pos_embed[:, 1:T + 1, :].expand(B, -1, -1)
            x = x + pos
        else:
            pos = self.interpolate_pos_encoding(x, h, w)
            x = x + pos[:, 1:]

        # pos = self.pos_embed[:,1:T+1,:].expand(B,-1,-1)
        # x = x + pos

        return self.pos_drop(x), pos, mel_patches, h, w, patch_length

    def forward(self, x, mask_index=None, mask_input=True, length=None):
        x, pos, mel_patches, h, w, patch_length = self.prepare_tokens(x, mask_index, length, mask_input)

        length_mask = torch.arange(mel_patches.shape[1]).to(x.device) < patch_length.unsqueeze(1)
        length_mask = length_mask.to(x.device)
        mask_index = mask_index & length_mask

        if self.nprompt > 0:
            x = torch.cat([self.prompt_embed.expand(x.shape[0], -1, -1), x], dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x, patch_length + self.nprompt)

        frame_repr = self.norm_frame(x)

        return frame_repr[:, self.nprompt:][mask_index]

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == self.spec_w and h == self.spec_h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_width
        h0 = h // self.patch_embed.patch_height
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, self.spec_h // self.patch_h, self.spec_w // self.patch_w, dim).permute(0, 3, 1,
                                                                                                              2),
            scale_factor=(h0 / (self.spec_h // self.patch_h), w0 / (self.spec_w // self.patch_w)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def get_last_selfattention(self, x):
        x, _, _, _, _, _ = self.prepare_tokens(x, mask_index=None, length=None, mask=False)
        atts = []
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x, att = blk(x, return_attention=True)
                atts.append(att)
            else:
                x, att = blk(x, return_attention=True)
                atts.append(att)
                return atts
                # return attention of the last block

    def get_intermediate_layers(self, x, length, n=1, scene=True, other_emb=None):
        x, _, _, _, _, patch_length = self.prepare_tokens(x, mask_index=None, length=length, mask=False)
        # we return the output tokens from the `n` last blocks
        if other_emb is not None:
            x = torch.cat([other_emb, x], dim=1)
        output = []
        if self.nprompt > 0:
            x = torch.cat([self.prompt_embed.expand(x.shape[0], -1, -1), x], dim=1)
        for i, blk in enumerate(self.conformer_blocks):
            x = blk(x)
            if len(self.conformer_blocks) - i <= n:
                norm_x = self.norm_frame(x)
                output.append(norm_x[:, self.nprompt:])

        return torch.cat(output, dim=-1)