from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from celle.reversible import SequentialSequence
from celle.attention import Attention

from rotary_embedding_torch import RotaryEmbedding, broadcat
from celle.utils import exists, default, cast_tuple

# https://arxiv.org/abs/2103.17239
class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


# layer norm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.Identity()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.norm_out(x)


# feed forward


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.0, mult=4.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


# main transformer class
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        causal=True,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        image_fmap_size=None,
        num_images=None,
        stable=False,
        rotary_emb=True,
    ):
        super().__init__()
        layers = nn.ModuleList([])

        self.seq_len = seq_len
        self.image_fmap_size = image_fmap_size

        for ind in range(depth):
            
            attn_class = partial(Attention, stable=stable)

            attn = attn_class(
                dim,
                causal=causal,
                seq_len=seq_len,
                heads=heads,
                dim_head=dim_head,
                dropout=attn_dropout,
            )

            ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)

            layers.append(
                nn.ModuleList(
                    [
                        LayerScale(
                            dim, ind + 1, PreNorm(dim, attn)
                        ),
                        LayerScale(
                            dim, ind + 1, PreNorm(dim, ff)
                        ),
                    ]
                )
            )

        # pairs arguments with attention layer
        route_attn = ((True, False),) * depth
        attn_route_map = {
            "mask": route_attn,
            "rotary_pos_emb": route_attn,
        }

        self.layers = SequentialSequence(layers, args_route=attn_route_map)

        # generate positional embeddings for rotary

        pos_emb = None
        if rotary_emb:
            rot_dim = dim_head // 3
            img_seq_len = ((image_fmap_size // num_images) ** 2) * num_images

            text_len = seq_len - img_seq_len + 1

            text_pos_emb = RotaryEmbedding(dim=rot_dim)

            img_axial_pos_emb = RotaryEmbedding(dim=rot_dim, freqs_for="pixel")

            text_freqs = text_pos_emb(torch.arange(text_len))

            img_to_text_freqs = text_pos_emb(
                torch.full((img_seq_len,), 8192)
            )  # image is given a position far away from text

            text_freqs = torch.cat((text_freqs, img_to_text_freqs), dim=0)

            img_freqs_axial = img_axial_pos_emb(
                torch.linspace(-1, 1, steps=image_fmap_size)
            )

            if num_images > 1:
                split_img_freqs_axial = torch.split(
                    img_freqs_axial, image_fmap_size // num_images, dim=0
                )

                split_img_freqs = [
                    broadcat(
                        (
                            rearrange(img_freqs_axial_per_image, "i d -> i () d"),
                            rearrange(img_freqs_axial_per_image, "j d -> () j d"),
                        ),
                        dim=-1,
                    )
                    for img_freqs_axial_per_image in split_img_freqs_axial
                ]

                split_img_freqs = [
                    rearrange(img_freqs_per_image, "h w d -> (h w) d")
                    for img_freqs_per_image in split_img_freqs
                ]

                # concat per image-image_freqs

                img_freqs = torch.cat(split_img_freqs, dim=0)

            elif num_images == 1:
                img_freqs = broadcat(
                    (
                        rearrange(img_freqs_axial, "i d -> i () d"),
                        rearrange(img_freqs_axial, "j d -> () j d"),
                    ),
                    dim=-1,
                )

                img_freqs = rearrange(img_freqs, "h w d -> (h w) d")

            else:
                assert False, "num_images must be int greater than 0"
            self.img_axial_pos_emb = img_axial_pos_emb
            self.text_pos_emb = text_pos_emb

            text_axial_freqs = img_axial_pos_emb(
                torch.full((text_len,), -10.0)
            )  # text is given a position of -10 apart from the image axial positions, which is from range [-1, 1]

            text_axial_freqs = torch.cat((text_axial_freqs, text_axial_freqs), dim=-1)

            img_freqs = torch.cat((text_axial_freqs, img_freqs), dim=0)

            pos_emb = torch.cat((text_freqs, img_freqs), dim=-1)

            pos_emb = rearrange(pos_emb, "n d -> () n d")

        self.register_buffer("pos_emb", pos_emb)

    def forward(self, x, **kwargs):
        return self.layers(x, rotary_pos_emb=self.pos_emb, **kwargs)