import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from rotary_embedding_torch import apply_rotary_emb
from celle.utils import exists, default, max_neg_value


# helpers
def stable_softmax(t, dim=-1, alpha=32**2):
    t = t / alpha
    t = t - torch.amax(t, dim=dim, keepdim=True).detach()
    return (t * alpha).softmax(dim=dim)


def apply_pos_emb(pos_emb, qkv):
    n = qkv[0].shape[-2]
    pos_emb = pos_emb[..., :n, :]
    return tuple(map(lambda t: apply_rotary_emb(pos_emb, t), qkv))


# classes
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        causal=False,
        heads=8,
        dim_head=64,
        dropout=0.0,
        stable=False,
        static_mask=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head**-0.5
        self.stable = stable
        self.causal = causal
        self.register_buffer("static_mask", static_mask, persistent=False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.save_attn = nn.Identity()

    def forward(self, x, context_mask=None, rotary_pos_emb=None):
        # x: [batch_size, seq_len, dim]
        b, n, _, h = *x.shape, self.heads
        device = x.device

        softmax = torch.softmax if not self.stable else stable_softmax

        # qkv: 3 tensors of shape [batch_size, seq_len, inner_dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # q,k,v: [batch_size, heads, seq_len, dim_head]
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb[..., :, :], (q, k, v))

        q *= self.scale

        # dots: [batch_size, heads, seq_len_i ,seq_len_j]
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        mask_value = max_neg_value(dots)

        if exists(context_mask):
            # context_mask: [batch_size ,1 ,1 ,seq_len_j]
            context_mask = rearrange(context_mask, "b j -> b 1 1 j")
            context_mask = F.pad(context_mask, (1, 0), value=True)

            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~context_mask, mask_value)

        if self.causal:
            i, j = dots.shape[-2:]
            context_mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
            dots.masked_fill_(context_mask, mask_value)

        if exists(self.static_mask):
            dots.masked_fill_(~self.static_mask[:n, :n], mask_value)

        # attn: [batch_size ,heads ,seq_len_i ,seq_len_j]
        attn = softmax(dots, dim=-1)
        attn = self.save_attn(attn)

        # out: [batch_size ,heads ,seq_len_i ,dim_head]
        out = torch.einsum("b h n j, b h j d -> b h n d", attn, v)

        # out: [batch_size ,seq_len_i ,(heads*dim_head)]
        out = rearrange(out, "b h n d -> b n (h d)")

        # out: [batch_size ,seq_len_i ,dim]
        out = self.to_out(out)

        return out


# sparse attention with convolutional pattern, as mentioned in the blog post. customizable kernel size and dilation


class SparseConvCausalAttention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        image_size=32,
        kernel_size=5,
        dilation=1,
        heads=8,
        dim_head=64,
        dropout=0.0,
        stable=False,
        **kwargs,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"

        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head**-0.5
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.stable = stable

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None, rotary_pos_emb=None):
        b, n, _, h, img_size, kernel_size, dilation, seq_len, device = (
            *x.shape,
            self.heads,
            self.image_size,
            self.kernel_size,
            self.dilation,
            self.seq_len,
            x.device,
        )
        softmax = torch.softmax if not self.stable else stable_softmax

        img_seq_len = img_size**2
        text_len = seq_len + 1 - img_seq_len

        # padding

        padding = seq_len - n + 1
        mask = default(mask, lambda: torch.ones(b, text_len, device=device).bool())

        x = F.pad(x, (0, 0, 0, padding), value=0)
        mask = mask[:, :text_len]

        # derive query / keys / values

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), qkv)

        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))

        q *= self.scale

        ((q_text, q_img), (k_text, k_img), (v_text, v_img)) = map(
            lambda t: (t[:, :-img_seq_len], t[:, -img_seq_len:]), (q, k, v)
        )

        # text attention

        dots_text = einsum("b i d, b j d -> b i j", q_text, k_text)
        mask_value = max_neg_value(dots_text)

        i, j = dots_text.shape[-2:]
        text_causal_mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
        dots_text.masked_fill_(text_causal_mask, mask_value)

        attn_text = softmax(dots_text, dim=-1)
        out_text = einsum("b i j, b j d -> b i d", attn_text, v_text)

        # image attention

        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding = effective_kernel_size // 2

        k_img, v_img = map(
            lambda t: rearrange(t, "b (h w) c -> b c h w", h=img_size), (k_img, v_img)
        )
        k_img, v_img = map(
            lambda t: F.unfold(t, kernel_size, padding=padding, dilation=dilation),
            (k_img, v_img),
        )
        k_img, v_img = map(
            lambda t: rearrange(t, "b (d j) i -> b i j d", j=kernel_size**2),
            (k_img, v_img),
        )

        # let image attend to all of text

        dots_image = einsum("b i d, b i j d -> b i j", q_img, k_img)
        dots_image_to_text = einsum("b i d, b j d -> b i j", q_img, k_text)

        # calculate causal attention for local convolution

        i, j = dots_image.shape[-2:]
        img_seq = torch.arange(img_seq_len, device=device)
        k_img_indices = rearrange(img_seq.float(), "(h w) -> () () h w", h=img_size)
        k_img_indices = F.pad(
            k_img_indices, (padding,) * 4, value=img_seq_len
        )  # padding set to be max, so it is never attended to
        k_img_indices = F.unfold(k_img_indices, kernel_size, dilation=dilation)
        k_img_indices = rearrange(k_img_indices, "b j i -> b i j")

        # mask image attention

        q_img_indices = rearrange(img_seq, "i -> () i ()")
        causal_mask = q_img_indices < k_img_indices

        # concat text mask with image causal mask

        causal_mask = repeat(causal_mask, "() i j -> b i j", b=b * h)
        mask = repeat(mask, "b j -> (b h) i j", i=i, h=h)
        mask = torch.cat((~mask, causal_mask), dim=-1)

        # image can attend to all of text

        dots = torch.cat((dots_image_to_text, dots_image), dim=-1)
        dots.masked_fill_(mask, mask_value)

        attn = softmax(dots, dim=-1)

        # aggregate

        attn_image_to_text, attn_image = attn[..., :text_len], attn[..., text_len:]

        out_image_to_image = einsum("b i j, b i j d -> b i d", attn_image, v_img)
        out_image_to_text = einsum("b i j, b j d -> b i d", attn_image_to_text, v_text)

        out_image = out_image_to_image + out_image_to_text

        # combine attended values for both text and image

        out = torch.cat((out_text, out_image), dim=1)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        out = self.to_out(out)

        return out[:, :n]