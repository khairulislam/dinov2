import torch
from torch import Tensor, nn
from flash_attn import flash_attn_func
import logging, os, warnings

logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")

class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, C = x.shape  # Batch, Seq Len, Channels

        # Compute Q, K, V
        q = self.Wq(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Use Flash Attention
        attn_output = flash_attn_func(q, k, v, causal=False)  # Set causal=True for autoregressive models

        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(B, L, C)
        return self.out_proj(attn_output)

class MemEffFlashAttention(FlashAttention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x