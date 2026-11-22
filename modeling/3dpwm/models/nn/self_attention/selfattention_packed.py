import torch
import torch.nn as nn
from flash_attn.modules.mha import MHA
from torch.nn import functional as F


class PositionwiseFFN(nn.Module):
    """
    Position-wise FFN [Vaswani et al. NeurIPS 2017].
    Source Code: https://github.com/tonyduan/transformer-blocks/blob/master/src/blocks.py#L155C1-L170C1
    """

    def __init__(self, dim, hidden_dim, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        for module in (self.fc1, self.fc2):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class SelfAttentionEncoderBlock(nn.Module):
    """
    Self Attention layer between points' features as query, key and value.
    """

    def __init__(
        self, dim, hidden_dim, num_heads=8, dropout_prob=0.0, use_flash_attn=True
    ):
        super().__init__()
        self.attn = MHA(dim, num_heads, use_flash_attn=use_flash_attn)
        self.ffn = PositionwiseFFN(dim, hidden_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, points, cu_seqlens=None, max_seqlen=None):
        x_ = self.ln1(points)
        if cu_seqlens is None:
            x = points + self.dropout(self.attn(x=x_))
        else:
            x = points + self.dropout(
                self.attn(x=x_, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            )
        x_ = self.ln2(x)
        x = x + self.dropout(self.ffn(x_))
        return x


class SelfAttentionEncoder(nn.Module):
    """
    Self Attention Encoder consisting of stacked layers of `SelfAttentionEncoderBlock`
    """

    def __init__(
        self,
        dim,
        hidden_dim,
        num_heads=8,
        dropout_prob=0.0,
        num_layers=2,
        use_flash_attn=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                SelfAttentionEncoderBlock(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                    use_flash_attn=use_flash_attn,
                )
            )

    def forward(self, points, cu_seqlens=None):
        """
        points: (B*N, dim)
        """
        # assert cu_seqlens is not None, "cu_seqlens is required for packed self attention"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            max_seqlen = (
                max(cu_seqlens[1:] - cu_seqlens[:-1]).item()
                if cu_seqlens is not None
                else None
            )
            x = points
            for layer in self.layers:
                x = layer(points=x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            return x
