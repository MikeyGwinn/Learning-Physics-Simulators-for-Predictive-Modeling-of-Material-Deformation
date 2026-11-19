import torch
from flash_attn.modules.mha import MHA
from torch import nn
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


class CrossAttentionEncoderBlock(nn.Module):
    """
    Cross Attention layer between points' features as query and action (as key and value).

    Code modified from: https://github.com/tonyduan/transformer-blocks/blob/master/src/blocks.py#L172
    """

    def __init__(
        self, dim, hidden_dim, num_heads=8, dropout_prob=0.0, use_flash_attn=False
    ):
        super().__init__()
        self.attn = MHA(dim, num_heads, cross_attn=True, use_flash_attn=use_flash_attn)
        self.ffn = PositionwiseFFN(dim, hidden_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(
        self,
        points,
        action,
        cu_seqlens=None,
        max_seqlen=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
    ):

        x_ = self.ln1(points)
        if cu_seqlens is None:
            action = action.unsqueeze(0)
        if cu_seqlens is not None:
            attn = self.attn(
                x=x_,
                x_kv=action,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max_seqlen_k,
            )
        else:
            attn = self.attn(x=x_, x_kv=action)
        x = points + self.dropout(attn)
        x_ = self.ln2(x)
        x = x + self.dropout(self.ffn(x_))
        return x


class CrossAttentionEncoder(nn.Module):
    """
    Cross Attention Encoder consisting of stacked layers of `CrossAttentionEncoderBlock`
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
                CrossAttentionEncoderBlock(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                    use_flash_attn=use_flash_attn,
                )
            )

    def forward(
        self,
        points,
        action,
        cu_seqlens=None,
        max_seqlen=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
    ):
        """
        points: (B*N, dim)
        action: (B, dim)
        """
        with torch.autocast("cuda", dtype=torch.float16):
            max_seqlen = (
                max(cu_seqlens[1:] - cu_seqlens[:-1]).item()
                if cu_seqlens is not None
                else None
            )
            x = points
            for layer in self.layers:
                x = layer(
                    points=x,
                    action=action,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_k=max_seqlen_k,
                )
            return x
