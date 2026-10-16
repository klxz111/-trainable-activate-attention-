# aa/block.py
import torch.nn as nn
from aa.attention import SharedKVPathAttention
from aa.config import AAConfig


class TransformerBlock(nn.Module):
    def __init__(self, config: AAConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SharedKVPathAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, rope):
        y, rstats = self.attn(self.ln1(x), rope)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, rstats
