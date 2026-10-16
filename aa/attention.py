# aa/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from aa.config import AAConfig
from aa.router import AARouter


class _Path(nn.Module):
    """单条注意力路径：W_Q + W_O（shared_kv=True）或完整 W_QKV + W_O（False）。"""

    def __init__(self, config: AAConfig, shared_kv: bool):
        super().__init__()
        d = config.n_embd
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = None if shared_kv else nn.Linear(d, d, bias=False)
        self.wv = None if shared_kv else nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, x, rope, shared_k=None, shared_v=None):
        B, T, d = x.shape
        n_h, hd = self.n_head, self.head_dim
        q = self.wq(x).view(B, T, n_h, hd).transpose(1, 2)
        k = self.wk(x) if self.wk is not None else shared_k
        v = self.wv(x) if self.wv is not None else shared_v
        k = k.view(B, T, n_h, hd).transpose(1, 2)
        v = v.view(B, T, n_h, hd).transpose(1, 2)
        q, k = apply_rope(q, rope), apply_rope(k, rope)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, n_h * hd)
        return self.wo(out)


def apply_rope(x, rope):
    cos, sin = rope  # [T, hd/2]
    B, n_h, T, hd = x.shape
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SharedKVPathAttention(nn.Module):
    def __init__(self, config: AAConfig):
        super().__init__()
        self.config = config
        self.router = AARouter(config)
        d = config.n_embd
        self.wk_shared = nn.Linear(d, d, bias=False) if config.shared_kv else None
        self.wv_shared = nn.Linear(d, d, bias=False) if config.shared_kv else None
        self.paths = nn.ModuleList(
            [_Path(config, config.shared_kv) for _ in range(config.n_paths)])
        for p in self.paths:
            p.n_head, p.head_dim = config.n_head, config.head_dim
        self.resid_dropout = nn.Dropout(config.dropout)

    def attn_params(self, active_only: bool = False):
        """attention 参数遍历。active_only=True 只数 K=1 激活口径
        （共享 KV + 1 条路径），用于匹配约束断言。"""
        if self.config.shared_kv:
            for w in (self.wk_shared.weight, self.wv_shared.weight):
                yield w
            if active_only:
                yield from self.paths[0].parameters()
                return
        for p in self.paths:
            yield from p.parameters()

    def path_outputs(self, x, rope):
        B, T, d = x.shape
        if self.config.shared_kv:
            k, v = self.wk_shared(x), self.wv_shared(x)
            return [p(x, rope, shared_k=k, shared_v=v) for p in self.paths]
        return [p(x, rope) for p in self.paths]

    def forward(self, x, rope):
        B, T, d = x.shape
        weights, idx, rstats = self.router(x)          # [B,T,K]
        outs = self.path_outputs(x, rope)              # E × [B,T,d]
        stacked = torch.stack(outs, dim=-2)            # [B,T,E,d]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        sel = stacked.gather(-2, gather_idx)           # [B,T,K,d]
        y = (sel * weights.unsqueeze(-1)).sum(-2)      # [B,T,d]
        usage = torch.zeros(self.config.n_paths, device=x.device)
        usage.scatter_add_(0, idx.reshape(-1),
                           torch.ones(idx.numel(), device=x.device))
        return self.resid_dropout(y), {**rstats, "path_usage": usage}
