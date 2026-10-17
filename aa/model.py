# aa/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from aa.config import AAConfig
from aa.block import TransformerBlock


def _rope_cache(T, hd, device):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=device).float() / hd))
    freqs = torch.outer(torch.arange(T, device=device).float(), inv)
    return torch.cos(freqs), torch.sin(freqs)


class AATransformer(nn.Module):
    def __init__(self, config: AAConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        rope = _rope_cache(T, self.config.head_dim, idx.device)
        x = self.wte(idx)
        aux, usages = {}, []
        entropies = []
        for i, blk in enumerate(self.blocks):
            x, rstats = blk(x, rope)
            usages.append(rstats.pop("path_usage"))
            entropies.append(rstats.pop("mean_entropy"))
            aux[f"layer{i}"] = rstats
        aux["path_usage"] = torch.stack(usages)          # [L, E]
        aux["mean_entropy"] = sum(entropies) / len(entropies)
        x = self.ln_f(x)
        if targets is not None:
            loss = self._chunked_loss(x, targets)
            logits = None
        else:
            # 无 targets 时返回全长 logits [B, T, V]：
            # test_rope_causality_end_to_end 依赖 l[:, :40] 语义为前 40 个位置
            # （若只取末位 x[:, -1:, :]，末位本身被篡改，因果性断言无意义）。
            logits = self.lm_head(x)
            loss = None
        return logits, loss, aux

    def _chunked_loss(self, x, targets, chunk=256):
        """分块 lm_head+CE，每块用 checkpoint 包裹：backward 重算 logits，
        logits 峰值显存从 [B,T,V] 降到 [B,chunk,V]（E16 OOM 的最后大头）。"""
        from torch.utils.checkpoint import checkpoint
        B, T = x.shape[:2]
        total = x.new_zeros((), dtype=torch.float32)
        n = 0
        for i in range(0, T, chunk):
            xc, tc = x[:, i:i + chunk], targets[:, i:i + chunk]

            def _ce(xc=xc, tc=tc):
                logits = self.lm_head(xc).float()
                return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       tc.reshape(-1), reduction="sum")

            total = total + (checkpoint(_ce, use_reentrant=False)
                             if self.training and torch.is_grad_enabled()
                             else _ce())
            n += tc.numel()
        return total / n


def count_params(model, active_only: bool = False) -> int:
    if not active_only:
        return sum(p.numel() for p in model.parameters())
    return int(model.config.active_params())
