# aa/router.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from aa.config import AAConfig


class AARouter(nn.Module):
    """Token-choice Top-K 路由。

    ste 模式: forward 硬 Top-K（评估与最终执行口径），
    backward 用 softmax 全概率直通（coefficient STE，MoE 标准做法）。
    soft 模式: 训练期全路径软混合，仅用于冒烟对比（proposal §4.1）。
    """

    def __init__(self, config: AAConfig):
        super().__init__()
        assert config.router_mode in ("ste", "soft")
        self.mode = config.router_mode
        self.E = config.n_paths
        self.K = config.n_active
        self.gate = nn.Linear(config.n_embd, config.n_paths, bias=False)

    def path_probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(x), dim=-1)  # [B, T, E]

    def forward(self, x: torch.Tensor):
        probs = self.path_probs(x)
        if self.mode == "soft":
            # soft 近似：忽略 K，全路径软混合（weights 即全 softmax 概率）
            B, T, _ = probs.shape
            idx = (torch.arange(self.E, device=probs.device)
                   .view(1, 1, self.E).expand(B, T, self.E))
            w = probs
        else:
            # Top-K 硬选择；K=E 时退化为 soft
            w_top, idx = torch.topk(probs, self.K, dim=-1)
            if self.K < self.E:
                # coefficient STE: 前向用归一化 top-K 权重，梯度走全 softmax
                w_norm = w_top / w_top.sum(-1, keepdim=True).clamp_min(1e-9)
                w = w_norm + w_top - w_top.detach()  # 直通
            else:
                w = probs
        stats = {
            "mean_max_weight": float(w.max(-1).values.mean()),
            "mean_entropy": float(-(probs * (probs + 1e-9).log()).sum(-1).mean()),
        }
        return w, idx, stats
