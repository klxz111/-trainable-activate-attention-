# aa/config.py
from dataclasses import dataclass

@dataclass
class AAConfig:
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    n_paths: int = 1
    n_active: int = 1
    shared_kv: bool = True
    router_mode: str = "ste"
    vocab_size: int = 50257
    block_size: int = 1024
    dropout: float = 0.0
    grad_checkpoint: bool = False

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def active_attn_params(self) -> int:
        """K=1 激活口径的 attention 参数（每层），解析式。"""
        per_layer = 4 * self.n_embd * self.n_embd
        return self.n_layer * per_layer

    def total_attn_params(self) -> int:
        per_layer = ((2 if self.shared_kv else 4 * self.n_paths) *
                     self.n_embd * self.n_embd +
                     (0 if self.shared_kv else 0))
        if self.shared_kv:
            per_layer = (2 + 2 * self.n_paths) * self.n_embd * self.n_embd
        return self.n_layer * per_layer

    def active_params(self) -> int:
        d2 = self.n_embd * self.n_embd
        attn = (2 + 2 * self.n_active) if self.shared_kv else 4 * self.n_active
        ffn = 8
        total = self.n_layer * (attn + ffn) * d2
        total += self.vocab_size * self.n_embd + 2 * self.n_layer * 2 * self.n_embd
        total += 2 * self.n_embd  # ln_f
        total += self.n_layer * self.n_paths * self.n_embd  # router gate
        return total

    def total_params(self) -> int:
        """全路径物化口径的总参数（E 全部计入），解析式，供扫描表生成。"""
        d2 = self.n_embd * self.n_embd
        attn = (2 + 2 * self.n_paths) if self.shared_kv else 4 * self.n_paths
        ffn = 8
        total = self.n_layer * (attn + ffn) * d2
        total += self.vocab_size * self.n_embd + 2 * self.n_layer * 2 * self.n_embd
        total += 2 * self.n_embd  # ln_f
        total += self.n_layer * self.n_paths * self.n_embd  # router gate
        return total
