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

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head
