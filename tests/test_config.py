# tests/test_config.py
from aa.config import AAConfig

def test_config_defaults_are_dense_equivalent():
    c = AAConfig()
    assert c.n_paths == 1 and c.n_active == 1 and c.shared_kv is True
    assert c.n_embd % c.n_head == 0

def test_config_smoke_scale():
    c = AAConfig(n_layer=4, n_embd=256, n_head=4, n_paths=4, vocab_size=50257)
    assert c.head_dim == 64
