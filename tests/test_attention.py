# tests/test_attention.py
import torch
from aa.config import AAConfig
from aa.attention import SharedKVPathAttention

def _rope(T, hd, device="cpu"):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
    t = torch.arange(T).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)

def test_e1_k1_matches_dense_param_count():
    d = 128
    c = AAConfig(n_embd=d, n_head=4, n_paths=1, n_active=1, shared_kv=True)
    m = SharedKVPathAttention(c)
    # dense: W_QKV 3d² + W_O d² = 4d²
    attn_params = sum(p.numel() for p in m.attn_params())
    assert attn_params == 4 * d * d

def test_e4_shared_kv_active_params_stay_4d2():
    d = 128
    c = AAConfig(n_embd=d, n_head=4, n_paths=4, n_active=1, shared_kv=True)
    m = SharedKVPathAttention(c)
    total = sum(p.numel() for p in m.attn_params())          # 2d² + 2E·d² = 10d²
    active = sum(p.numel() for p in m.attn_params(active_only=True))
    assert total == 10 * d * d
    assert active == 4 * d * d   # K=1 匹配约束（Global Constraints）

def test_e4_independent_kv_moa_arm():
    d = 128
    c = AAConfig(n_embd=d, n_head=4, n_paths=4, n_active=1, shared_kv=False)
    m = SharedKVPathAttention(c)
    total = sum(p.numel() for p in m.attn_params())
    assert total == 16 * d * d   # E·4d²

def test_causal_mask_no_future_leak():
    d = 128
    c = AAConfig(n_embd=d, n_head=4, n_paths=2, n_active=1, shared_kv=True)
    m = SharedKVPathAttention(c).eval()
    rope = _rope(8, c.head_dim)
    x = torch.randn(1, 8, d)
    with torch.no_grad():
        y1, _ = m(x, rope)
        x2 = x.clone(); x2[:, 5:] += 10.0   # 篡改未来 token
        y2, _ = m(x2, rope)
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-5)

def test_paths_diverge_on_same_kv():
    d = 128
    c = AAConfig(n_embd=d, n_head=4, n_paths=4, n_active=4, shared_kv=True)
    m = SharedKVPathAttention(c).eval()
    rope = _rope(8, c.head_dim)
    x = torch.randn(1, 8, d)
    with torch.no_grad():
        outs = m.path_outputs(x, rope)  # E 个 [B,T,d] 输出
    cos = torch.nn.functional.cosine_similarity(outs[0], outs[1], dim=-1)
    assert cos.mean() < 0.999  # 不同 W_Q^(e) 在共享 KV 上产生不同输出

def test_backward_reaches_all_paths_soft():
    d = 128
    c = AAConfig(n_embd=d, n_head=4, n_paths=4, n_active=2, shared_kv=True,
                 router_mode="soft")
    m = SharedKVPathAttention(c)
    rope = _rope(8, c.head_dim)
    y, _ = m(torch.randn(2, 8, d), rope)
    y.sum().backward()
    for e, path in enumerate(m.paths):
        assert path.wq.weight.grad is not None and path.wq.weight.grad.abs().sum() > 0, f"path {e} 无梯度"
