# tests/test_router.py
import torch
from aa.config import AAConfig
from aa.router import AARouter

def test_router_shapes_and_topk():
    c = AAConfig(n_embd=64, n_paths=4, n_active=1, router_mode="ste")
    r = AARouter(c)
    x = torch.randn(2, 8, 64)
    w, idx, stats = r(x)
    assert w.shape == (2, 8, 1) and idx.shape == (2, 8, 1)
    assert torch.allclose(w.sum(-1), torch.ones(2, 8), atol=1e-5)

def test_router_ste_gradient_flows():
    c = AAConfig(n_embd=64, n_paths=4, n_active=1, router_mode="ste")
    r = AARouter(c)
    x = torch.randn(2, 8, 64, requires_grad=True)
    w, idx, _ = r(x)
    w.sum().backward()
    assert r.gate.weight.grad is not None
    assert r.gate.weight.grad.abs().sum() > 0

def test_router_soft_mode_uses_full_distribution():
    c = AAConfig(n_embd=64, n_paths=4, n_active=4, router_mode="soft")
    r = AARouter(c)
    x = torch.randn(1, 4, 64)
    w, idx, _ = r(x)
    # soft 模式忽略 K：全 E 路径软混合
    assert w.shape == (1, 4, 4) and idx.shape == (1, 4, 4)
    p = r.path_probs(x)
    assert torch.allclose(w, p, atol=1e-5)

def test_router_uniform_input_balanced():
    torch.manual_seed(0)
    c = AAConfig(n_embd=64, n_paths=4, n_active=1, router_mode="ste")
    r = AARouter(c)
    x = torch.randn(64, 8, 64)  # 大 batch 随机输入
    p = r.path_probs(x).reshape(-1, 4).mean(0)
    assert p.std() < 0.25  # 随机初始化下无系统性偏斜
