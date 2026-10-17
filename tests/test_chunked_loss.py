import torch
from aa.config import AAConfig
from aa.model import AATransformer

def _build():
    torch.manual_seed(0)
    cfg = AAConfig(n_layer=1, n_embd=64, n_head=4, n_paths=1, vocab_size=256,
                   block_size=64)
    return AATransformer(cfg)

def test_chunked_loss_matches_direct():
    m = _build(); m.eval()
    x = torch.randn(2, 64, 64)
    t = torch.randint(0, 256, (2, 64))
    import torch.nn.functional as F
    direct = F.cross_entropy(m.lm_head(x).reshape(-1, 256), t.reshape(-1))
    with torch.no_grad():
        chunked = m._chunked_loss(x, t)
    assert torch.allclose(direct, chunked, atol=1e-5)

def test_chunked_loss_grad_matches():
    m = _build(); m.train()
    x1 = torch.randn(2, 64, 64, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    t = torch.randint(0, 256, (2, 64))
    l1 = m._chunked_loss(x1, t)
    l1.backward()
    import torch.nn.functional as F
    # 直接口径（同 chunking-free）对照
    l2 = F.cross_entropy(m.lm_head(x2).reshape(-1, 256), t.reshape(-1))
    l2.backward()
    assert torch.allclose(l1, l2, atol=1e-5)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-4)
    g_on = m.lm_head.weight.grad.clone()
    m.zero_grad()
    # eval（无 checkpoint）路径对照
    m.eval()
    l3 = m._chunked_loss(x1.detach().requires_grad_(True), t)
    assert l3.shape == ()  # eval + 外层无 no_grad：_ce 直接执行，返回带梯度的标量 loss
