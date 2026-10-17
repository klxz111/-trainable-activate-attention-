# tests/test_checkpoint.py
# gradient checkpointing：数值与普通前向/反向一致，usage 统计不受影响。
import torch
from aa.config import AAConfig
from aa.attention import SharedKVPathAttention


def _rope(T, hd):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
    freqs = torch.outer(torch.arange(T).float(), inv)
    return torch.cos(freqs), torch.sin(freqs)


def _cfg_all(flag):
    return AAConfig(n_embd=128, n_head=4, n_paths=4, n_active=1,
                    shared_kv=True, grad_checkpoint=bool(flag))


def test_checkpoint_matches_plain():
    torch.manual_seed(0)
    c_off = _cfg_all(0); c_on = _cfg_all(1)
    m_off = SharedKVPathAttention(c_off)
    m_on = SharedKVPathAttention(c_on)
    m_on.load_state_dict(m_off.state_dict())
    x = torch.randn(2, 16, 128, requires_grad=True)
    rope = _rope(16, 32)
    y0, s0 = m_off(x, rope); y0.sum().backward()
    y1, s1 = m_on(x, rope); y1.sum().backward()
    assert torch.allclose(y0, y1, atol=1e-5)
    assert torch.allclose(m_on.paths[0].wq.weight.grad,
                          m_off.paths[0].wq.weight.grad, atol=1e-5)
    assert torch.allclose(s1["path_usage"], s0["path_usage"])


def test_checkpoint_off_by_default():
    c = _cfg_all(0)
    assert c.grad_checkpoint is False


def test_usage_still_counted_under_checkpoint():
    c = _cfg_all(1)
    m = SharedKVPathAttention(c)
    m.train()
    rope = _rope(16, 32)
    y, s = m(torch.randn(2, 16, 128), rope)
    assert s["path_usage"].shape == (4,)
    assert int(s["path_usage"].sum()) == 2 * 16 * 1  # B*T*K
