# tests/test_model.py
import torch
from aa.config import AAConfig
from aa.model import AATransformer, count_params

def test_forward_shapes_and_loss():
    c = AAConfig(n_layer=2, n_embd=128, n_head=4, n_paths=4, vocab_size=256,
                 block_size=64)
    m = AATransformer(c)
    idx = torch.randint(0, 256, (2, 64))
    logits, loss, aux = m(idx, targets=idx)
    # 分块 CE 后训练路径不再物化全长 logits（E16 显存优化的核心契约）；
    # 无 targets 分支仍返回全长 logits（因果性测试依赖）。
    assert logits is None and loss is not None
    assert "path_usage" in aux

def test_tied_embedding_param_count():
    c = AAConfig(n_layer=2, n_embd=128, n_head=4, n_paths=1, vocab_size=256,
                 block_size=64)
    m = AATransformer(c)
    # attn 4d² + ffn 8d² + 2×LN(2d) = 12d²+4d/层, tied emb 一次, +ln_f +router gate(d×E)
    per_layer = 12 * 128 * 128 + 2 * 2 * 128
    expected = (2 * per_layer + 256 * 128 + 2 * 128
                + 2 * 128 * 1)  # router gate: n_layer × n_embd × n_paths
    assert count_params(m) == expected

def test_active_attn_params_formula():
    d = 128
    c = AAConfig(n_layer=2, n_embd=d, n_head=4, n_paths=8, n_active=1,
                 shared_kv=True, vocab_size=256)
    assert c.active_attn_params() == c.n_layer * 4 * d * d  # 每层激活 4d²

def test_rope_causality_end_to_end():
    c = AAConfig(n_layer=2, n_embd=128, n_head=4, n_paths=2, vocab_size=256,
                 block_size=64)
    m = AATransformer(c).eval()
    x = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        l1, _, _ = m(x)
        x2 = x.clone(); x2[:, 40:] = (x2[:, 40:] + 7) % 256
        l2, _, _ = m(x2)
    assert torch.allclose(l1[:, :40], l2[:, :40], atol=1e-4)
