# Stage A: Activation Attention 容量扫描 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建三臂对照（Dense / AA 共享 KV / MoA 风格独立 KV）的小规模预训练框架，先冒烟验证路由行为，再跑 E∈{1,2,4,8,16} 容量扫描回答 RQ1。

**Architecture:** nanoGPT 风格的精简 causal Transformer（RoPE + pre-LN + tied embedding），attention 层替换为可配置的路径专家模块：共享 KV + E 个 (W_Q, W_O) 路径 + Top-K 路由（STE/soft 双模式）。`E=1,K=1` 严格退化为 dense MHA 作为基线臂；`shared_kv=False` 即 MoA 风格基线臂。同一代码路径三臂切换，消除实现差异。

**Tech Stack:** PyTorch ≥2.2（bf16 autocast）、tiktoken（GPT-2 vocab 50257）、numpy memmap 数据管线、pytest。

**Spec:** `README.md`（proposal 全文，尤其 §3.1 架构、§4.1 训练目标、§6 Stage A、§8 成功标准）

## Global Constraints

- 单卡 RTX 4070 Laptop 8GB：micro-batch ≤ 16（seq 1024），gradient checkpointing 默认开启
- 训练 bf16 autocast；**最终评估必须硬路由**（proposal §4.1）
- 匹配约束：K=1 主扫描下每层激活 attention 参数恒等于 dense 的 4d²（shared KV: 2d²+2Kd²=4d²）
- `E=1, K=1` 必须与 dense MHA 数学等价（同一前向路径，禁止分支特判基线）
- 模型规模锚点：L=8, d=512, ffn 4x, vocab 50257, RoPE, tied embedding → dense ≈ 51M 总参
- 判读协议（继承自 PushT 方法论）：单臂冒烟过判据才准上扫描；扫描 2 seeds；结论看 val loss 幂律拟合而非单点
- HF 下载失败时用 `HF_ENDPOINT=https://hf-mirror.com` 重试

## 文件结构

```
activation-attention/
├── aa/
│   ├── __init__.py
│   ├── config.py        # AAConfig dataclass（全部超参的唯一来源）
│   ├── router.py        # TopK 路由：STE / soft 双模式 + 利用率统计
│   ├── attention.py     # SharedKVPathAttention（shared_kv 开关 = AA/MoA 臂切换）
│   ├── block.py         # TransformerBlock（pre-LN 残差）
│   ├── model.py         # AATransformer（RoPE、tied emb、LM head）+ 参数/FLOPs 计数
│   ├── data.py          # uint16 bin memmap 数据集
│   ├── telemetry.py     # 路由遥测聚合（利用率/熵/路径相似度）
│   └── train.py         # 训练循环 CLI（AdamW + cosine + grad accum + ckpt）
├── scripts/
│   ├── prepare_data.py  # TinyStories（冒烟）/ OpenWebText 子集（扫描）→ bin
│   ├── run_smoke.sh     # 三臂冒烟一键
│   └── run_sweep.sh     # 5 容量点 × 2 seeds 一键
└── tests/
    ├── test_router.py
    ├── test_attention.py
    ├── test_model.py
    └── test_data.py
```

---

### Task 1: 环境与仓库骨架

**Files:**
- Create: `requirements.txt`, `aa/__init__.py`（空占位）, `tests/__init__.py`（空占位）

**Interfaces:**
- Produces: 可运行的 `python -c "import torch"` 环境（conda env `aa`），后续所有任务的 pytest 基础

- [ ] **Step 1: 创建 conda 环境并安装依赖**

```bash
conda create -n aa python=3.11 -y
conda run -n aa pip install torch --index-url https://download.pytorch.org/whl/cu121
conda run -n aa pip install numpy tiktoken datasets pytest
```

- [ ] **Step 2: 写 requirements.txt**

```
torch>=2.2
numpy
tiktoken
datasets
pytest
```

- [ ] **Step 3: 验证 GPU 可用**

Run: `conda run -n aa python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
Expected: `True NVIDIA GeForce RTX 4070 Laptop GPU`

- [ ] **Step 4: 验证 pytest 空跑**

Run: `cd /home/lhyzyrx/activation-attention && conda run -n aa python -m pytest tests/ -q`
Expected: `no tests ran`（0 failed）

- [ ] **Step 5: Commit**

```bash
git add requirements.txt aa/__init__.py tests/__init__.py
git commit -m "chore: aa 环境与仓库骨架"
```

---

### Task 2: AAConfig 配置中心

**Files:**
- Create: `aa/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `AAConfig` dataclass，后续所有模块的唯一超参来源

```python
# 核心字段（Task 3-6 依赖的精确签名）
@dataclass
class AAConfig:
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8          # 每 path 的 head 数（head_dim = n_embd // n_head = 64）
    n_paths: int = 1         # E：候选路径数
    n_active: int = 1        # K：每 token 激活路径数
    shared_kv: bool = True   # True=AA 臂；False=MoA 风格独立 KV 臂
    router_mode: str = "ste" # "ste" | "soft"
    vocab_size: int = 50257
    block_size: int = 1024
    dropout: float = 0.0
```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config.py
from aa.config import AAConfig

def test_config_defaults_are_dense_equivalent():
    c = AAConfig()
    assert c.n_paths == 1 and c.n_active == 1 and c.shared_kv is True
    assert c.n_embd % c.n_head == 0

def test_config_smoke_scale():
    c = AAConfig(n_layer=4, n_embd=256, n_head=4, n_paths=4, vocab_size=50257)
    assert c.head_dim == 64
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n aa python -m pytest tests/test_config.py -q`
Expected: FAIL（ModuleNotFoundError: aa.config）

- [ ] **Step 3: 实现**

```python
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
```

`aa/__init__.py` 加 `from aa.config import AAConfig`。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n aa python -m pytest tests/test_config.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add aa/config.py aa/__init__.py tests/test_config.py
git commit -m "feat: AAConfig 配置中心"
```

---

### Task 3: TopK 路由器（STE / soft 双模式）

**Files:**
- Create: `aa/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: `AAConfig`
- Produces: `AARouter(config)`，`forward(x) -> (weights, idx, stats)`
  - `weights`: `[B, T, K]` 选中路径的归一化门控权重 α
  - `idx`: `[B, T, K]` long 张量，选中路径下标
  - `stats`: dict `{"mean_max_weight": float, "mean_entropy": float}`
  - `path_probs(x) -> [B, T, E]`：完整 softmax 概率（遥测用）

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n aa python -m pytest tests/test_router.py -q`
Expected: FAIL（No module named 'aa.router'）

- [ ] **Step 3: 实现**

```python
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
                w = w_norm + probs - probs.detach()  # 直通
            else:
                w = probs
        stats = {
            "mean_max_weight": float(w.max(-1).values.mean()),
            "mean_entropy": float(-(probs * (probs + 1e-9).log()).sum(-1).mean()),
        }
        return w, idx, stats
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n aa python -m pytest tests/test_router.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add aa/router.py tests/test_router.py
git commit -m "feat: TopK 路由器 STE/soft 双模式"
```

---

### Task 4: SharedKVPathAttention（核心模块）

**Files:**
- Create: `aa/attention.py`
- Test: `tests/test_attention.py`

**Interfaces:**
- Consumes: `AAConfig`、`AARouter`
- Produces: `SharedKVPathAttention(config)`，`forward(x, rope) -> (y, route_stats)`
  - `x`: `[B, T, d]`；`rope`: `(cos, sin)` 各 `[T, head_dim/2]`（Task 5 生成）
  - `y`: `[B, T, d]`；`route_stats`: 含 `"path_usage"` `[E]` 张量（token 选中计数，遥测用）
- 数学定义（proposal §3.1）：
  - `shared_kv=True`: 共享 `W_K, W_V`（2d²），E 条路径各持 `W_Q^(e), W_O^(e)`（各 d²）
  - `shared_kv=False`: E 条路径各持完整 `W_QKV^(e), W_O^(e)`（MoA 风格基线臂）
  - 输出 `h' = h + Σ_{e∈TopK} α_e · Attn(Q^(e), K, V) W_O^(e)`，无路径间残差偏置

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n aa python -m pytest tests/test_attention.py -q`
Expected: FAIL（No module named 'aa.attention'）

- [ ] **Step 3: 实现**

```python
# aa/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from aa.config import AAConfig
from aa.router import AARouter


class _Path(nn.Module):
    """单条注意力路径：W_Q + W_O（shared_kv=True）或完整 W_QKV + W_O（False）。"""

    def __init__(self, config: AAConfig, shared_kv: bool):
        super().__init__()
        d = config.n_embd
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = None if shared_kv else nn.Linear(d, d, bias=False)
        self.wv = None if shared_kv else nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, x, rope, shared_k=None, shared_v=None):
        B, T, d = x.shape
        n_h, hd = self.n_head, self.head_dim
        q = self.wq(x).view(B, T, n_h, hd).transpose(1, 2)
        k = self.wk(x) if self.wk is not None else shared_k
        v = self.wv(x) if self.wv is not None else shared_v
        k = k.view(B, T, n_h, hd).transpose(1, 2)
        v = v.view(B, T, n_h, hd).transpose(1, 2)
        q, k = apply_rope(q, rope), apply_rope(k, rope)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, n_h * hd)
        return self.wo(out)


def apply_rope(x, rope):
    cos, sin = rope  # [T, hd/2]
    B, n_h, T, hd = x.shape
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SharedKVPathAttention(nn.Module):
    def __init__(self, config: AAConfig):
        super().__init__()
        self.config = config
        self.router = AARouter(config)
        d = config.n_embd
        self.wk_shared = nn.Linear(d, d, bias=False) if config.shared_kv else None
        self.wv_shared = nn.Linear(d, d, bias=False) if config.shared_kv else None
        self.paths = nn.ModuleList(
            [_Path(config, config.shared_kv) for _ in range(config.n_paths)])
        for p in self.paths:
            p.n_head, p.head_dim = config.n_head, config.head_dim
        self.resid_dropout = nn.Dropout(config.dropout)

    def attn_params(self, active_only: bool = False):
        """attention 参数遍历。active_only=True 只数 K=1 激活口径
        （共享 KV + 1 条路径），用于匹配约束断言。"""
        if self.config.shared_kv:
            for w in (self.wk_shared.weight, self.wv_shared.weight):
                yield w
            if active_only:
                yield from self.paths[0].parameters()
                return
        for p in self.paths:
            yield from p.parameters()

    def path_outputs(self, x, rope):
        B, T, d = x.shape
        if self.config.shared_kv:
            k, v = self.wk_shared(x), self.wv_shared(x)
            return [p(x, rope, shared_k=k, shared_v=v) for p in self.paths]
        return [p(x, rope) for p in self.paths]

    def forward(self, x, rope):
        B, T, d = x.shape
        weights, idx, rstats = self.router(x)          # [B,T,K]
        outs = self.path_outputs(x, rope)              # E × [B,T,d]
        stacked = torch.stack(outs, dim=-2)            # [B,T,E,d]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        sel = stacked.gather(-2, gather_idx)           # [B,T,K,d]
        y = (sel * weights.unsqueeze(-1)).sum(-2)      # [B,T,d]
        usage = torch.zeros(self.config.n_paths, device=x.device)
        usage.scatter_add_(0, idx.reshape(-1),
                           torch.ones(idx.numel(), device=x.device))
        return self.resid_dropout(y), {**rstats, "path_usage": usage}
```

注意：`path_outputs` 对所有 E 条路径都计算了输出再 gather——实现简单、E≤16 时正确；条件执行优化（真·省算力）是 Stage D 的事，此处 YAGNI。`soft` 模式 K=E 时 gather 自然取全路径，与权重一致。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n aa python -m pytest tests/test_attention.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add aa/attention.py tests/test_attention.py
git commit -m "feat: SharedKVPathAttention 核心 — AA/MoA 双臂 + STE 路由"
```

---

### Task 5: Transformer 主体 + 参数/FLOPs 计数

**Files:**
- Create: `aa/block.py`, `aa/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `SharedKVPathAttention`
- Produces:
  - `TransformerBlock(config)`：pre-LN + attention + MLP(4x, GELU)
  - `AATransformer(config)`：`forward(idx, targets=None) -> (logits, loss, aux)`
    - `idx`: `[B, T]` long；`aux`: dict（各层路由统计聚合）
    - RoPE 预计算 buffer，tied embedding（wte 与 lm_head 共享权重）
  - `count_params(model, active_only=False) -> int`
  - `AAConfig.total_params()` / `AAConfig.active_attn_params()`：解析式计数（E/K 的函数，供扫描表生成）

- [ ] **Step 1: 写失败测试**

```python
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
    assert logits.shape == (2, 64, 256) and loss is not None
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
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n aa python -m pytest tests/test_model.py -q`
Expected: FAIL

- [ ] **Step 3: 实现 block 与 model**

```python
# aa/block.py
import torch.nn as nn
from aa.attention import SharedKVPathAttention
from aa.config import AAConfig


class TransformerBlock(nn.Module):
    def __init__(self, config: AAConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SharedKVPathAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, rope):
        y, rstats = self.attn(self.ln1(x), rope)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, rstats
```

```python
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
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1))
        else:
            logits = self.lm_head(x[:, -1:, :])
            loss = None
        return logits, loss, aux


def count_params(model, active_only: bool = False) -> int:
    if not active_only:
        return sum(p.numel() for p in model.parameters())
    return int(model.config.active_params())
```

在 `AAConfig` 中追加解析式计数方法（与测试断言对齐）：

```python
    # 追加到 aa/config.py 的 AAConfig
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
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n aa python -m pytest tests/test_model.py tests/test_attention.py tests/test_router.py -q`
Expected: all passed

- [ ] **Step 5: 打印规模锚点表并人工核对**

Run:
```bash
conda run -n aa python -c "
from aa.config import AAConfig
for E in [1, 2, 4, 8, 16]:
    c = AAConfig(n_paths=E, n_active=1)
    print(f'E={E:2d}  total={c.total_params()/1e6:.1f}M  active_attn/layer={c.active_attn_params()//c.n_layer//1024}K')
"
```
Expected: E=1 总参 ~51M；E=16 总参 ~110M 级；每层激活 attn 参数恒 1024K（=4d²）。与 Global Constraints 核对，不符则修公式。

- [ ] **Step 6: Commit**

```bash
git add aa/block.py aa/model.py aa/config.py tests/test_model.py
git commit -m "feat: AATransformer 主体 + RoPE/tied-emb/参数计数"
```

---

### Task 6: 数据管线

**Files:**
- Create: `aa/data.py`, `scripts/prepare_data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: tiktoken GPT-2 编码器
- Produces:
  - `BinDataset(path, block_size)`：`__len__`、`__getitem__(i) -> (x, y)`，x/y 均 `[block_size]` long，uint16 memmap 零拷贝
  - `scripts/prepare_data.py --dataset tinystories|openwebtext --out data/<name> --max-tokens N`：产出 `train.bin` / `val.bin`（各 5% 验证）

- [ ] **Step 1: 写失败测试（不依赖网络，用合成 bin）**

```python
# tests/test_data.py
import numpy as np
import torch
from aa.data import BinDataset

def test_bin_dataset_shapes_and_targets(tmp_path):
    tokens = np.arange(1000, dtype=np.uint16)
    tokens.tofile(tmp_path / "t.bin")
    ds = BinDataset(str(tmp_path / "t.bin"), block_size=64)
    x, y = ds[0]
    assert x.shape == y.shape == (64,)
    assert x.dtype == torch.long
    assert (y[:-1] == x[1:]).all()          # next-token 对齐
    assert len(ds) == 1000 - 64             # 步长 1 窗口数
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n aa python -m pytest tests/test_data.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# aa/data.py
import numpy as np
import torch


class BinDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, block_size: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        j = np.random.randint(len(self)) if i == -1 else i
        chunk = self.data[j:j + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
```

```python
# scripts/prepare_data.py
"""下载 → GPT-2 tokenize → uint16 train/val.bin。
用法: python scripts/prepare_data.py --dataset tinystories --out data/ts --max-tokens 500_000_000
"""
import argparse, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tinystories", "openwebtext"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1_000_000_000)
    args = ap.parse_args()
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(args.out, exist_ok=True)

    def token_stream():
        ds = load_dataset("roneneldan/TinyStories" if args.dataset == "tinystories"
                          else "Skylion007/openwebtext", split="train",
                          streaming=True)
        n = 0
        for rec in ds:
            ids = enc.encode_ordinary(rec["text"])
            if ids:
                yield ids
                n += len(ids)
                if n >= args.max_tokens:
                    break

    # 单遍写 train.bin（尾部 5% 切给 val）
    tmp = os.path.join(args.out, "all.bin")
    all_ids = []
    with open(tmp, "wb") as f:
        buf = np.empty(1 << 20, dtype=np.uint16); k = 0
        for ids in token_stream():
            for t in ids:
                buf[k] = t; k += 1
                if k == len(buf):
                    buf.tofile(f); k = 0
            if k:
                buf[:k].tofile(f); k = 0
    # 重新 memmap 切分
    data = np.memmap(tmp, dtype=np.uint16, mode="r")
    cut = int(len(data) * 0.95)
    data[:cut].tofile(os.path.join(args.out, "train.bin"))
    data[cut:].tofile(os.path.join(args.out, "val.bin"))
    os.remove(tmp)
    print(f"done: {cut:,} train / {len(data)-cut:,} val tokens -> {args.out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n aa python -m pytest tests/test_data.py -q`
Expected: 1 passed

- [ ] **Step 5: 实际下载冒烟数据（TinyStories 200M tokens，约 10 分钟）**

Run:
```bash
export HF_ENDPOINT=https://hf-mirror.com  # 直连失败时
conda run -n aa python scripts/prepare_data.py --dataset tinystories --out data/ts200m --max-tokens 200_000_000
```
Expected: `done: ~190,000,000 train / ~10,000,000 val tokens -> data/ts200m`

- [ ] **Step 6: Commit**

```bash
git add aa/data.py scripts/prepare_data.py tests/test_data.py
git commit -m "feat: uint16 bin 数据管线 + TinyStories/OWT 下载脚本"
```

---

### Task 7: 训练循环 + 路由遥测

**Files:**
- Create: `aa/train.py`, `aa/telemetry.py`
- Test: `tests/test_telemetry.py`（训练循环以 200 步实跑代替单测）

**Interfaces:**
- Consumes: `AATransformer`、`BinDataset`
- Produces:
  - `python -m aa.train --config-json '<json>' --data data/ts200m --steps N --out runs/<name>`（config-json 覆盖 AAConfig 字段）
  - 产物：`runs/<name>/ckpt.pt`（model+config+step）、`runs/<name>/metrics.jsonl`（每 eval 间隔一行）
  - `telemetry.aggregate(path_usage, E) -> dict`：`{"utilization": [E] float, "min_frac": float, "entropy": float}`

- [ ] **Step 1: 写遥测失败测试**

```python
# tests/test_telemetry.py
import torch
from aa.telemetry import aggregate

def test_aggregate_utilization_and_collapse_detection():
    usage = torch.tensor([[900.0, 50.0, 40.0, 10.0],   # layer0: path3 濒死
                          [250.0, 250.0, 250.0, 250.0]])  # layer0: 均衡
    stats = aggregate(usage, E=4)
    assert abs(stats["per_layer"][0]["utilization"][3] - 10.0 / 1000.0) < 1e-6
    assert stats["per_layer"][0]["min_frac"] == 0.01
    assert stats["collapsed_layers"] == [0]   # min_frac < 0.05 判塌缩
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n aa python -m pytest tests/test_telemetry.py -q`
Expected: FAIL

- [ ] **Step 3: 实现遥测与训练循环**

```python
# aa/telemetry.py
import torch

COLLAPSE_FRAC = 0.05

def aggregate(path_usage: torch.Tensor, E: int) -> dict:
    """path_usage: [L, E] 累计选中计数。"""
    per_layer = []
    for l in range(path_usage.shape[0]):
        u = path_usage[l] / path_usage[l].sum().clamp_min(1.0)
        per_layer.append({"utilization": u.tolist(), "min_frac": float(u.min())})
    return {
        "per_layer": per_layer,
        "collapsed_layers": [l for l, s in enumerate(per_layer)
                             if s["min_frac"] < COLLAPSE_FRAC],
    }
```

```python
# aa/train.py
"""nanoGPT 配方精简版：AdamW(0.9,0.95)+cosine+bf16+grad-accum+checkpointing。"""
import argparse, dataclasses, json, math, os, time
import torch
from aa.config import AAConfig
from aa.model import AATransformer
from aa.data import BinDataset
from aa.telemetry import aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", default="{}")
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    overrides = json.loads(args.config_json)
    cfg = AAConfig(**overrides)
    device = "cuda"
    torch.manual_seed(42)
    model = AATransformer(cfg).to(device)
    model = torch.compile(model)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr,
        betas=(0.9, 0.95))
    train_ds = BinDataset(os.path.join(args.data, "train.bin"), cfg.block_size)
    val_ds = BinDataset(os.path.join(args.data, "val.bin"), cfg.block_size)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.micro_batch, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    it = iter(train_dl)

    def lr_at(s):
        if s < args.warmup:
            return s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    os.makedirs(args.out, exist_ok=True)
    metrics = open(os.path.join(args.out, "metrics.jsonl"), "a")
    t0, usage_acc = time.time(), None
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_dl); x, y = next(it)
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, aux = model(x, y)
            (loss / args.accum).backward()
            loss_acc += float(loss) / args.accum
            u = aux["path_usage"].detach().cpu()
            usage_acc = u if usage_acc is None else usage_acc + u
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vl = 0.0
                vit = iter(torch.utils.data.DataLoader(
                    val_ds, batch_size=args.micro_batch))
                for i, (x, y) in enumerate(vit):
                    if i >= args.eval_steps: break
                    x, y = x.to(device), y.to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, loss, _ = model(x, y)
                    vl += float(loss)
                vl /= args.eval_steps
            model.train()
            tel = aggregate(usage_acc, cfg.n_paths)
            row = {"step": step, "train_loss": loss_acc, "val_loss": vl,
                   "tok_per_s": step * args.micro_batch * args.accum *
                                cfg.block_size / (time.time() - t0),
                   **tel}
            metrics.write(json.dumps(row) + "\n"); metrics.flush()
            print(json.dumps(row), flush=True)
            usage_acc = None
    torch.save({"model": (model._orig_mod if hasattr(model, "_orig_mod")
                          else model).state_dict(),
                "config": dataclasses.asdict(cfg), "step": args.steps},
               os.path.join(args.out, "ckpt.pt"))
    print("saved ->", os.path.join(args.out, "ckpt.pt"))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行遥测测试通过**

Run: `conda run -n aa python -m pytest tests/test_telemetry.py -q`
Expected: 1 passed

- [ ] **Step 5: 训练冒烟实跑（dense 臂 200 步，验证管线端到端 + 吞吐）**

Run:
```bash
cd /home/lhyzyrx/activation-attention && conda run -n aa python -m aa.train \
  --config-json '{"n_layer":4,"n_embd":256,"n_head":4,"block_size":512}' \
  --data data/ts200m --steps 200 --micro-batch 16 --accum 2 \
  --eval-every 100 --out runs/pipeline_smoke
```
Expected: loss 从 ~10.8 降至 <7，`tok_per_s` ≥ 15k（8GB 显存不 OOM；若 OOM 降 micro-batch 至 8 并在结果中记录）

- [ ] **Step 6: Commit**

```bash
git add aa/train.py aa/telemetry.py tests/test_telemetry.py
git commit -m "feat: 训练循环 + 路由塌缩遥测"
```

---

### Task 8: 三臂冒烟实验与判读

**Files:**
- Create: `scripts/run_smoke.sh`
- Modify: 无（本任务产出数据与判读记录）

**Interfaces:**
- Consumes: Task 7 训练入口
- Produces: `runs/smoke_{dense,aa4,moa4}/metrics.jsonl` + `docs/findings.md` 冒烟判读记录

**判据（全过才准上扫描，对应 proposal §8）：**
1. 三臂 2000 步 val loss 均下降且无 NaN
2. aa4 臂每层 `min_frac ≥ 0.05`（router 无塌缩）
3. aa4 与 moa4 的吞吐 ≥ dense 的 60%（路由开销可控）
4. aa4 的 2000 步 val loss ≤ dense + 0.15（容量路径至少没拖后腿——只是"不崩"判据，正向增益留给扫描验证）

- [ ] **Step 1: 写冒烟脚本**

```bash
#!/bin/bash
# scripts/run_smoke.sh — 三臂冒烟：dense / AA(E=4) / MoA风格(E=4)
set -e
cd "$(dirname "$0")/.."
COMMON='{"n_layer":4,"n_embd":256,"n_head":4,"block_size":512'
conda run -n aa python -m aa.train --data data/ts200m --steps 2000 \
  --micro-batch 16 --accum 2 --eval-every 250 \
  --config-json "$COMMON}" --out runs/smoke_dense
conda run -n aa python -m aa.train --data data/ts200m --steps 2000 \
  --micro-batch 16 --accum 2 --eval-every 250 \
  --config-json "$COMMON,\"n_paths\":4,\"n_active\":1,\"shared_kv\":true,\"router_mode\":\"ste\"}" \
  --out runs/smoke_aa4
conda run -n aa python -m aa.train --data data/ts200m --steps 2000 \
  --micro-batch 16 --accum 2 --eval-every 250 \
  --config-json "$COMMON,\"n_paths\":4,\"n_active\":1,\"shared_kv\":false,\"router_mode\":\"ste\"}" \
  --out runs/smoke_moa4
```

- [ ] **Step 2: 运行三臂冒烟（约 30-60 分钟）**

Run: `bash scripts/run_smoke.sh 2>&1 | tee logs_smoke.txt`
Expected: 三组训练完成，无 OOM / NaN

- [ ] **Step 3: 判读并落盘 findings**

对每臂末 3 行 metrics 对比 val_loss 与 min_frac，写入 `docs/findings.md`：

```markdown
# 冒烟判读（日期）
| 臂 | 2k 步 val loss | min_frac（逐层） | tok/s | 判定 |
|---|---|---|---|---|
| dense | … | — | … | … |
| aa4 | … | … | … | … |
| moa4 | … | … | … | … |
四条判据逐条 PASS/FAIL，FAIL 时：塌缩→切 expert-choice 路由重跑 aa4；吞吐差→profile gather 路径；
loss 差→检查 STE 梯度（对照 soft 模式 500 步）。
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_smoke.sh docs/findings.md
git commit -m "exp: 三臂冒烟判读（判据 1-4 落盘）"
```

---

### Task 9: 容量扫描脚本（正式实验）

**Files:**
- Create: `scripts/run_sweep.sh`

**Interfaces:**
- Consumes: Task 7 训练入口、TinyStories 冒烟通过后的 OWT 数据（`data/owt1b/`，Task 6 脚本 `--dataset openwebtext --max-tokens 1_000_000_000` 生成）

**实验矩阵（对应 proposal §6 Stage A）：**
- 臂 1（dense 基线）：`n_paths=1`，1 个点
- 臂 2（AA 主曲线）：`n_paths ∈ {2,4,8,16}`，`n_active=1, shared_kv=true, router_mode=ste`，4 个点
- 臂 3（MoA 对照）：`n_paths=4, shared_kv=false`，1 个点
- 每点 2 seeds（`--seed 42/123`，train.py 需补 `--seed` 参数透传 `torch.manual_seed`）
- 统一：OWT 1B tokens，L=8, d=512（50M 锚点），micro-batch 16 × accum 4（effective 64），lr 6e-4 cosine，步数 = 1B / (64×1024) ≈ 15.3k 步

- [ ] **Step 1: 给 train.py 补 seed 参数**

`ap.add_argument("--seed", type=int, default=42)` 并将 `torch.manual_seed(42)` 改为 `torch.manual_seed(args.seed)`。

- [ ] **Step 2: 下载扫描数据**

Run: `conda run -n aa python scripts/prepare_data.py --dataset openwebtext --out data/owt1b --max-tokens 1_000_000_000`
Expected: ~950M train tokens（HF 直连失败时前置 `HF_ENDPOINT=https://hf-mirror.com`；下载约 1-2 小时）

- [ ] **Step 3: 写扫描脚本**

```bash
#!/bin/bash
# scripts/run_sweep.sh — RQ1 主曲线：12 个 run ≈ 36h（单卡顺序，可拆分）
set -e
cd "$(dirname "$0")/.."
CFG='{"n_layer":8,"n_embd":512,"n_head":8,"block_size":1024'
STEPS=15300
run() {  # $1=tag $2=extra-json $3=seed
  conda run -n aa python -m aa.train --data data/owt1b --steps $STEPS \
    --micro-batch 16 --accum 4 --eval-every 500 --seed $3 \
    --config-json "$CFG,$2}" --out runs/sweep_$1_s$3
}
# 臂 1: dense
for s in 42 123; do run dense "\"n_paths\":1" $s; done
# 臂 2: AA E∈{2,4,8,16}
for E in 2 4 8 16; do for s in 42 123; do
  run aa$E "\"n_paths\":$E,\"n_active\":1,\"shared_kv\":true,\"router_mode\":\"ste\"" $s
done; done
# 臂 3: MoA 风格 E=4
for s in 42 123; do run moa4 "\"n_paths\":4,\"n_active\":1,\"shared_kv\":false,\"router_mode\":\"ste\"" $s; done
```

- [ ] **Step 4: 试跑单点前 300 步验证配置（E=16 最大显存压力点）**

```bash
conda run -n aa python -m aa.train --data data/owt1b --steps 300 \
  --micro-batch 16 --accum 4 --eval-every 300 --seed 42 \
  --config-json '{"n_layer":8,"n_embd":512,"n_head":8,"block_size":1024,"n_paths":16,"n_active":1,"shared_kv":true}' \
  --out runs/sweep_probe_e16
```
Expected: 无 OOM（E=16 总参 ~110M，优化器状态 ~1.8GB，8GB 内）；tok/s ≥ 15k。OOM 则 micro-batch 降 8、accum 升 8，并同步更新 sweep 脚本保持 effective batch 64。

- [ ] **Step 5: Commit + 启动说明**

```bash
git add scripts/run_sweep.sh aa/train.py
git commit -m "exp: RQ1 容量扫描脚本 — 3 臂 12 run"
```

---

### Task 10: 扫描判读与 RQ1 结论模板

**Files:**
- Create: `docs/analysis_rq1.md`, `scripts/plot_sweep.py`

**Interfaces:**
- Consumes: `runs/sweep_*/metrics.jsonl`
- Produces: RQ1 回答（幂律拟合 + 塌缩检查 + 参数效率表），对应 proposal §8 成功标准

- [ ] **Step 1: 写汇总绘图脚本**

```python
# scripts/plot_sweep.py
"""汇总 sweep runs → 容量-性能表 + loss 曲线图。
用法: python scripts/plot_sweep.py --runs 'runs/sweep_*' --out docs/analysis_rq1.md
"""
import argparse, glob, json, os
import numpy as np

def final_val_loss(metrics_path, avg_last=3):
    rows = [json.loads(l) for l in open(metrics_path)]
    return float(np.mean([r["val_loss"] for r in rows[-avg_last:]]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/sweep_*")
    ap.add_argument("--out", default="docs/analysis_rq1.md")
    args = ap.parse_args()
    results = {}
    for d in sorted(glob.glob(args.runs)):
        name = os.path.basename(d)
        m = os.path.join(d, "metrics.jsonl")
        if os.path.exists(m):
            results[name] = final_val_loss(m)
    lines = ["# RQ1: Attention 容量扫描结果", "",
             "| run | final val loss (末3次均值) |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v:.4f} |")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(results)} runs)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 扫描完成后运行汇总并人工判读**

Run: `conda run -n aa python scripts/plot_sweep.py`
判读框架（写入 `docs/analysis_rq1.md`）：
1. **主问题**：AA 臂 val loss 随 E 增加是否持续下降？两点间降幅 < 0.01 视为平台信号
2. **对照**：AA(E) vs dense 的差距在哪个 E 达到 0.05+（匹配约束下即净容量增益）
3. **MoA 对照**：shared_kv 的成本（AA vs MoA 同 E）与收益（KV cache 不随 E 涨）
4. **塌缩审计**：所有 AA run 的 `collapsed_layers` 是否为空；非空则该点标注无效并加 expert-choice 重跑
5. **结论归类**（proposal §8 措辞）："reproducible quality improvement at matched active computation" 是否成立，或在哪个 E 段饱和

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_sweep.py docs/analysis_rq1.md
git commit -m "exp: RQ1 扫描判读落盘"
```

---

## 依赖关系

Task 1 → Task 2 → {Task 3, Task 6 可并行} → Task 4 → Task 5 → Task 7 → Task 8 → Task 9 → Task 10
关键路径上 Task 4（核心模块）和 Task 8（冒烟判据）是不可跳过的门：冒烟 FAIL 不进扫描。

## 后续阶段（不在本计划内）

- RQ1 扫描完成 → K=2 消融（半宽 path 匹配 FLOPs）→ 独立 KV 上限消融
- RQ2 adaptive depth（proposal §5）另立计划
- 云端租卡后的 100M+ 回补验证另立计划
