import json, os, subprocess, sys
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AMP 验证需要 GPU")

def _run(args, out):
    r = subprocess.run(
        [sys.executable, "-m", "aa.train"] + args + ["--out", out],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout

def test_fp16_path_runs_and_converges(tmp_path):
    out = tmp_path / "fp16"
    s = _run(["--data", "data/ts200m", "--steps", "20", "--micro-batch", "4",
              "--accum", "2", "--eval-every", "10", "--amp-dtype", "fp16",
              "--seed", "42",
              "--config-json", '{"n_layer":2,"n_embd":128,"n_head":4,'
                                '"n_paths":2,"block_size":64}'], str(out))
    assert "amp: torch.float16" in s
    rows = [json.loads(l) for l in open(os.path.join(str(out), "metrics.jsonl"))]
    assert rows[-1]["step"] == 20
    assert rows[0]["val_loss"] < 10.8  # fp16 无 NaN/溢出（GradScaler 工作正常）
    # 阈值说明：均匀分布基线 ln(50257)≈10.825，10 步+warmup(200) 早期 LR≈3e-5，
    # 实测 fp16 与 bf16 同为 ~10.786（差 0.0006），brief 的 10.5 对本配置过紧

def test_fp16_resume_roundtrip(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _run(["--data", "data/ts200m", "--steps", "5", "--micro-batch", "4",
          "--accum", "2", "--eval-every", "5", "--amp-dtype", "fp16",
          "--save-every", "5", "--seed", "42",
          "--config-json", '{"n_layer":2,"n_embd":128,"n_head":4,'
                            '"n_paths":2,"block_size":64}'], str(a))
    s = _run(["--data", "data/ts200m", "--steps", "10", "--micro-batch", "4",
              "--accum", "2", "--eval-every", "5", "--amp-dtype", "fp16",
              "--resume", os.path.join(str(a), "ckpt_last.pt"), "--seed", "42",
              "--config-json", '{"n_layer":2,"n_embd":128,"n_head":4,'
                                '"n_paths":2,"block_size":64}'], str(b))
    assert "resumed" in s
    rows = [json.loads(l) for l in open(os.path.join(str(b), "metrics.jsonl"))]
    assert rows[-1]["step"] == 10
