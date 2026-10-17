import json, os, subprocess, sys
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="resume 验证需要 GPU")

def _run(args, out):
    r = subprocess.run(
        [sys.executable, "-m", "aa.train"] + args + ["--out", out],
        capture_output=True, text=True, cwd=os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout

def _final_val(out):
    rows = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    return rows[-1]["val_loss"]

def test_resume_matches_continuous(tmp_path):
    cfg = ("--config-json", '{"n_layer":2,"n_embd":128,"n_head":4,"n_paths":2,'
           '"vocab_size":50257,"block_size":64,"grad_checkpoint":true}')
    # vocab_size=50257: data/ts200m 为 GPT-2 分词（最大 id 50255），256 会越界 assert
    # 分两段：10 步 + 保存，resume 到 20 步
    a = tmp_path / "a"; b = tmp_path / "b"
    _run(["--data", "data/ts200m", "--steps", "10", "--micro-batch", "4",
          "--accum", "2", "--eval-every", "5", "--save-every", "5",
          "--seed", "42", *cfg], str(a))
    _run(["--data", "data/ts200m", "--steps", "20", "--micro-batch", "4",
          "--accum", "2", "--eval-every", "5", "--resume",
          os.path.join(str(a), "ckpt_last.pt"), "--seed", "42", *cfg], str(b))
    # 连续 20 步
    c = tmp_path / "c"
    _run(["--data", "data/ts200m", "--steps", "20", "--micro-batch", "4",
          "--accum", "2", "--eval-every", "5", "--seed", "42", *cfg], str(c))
    va, vc = _final_val(str(b)), _final_val(str(c))
    assert abs(va - vc) < 0.05, f"resume {va} vs continuous {vc}"
    rows_b = [json.loads(l) for l in open(os.path.join(str(b), "metrics.jsonl"))]
    assert rows_b[-1]["step"] == 20
