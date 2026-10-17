"""Kaggle kernel: Activation Attention 单实验点全自动运行。
由 make_kernel.sh 从模板生成，RUN_TAG/SEED/CFG 在生成时替换占位符。
产物（metrics.jsonl + ckpt*）落 /kaggle/working/ 自动保存为 kernel output。
"""
import os
import subprocess

os.chdir("/kaggle/working")
REPO = "https://github.com/klxz111/-trainable-activate-attention-.git"
if not os.path.exists("aa"):
    subprocess.run(["git", "clone", REPO, "aa"], check=True)
os.chdir("aa")
subprocess.run("pip install -q tiktoken datasets", shell=True, check=True)

if not os.path.exists("data/owt1b/train.bin"):
    subprocess.run(["python", "scripts/prepare_data.py", "--dataset", "openwebtext",
                    "--out", "data/owt1b", "--max-tokens", "1000000000"], check=True)

RUN_TAG = "{{RUN_TAG}}"
SEED = "{{SEED}}"
EXTRA = "{{EXTRA}}"          # 例如 "n_paths":8,... 
CKPT = "true" if RUN_TAG.startswith("aa16") else "false"
STEPS = os.environ.get("STEPS", "15300")

cmd = (
    "python -m aa.train --compile --data data/owt1b "
    f"--steps {STEPS} --micro-batch 16 --accum 4 --eval-every 500 "
    f"--save-every 500 --seed {SEED} "
    f"--config-json '{{\"n_layer\":8,\"n_embd\":512,\"n_head\":8,\"block_size\":1024,"
    f"\"grad_checkpoint\":{CKPT},{EXTRA}}}' "
    f"--out /kaggle/working/run_out"
)
print("CMD:", cmd, flush=True)
subprocess.run(cmd, shell=True, check=True)
print("TRAIN DONE", flush=True)
