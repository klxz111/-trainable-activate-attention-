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
