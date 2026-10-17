#!/bin/bash
# scripts/run_sweep.sh — RQ1 主曲线：12 run。
# 统一配置（消除臂间执行差异对 matched-compute 对比的干扰）：
#   --no-compile          E16 大图编译期 RAM 爆（WSL 11GB 上限）
#   grad_checkpoint=true  E16 激活显存（8GB 卡）
#   mb8/accum8            mb16 下 logits+激活超 8GB
#   顺序遍历（默认）       9.5 亿窗口 randperm 需 7.6GB host 内存（已改为默认不洗牌）
#   effective batch = 8×8 = 64，与 Task 7 冒烟口径一致
set -e
cd "$(dirname "$0")/.."
CFG='{"n_layer":8,"n_embd":512,"n_head":8,"block_size":1024,"grad_checkpoint":true'
STEPS=15300
FLAGS="--no-compile --micro-batch 8 --accum 8"
run() {  # $1=tag $2=extra-json $3=seed
  conda run -n aa python -m aa.train --data data/owt1b --steps $STEPS \
    $FLAGS --eval-every 500 --seed $3 \
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
