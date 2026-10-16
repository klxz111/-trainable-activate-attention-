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
