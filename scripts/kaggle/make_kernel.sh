#!/bin/bash
# 用法: bash make_kernel.sh <RUN_TAG> <SEED>
# 例:   bash make_kernel.sh aa8_s42 42
# 生成 kaggle/<tag>/ 目录（run.py + kernel-metadata.json），token 配好后:
#   cd kaggle/<tag> && kaggle kernels push
set -e
cd "$(dirname "$0")"
TAG="$1"; SEED="$2"
case "$TAG" in
  aa8_*)   EXTRA='"n_paths":8,"n_active":1,"shared_kv":true,"router_mode":"ste"' ;;
  aa16_*)  EXTRA='"n_paths":16,"n_active":1,"shared_kv":true,"router_mode":"ste"' ;;
  moa4_*)  EXTRA='"n_paths":4,"n_active":1,"shared_kv":false,"router_mode":"ste"' ;;
  *) echo "unknown tag"; exit 1 ;;
esac
D="$TAG"
SLUG=$(echo "$TAG" | tr _ -)   # Kaggle slug 不允许下划线
mkdir -p "$D"
sed -e "s/{{RUN_TAG}}/$TAG/" -e "s/{{SEED}}/$SEED/" -e "s|{{EXTRA}}|$EXTRA|" \
    run_template.py > "$D/run.py"
cat > "$D/kernel-metadata.json" << JSON
{
  "id": "hangyulv/aa-$SLUG",
  "title": "aa $SLUG",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "enable_gpu": true,
  "enable_internet": true,
  "is_private": true
}
JSON
echo "generated: $D/ (kaggle kernels push 提交)"
