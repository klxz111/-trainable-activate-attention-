# aa/telemetry.py
import torch

COLLAPSE_FRAC = 0.05

def aggregate(path_usage: torch.Tensor, E: int) -> dict:
    """path_usage: [L, E] 累计选中计数。"""
    # 注：brief 原版直接用输入 dtype 归一化；输入为 float32 时
    # 10.0/1000.0 = 0.009999999... 与测试断言的 float64 字面量 0.01 严格
    # 相等失败。此处转 double 后归一化，数学语义不变，仅提升精度。
    per_layer = []
    for l in range(path_usage.shape[0]):
        counts = path_usage[l].double()
        u = counts / counts.sum().clamp_min(1.0)
        per_layer.append({"utilization": u.tolist(), "min_frac": float(u.min())})
    return {
        "per_layer": per_layer,
        "collapsed_layers": [l for l, s in enumerate(per_layer)
                             if s["min_frac"] < COLLAPSE_FRAC],
    }
