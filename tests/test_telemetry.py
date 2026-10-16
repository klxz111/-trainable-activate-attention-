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
