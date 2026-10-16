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
