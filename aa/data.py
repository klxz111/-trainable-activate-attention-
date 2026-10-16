# aa/data.py
import numpy as np
import torch


class BinDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, block_size: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        j = np.random.randint(len(self)) if i == -1 else i
        chunk = self.data[j:j + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
