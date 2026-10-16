"""下载 → GPT-2 tokenize → uint16 train/val.bin。
用法: python scripts/prepare_data.py --dataset tinystories --out data/ts --max-tokens 500_000_000
"""
import argparse, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tinystories", "openwebtext"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1_000_000_000)
    args = ap.parse_args()
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(args.out, exist_ok=True)

    def token_stream():
        ds = load_dataset("roneneldan/TinyStories" if args.dataset == "tinystories"
                          else "Skylion007/openwebtext", split="train",
                          streaming=True)
        n = 0
        for rec in ds:
            ids = enc.encode_ordinary(rec["text"])
            if ids:
                yield ids
                n += len(ids)
                if n >= args.max_tokens:
                    break

    # 单遍写 train.bin（尾部 5% 切给 val）
    tmp = os.path.join(args.out, "all.bin")
    all_ids = []
    with open(tmp, "wb") as f:
        buf = np.empty(1 << 20, dtype=np.uint16); k = 0
        for ids in token_stream():
            for t in ids:
                buf[k] = t; k += 1
                if k == len(buf):
                    buf.tofile(f); k = 0
            if k:
                buf[:k].tofile(f); k = 0
    # 重新 memmap 切分
    data = np.memmap(tmp, dtype=np.uint16, mode="r")
    cut = int(len(data) * 0.95)
    data[:cut].tofile(os.path.join(args.out, "train.bin"))
    data[cut:].tofile(os.path.join(args.out, "val.bin"))
    os.remove(tmp)
    print(f"done: {cut:,} train / {len(data)-cut:,} val tokens -> {args.out}")

if __name__ == "__main__":
    main()
