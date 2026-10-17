# aa/train.py
"""nanoGPT 配方精简版：AdamW(0.9,0.95)+cosine+bf16+grad-accum+checkpointing。"""
import argparse, dataclasses, json, math, os, time
import torch
from aa.config import AAConfig
from aa.model import AATransformer
from aa.data import BinDataset
from aa.telemetry import aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", default="{}")
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=0, help="每 N 步保存断点 ckpt_last.pt（0=仅结束保存）")
    ap.add_argument("--resume", type=str, default="", help="从 ckpt_last.pt 断点恢复")
    args = ap.parse_args()

    overrides = json.loads(args.config_json)
    cfg = AAConfig(**overrides)
    device = "cuda"
    torch.manual_seed(args.seed)
    model = AATransformer(cfg).to(device)
    model = model if args.no_compile else torch.compile(model)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": 0.1},
              {"params": no_decay, "weight_decay": 0.0}]
    try:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95),
                                fused=True)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
    train_ds = BinDataset(os.path.join(args.data, "train.bin"), cfg.block_size)
    val_ds = BinDataset(os.path.join(args.data, "val.bin"), cfg.block_size)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.micro_batch,
        shuffle=(not args.no_shuffle),
        num_workers=2, pin_memory=True, drop_last=True)
    it = iter(train_dl)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        (model._orig_mod if hasattr(model, "_orig_mod")
         else model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed from {args.resume} @ step {start_step}", flush=True)
        print("note: DataLoader 从头遍历（数据顺序单遍设计，重读一小段可接受）",
              flush=True)

    def lr_at(s):
        if s < args.warmup:
            return s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    os.makedirs(args.out, exist_ok=True)
    metrics = open(os.path.join(args.out, "metrics.jsonl"), "a")
    t0, usage_acc = time.time(), None
    for step in range(start_step + 1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_dl); x, y = next(it)
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, aux = model(x, y)
            (loss / args.accum).backward()
            loss_acc += float(loss) / args.accum
            u = aux["path_usage"].detach().cpu()
            usage_acc = u if usage_acc is None else usage_acc + u
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vl = 0.0
                vit = iter(torch.utils.data.DataLoader(
                    val_ds, batch_size=args.micro_batch))
                for i, (x, y) in enumerate(vit):
                    if i >= args.eval_steps: break
                    x, y = x.to(device), y.to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, loss, _ = model(x, y)
                    vl += float(loss)
                vl /= args.eval_steps
            model.train()
            tel = aggregate(usage_acc, cfg.n_paths)
            row = {"step": step, "train_loss": loss_acc, "val_loss": vl,
                   "tok_per_s": step * args.micro_batch * args.accum *
                                cfg.block_size / (time.time() - t0),
                   **tel}
            metrics.write(json.dumps(row) + "\n"); metrics.flush()
            print(json.dumps(row), flush=True)
            usage_acc = None
        if args.save_every > 0 and step % args.save_every == 0:
            torch.save({
                "model": (model._orig_mod if hasattr(model, "_orig_mod")
                          else model).state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "config": dataclasses.asdict(cfg),
            }, os.path.join(args.out, "ckpt_last.pt"))
    torch.save({"model": (model._orig_mod if hasattr(model, "_orig_mod")
                          else model).state_dict(),
                "config": dataclasses.asdict(cfg), "step": args.steps},
               os.path.join(args.out, "ckpt.pt"))
    print("saved ->", os.path.join(args.out, "ckpt.pt"))

if __name__ == "__main__":
    main()
