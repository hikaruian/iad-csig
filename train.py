#!/usr/bin/env python3
"""Train unified multi-class INP-Former. Single GPU or 2×T4 via torchrun.

    torchrun --standalone --nproc_per_node=2 train.py --train-root ... --amp
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.data import CSIGImageDataset, build_transform
from src.dist_utils import (
    barrier,
    cleanup,
    init_distributed,
    is_main,
    make_autocast,
    make_scaler,
    reduce_mean,
    setup_seed,
    unwrap,
)
from src.encoder import prefetch_encoder_weights
from src.losses import total_loss
from src.model import build_model
from src.optim import StableAdamW, WarmCosineScheduler


def parse_args():
    p = argparse.ArgumentParser(description="CSIG SOTA — INP-Former multi-class trainer (DDP)")
    p.add_argument("--train-root", type=str, required=True, help="CSIG/Train directory")
    p.add_argument("--save-dir", type=str, default="runs/inpformer")
    p.add_argument("--encoder", type=str, default="dinov2reg_vit_base_14",
                   choices=["dinov2reg_vit_small_14", "dinov2reg_vit_base_14", "dinov2reg_vit_large_14"])
    p.add_argument("--encoder-source", type=str, default="auto", choices=["auto", "hub", "timm"])
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--inp-num", type=int, default=6)
    p.add_argument("--decoder-depth", type=int, default=8)
    p.add_argument("--bottleneck-drop", type=float, default=0.0)
    p.add_argument("--residual", action="store_true", help="Enable INP-Former++ residual learning")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-GPU batch. 2×T4 16G + ViT-B/448/AMP: 4 is the safe default.")
    p.add_argument("--grad-accum", type=int, default=2,
                   help="Micro-steps. 2 GPU × 4 × accum 2 = global 16 (paper default).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--gather-weight", type=float, default=0.2)
    p.add_argument("--soft-y", type=float, default=3.0)
    p.add_argument("--num-workers", type=int, default=2,
                   help="Workers PER rank. 2 ranks × 2 = 4 readers total.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--amp", dest="amp", action="store_true", default=True,
                   help="fp16 AMP (default ON — T4 has no bf16).")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Checkpoint decoder blocks (use if 16G still OOMs).")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--print-freq", type=int, default=20)
    return p.parse_args()


def log(msg: str, info, save_dir: Path | None = None):
    if not is_main(info):
        return
    print(msg, flush=True)
    if save_dir is not None:
        with open(save_dir / "train.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def main():
    args = parse_args()
    # Must happen before NCCL comes up — first DINOv2 download can take >10 min.
    prefetch_encoder_weights(args.encoder, args.encoder_source, stamp_dir=str(Path(args.save_dir) / "_prefetch"))
    info = init_distributed()
    setup_seed(args.seed, rank=info.rank)
    device = info.device

    save_dir = Path(args.save_dir)
    if is_main(info):
        save_dir.mkdir(parents=True, exist_ok=True)
        payload = {**vars(args), "world_size": info.world_size, "global_batch":
                   args.batch_size * info.world_size * max(1, args.grad_accum)}
        (save_dir / "args.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    barrier()

    if args.image_size % 14 != 0:
        raise ValueError(f"--image-size must be divisible by 14 (DINOv2 patch), got {args.image_size}")
    if args.encoder == "dinov2reg_vit_large_14" and args.batch_size > 2 and is_main(info):
        print("[warn] ViT-L on 16GB T4 often OOMs above --batch-size 1. Try --grad-checkpoint --batch-size 1.")

    log(f"[dist]   rank={info.rank}/{info.world_size}  device={device}  amp={args.amp}", info)
    log(f"[data]   {args.train_root}", info)

    ds = CSIGImageDataset(
        args.train_root,
        transform=build_transform(args.image_size, is_train=True),
        image_size=args.image_size,
    )
    sampler = DistributedSampler(ds, num_replicas=info.world_size, rank=info.rank, shuffle=True, drop_last=True) \
        if info.distributed else None
    loader_kw = dict(
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2
    loader = DataLoader(ds, **loader_kw)
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError(
            "Empty DataLoader. Reduce --batch-size or nproc_per_node "
            f"(got {len(ds)} images, batch={args.batch_size}, world={info.world_size})."
        )
    accum = max(1, args.grad_accum)
    # include leftover micro-batch so the cosine schedule matches real optimizer steps
    opt_steps_per_epoch = (steps_per_epoch + accum - 1) // accum
    global_bs = args.batch_size * info.world_size * max(1, args.grad_accum)
    log(
        f"[data]   {len(ds)} images / {len(ds.classes)} classes / "
        f"{steps_per_epoch} micro-steps / global_batch={global_bs}",
        info,
    )

    model = build_model(
        encoder_name=args.encoder,
        inp_num=args.inp_num,
        decoder_depth=args.decoder_depth,
        bottleneck_drop=args.bottleneck_drop,
        residual=args.residual,
        encoder_source=args.encoder_source,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)

    if info.distributed:
        model = DDP(
            model,
            device_ids=[info.local_rank],
            output_device=info.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    raw = unwrap(model)
    trainable = [p for p in raw.trainable_parameters()]
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in raw.parameters())
    log(f"[model]  trainable {n_train/1e6:.2f}M / total {n_all/1e6:.2f}M  residual={args.residual}", info)

    optimizer = StableAdamW(
        [{"params": trainable}],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        amsgrad=True,
        eps=1e-10,
    )
    total_iters = args.epochs * opt_steps_per_epoch
    scheduler = WarmCosineScheduler(
        optimizer,
        base_value=args.lr,
        final_value=args.min_lr,
        total_iters=total_iters,
        warmup_iters=min(100, max(10, opt_steps_per_epoch)),
    )
    scaler = make_scaler(args.amp)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        from src.checkpoint import torch_load

        ckpt = torch_load(args.resume, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt
        from src.dist_utils import strip_module_prefix

        missing, unexpected = raw.load_state_dict(strip_module_prefix(state), strict=False)
        if is_main(info) and (missing or unexpected):
            print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            optimizer.weight_decay = args.weight_decay
        if "scheduler" in ckpt and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler"])
            scheduler.final_value = args.min_lr
        if "scaler" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        start_epoch = int(ckpt.get("epoch", 0))
        best_loss = float(ckpt.get("best_loss", best_loss))
        log(f"[resume] {args.resume} @ epoch {start_epoch}", info)

    accum = max(1, args.grad_accum)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        raw.encoder.train(False)  # keep DINOv2 frozen + eval
        if sampler is not None:
            sampler.set_epoch(epoch)

        running = []
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        iterator = loader
        if is_main(info):
            iterator = tqdm(loader, ncols=110, desc=f"epoch {epoch+1}/{args.epochs}")

        for step, (imgs, _) in enumerate(iterator):
            imgs = imgs.to(device, non_blocking=True)
            do_step = ((step + 1) % accum == 0) or ((step + 1) == steps_per_epoch)
            # Skip DDP allreduce on accumulation micro-steps (correctness unchanged, much faster).
            sync_ctx = (
                model.no_sync()
                if (info.distributed and not do_step)
                else nullcontext()
            )
            with sync_ctx:
                with make_autocast(args.amp):
                    en, de, g_loss = model(imgs)
                    loss = total_loss(en, de, g_loss, gather_weight=args.gather_weight, y=args.soft_y)
                    loss = loss / accum

                # All ranks must take the same control-flow path or DDP will deadlock.
                finite = torch.tensor(1 if torch.isfinite(loss) else 0, device=device)
                if info.distributed:
                    torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
                if int(finite.item()) == 0:
                    log(f"[warn] non-finite loss at epoch {epoch+1} step {step}, skip", info, save_dir)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
            if do_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            reduced = reduce_mean(loss.detach() * accum)
            running.append(float(reduced.cpu()))
            if is_main(info) and hasattr(iterator, "set_postfix") and (step % args.print_freq == 0):
                iterator.set_postfix(loss=f"{running[-1]:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        mean_loss = float(np.mean(running)) if running else float("inf")
        msg = (
            f"epoch {epoch+1}/{args.epochs}  loss={mean_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  time={time.time()-t0:.1f}s"
        )
        log(msg, info, save_dir)

        if is_main(info):
            ckpt = {
                "epoch": epoch + 1,
                "model": raw.state_dict(),
                "trainable": raw.trainable_state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
                "loss": mean_loss,
                "best_loss": min(best_loss, mean_loss),
            }
            #torch.save(ckpt, save_dir / "last.pth")
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(ckpt, save_dir / "best.pth")
        barrier()

    if is_main(info):
        torch.save({"model": raw.state_dict(), "args": vars(args)}, save_dir / "model.pth")
        log(f"[done] saved {save_dir / 'model.pth'}", info, save_dir)
    barrier()


if __name__ == "__main__":
    # Dual-T4 boxes without NVLink sometimes hang in NCCL P2P. Safe default.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    try:
        main()
    finally:
        cleanup()
