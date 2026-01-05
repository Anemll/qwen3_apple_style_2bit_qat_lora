#!/usr/bin/env python3
"""
Multi-chip TPU Training Script - Uses all TPU chips with data parallelism.

Usage on TPU v6-4 (4 chips):
    python scripts/train_v2_tpu_multi.py \
        --from-scratch \
        --cache-dir caches/openhermes_2.5_L128_K128_N50K \
        --config q4_r32 \
        --max-steps 3000 \
        --batch-size 8 \
        --accumulation-steps 4

This will use all 4 TPU chips with data parallelism:
- Effective batch = batch_size * accumulation_steps * num_chips
- For batch=8, accum=4, chips=4: effective batch = 128
"""

import argparse
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

# Check for TPU before anything else
def check_tpu():
    """Check if TPU is available and return number of chips."""
    try:
        import torch_xla
        # Try new API first, fall back to old API
        try:
            import torch_xla.runtime as xr
            num_devices = xr.world_size()
        except (ImportError, AttributeError):
            import torch_xla.core.xla_model as xm
            if hasattr(xm, 'xrt_world_size'):
                num_devices = xm.xrt_world_size()
            else:
                num_devices = 1
        return True, num_devices
    except ImportError:
        return False, 0
    except Exception as e:
        print(f"[WARN] TPU check failed: {e}")
        return False, 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-chip TPU Training')

    # Model/checkpoint args
    parser.add_argument('--v2-checkpoint', type=str, default=None)
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--cache-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='runs/v2_tpu_multi')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B')

    # Training args
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--hard-top1', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--constant-lr', action='store_true')
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Quantization config
    parser.add_argument('--config', type=str, default='q4_r32',
                        choices=['q2a4', 'q4a4', 'q4a4_r32', 'q4_r32', 'q2a2'])
    parser.add_argument('--group-size', type=int, default=32)

    # Wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='qwen3-qat')
    parser.add_argument('--wandb-run', type=str, default=None)

    # Multi-chip options
    parser.add_argument('--num-chips', type=int, default=None,
                        help='Number of TPU chips to use (default: all available)')

    return parser.parse_args()


def train_worker(index, args):
    """Training worker function - runs on each TPU chip."""
    import time
    import gc
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    # Import our modules
    from qat_lora import (
        AnemllQuantConfigV2,
        replace_linear_with_anemll_v2,
        freeze_Q_all,
    )
    from qat_lora.layer_qat import KDCacheDataset, compute_kd_loss_batch, collate_fn

    # Get device for this worker
    # Use new API (torch_xla.device) if available, else fall back
    try:
        device = torch_xla.device()
    except AttributeError:
        device = xm.xla_device()

    # Get world size - try new API first
    try:
        import torch_xla.runtime as xr
        world_size = xr.world_size()
        rank = xr.global_ordinal()
    except (ImportError, AttributeError):
        world_size = xm.xrt_world_size() if hasattr(xm, 'xrt_world_size') else 1
        rank = xm.get_ordinal()

    is_master = (rank == 0)

    def log(msg):
        if is_master:
            print(msg, flush=True)

    log(f"\n[TPU Multi-chip] Starting training on {world_size} chips")
    log(f"  Rank {rank}: device={device}")

    # Config presets
    CONFIG_PRESETS = {
        'q2a4': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 8},
        'q4a4': {'mlp_lut': 16, 'mlp_rank': 4, 'attn_lut': 16, 'attn_rank': 4},
        'q4a4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},
        'q4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},
        'q2a2': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 4, 'attn_rank': 32},
    }
    preset = CONFIG_PRESETS[args.config]

    # Training dtype (always BF16 on TPU)
    train_dtype = torch.bfloat16

    # Load model
    log("\n[1/4] Loading model...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=train_dtype,
        trust_remote_code=True,
    )

    # Replace with V2 layers
    v2_mlp_config = AnemllQuantConfigV2(
        lut_size=preset['mlp_lut'],
        scale_rank=preset['mlp_rank'],
        group_size=args.group_size,
        force_positive_scales=False,
        magnitude_activation='identity',
        use_ste_fp16=False,  # BF16 mode
    )
    v2_attn_config = AnemllQuantConfigV2(
        lut_size=preset['attn_lut'],
        scale_rank=preset['attn_rank'],
        group_size=args.group_size,
        force_positive_scales=False,
        magnitude_activation='identity',
        use_ste_fp16=False,
    )

    replace_linear_with_anemll_v2(
        model,
        mlp_config=v2_mlp_config,
        attn_config=v2_attn_config,
        quantize_attn=True,
        quantize_lm_head=False,
    )

    # Load checkpoint if provided
    if args.v2_checkpoint:
        state_dict = torch.load(args.v2_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        log(f"  Loaded checkpoint: {args.v2_checkpoint}")

    # Move to device
    model.to(device=device, dtype=train_dtype)
    log(f"  Model loaded ({time.time()-t0:.1f}s)")

    # Freeze Q
    freeze_Q_all(model)

    # Setup trainable parameters
    log("\n[2/4] Setting up training...")
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    log(f"  Trainable params: {n_params/1e6:.2f}M")

    # Optimizer
    from torch.optim import AdamW
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # LR Scheduler
    scheduler = None
    if not args.constant_lr or args.warmup_steps > 0:
        import math
        min_lr_ratio = 0.1

        def lr_lambda(step):
            if step < args.warmup_steps:
                return float(step) / float(max(1, args.warmup_steps))
            if not args.constant_lr:
                progress = float(step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)

    # Dataset (each worker loads full dataset, DataLoader handles sharding)
    log("\n[3/4] Loading dataset...")

    # KDCacheDataset is IterableDataset, but for distributed training we need
    # a map-style dataset. Use preload=True and wrap the cached examples.
    from torch.utils.data import Dataset as MapDataset

    class KDMapDataset(MapDataset):
        """Map-style wrapper for preloaded KDCacheDataset examples."""
        def __init__(self, cache_dir):
            # Load all examples using KDCacheDataset's preload
            iterable_ds = KDCacheDataset(cache_dir, shuffle=False, preload=True)
            self.examples = iterable_ds._cached_examples
            log(f"  Loaded {len(self.examples)} examples for distributed training")

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    dataset = KDMapDataset(args.cache_dir)

    # Sampler for distributed training
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Effective batch size
    effective_batch = args.batch_size * args.accumulation_steps * world_size
    total_tokens = effective_batch * 128 * args.max_steps  # Assume seq_len=128
    log(f"  Effective batch: {args.batch_size} x {args.accumulation_steps} x {world_size} = {effective_batch}")
    log(f"  Total tokens: {total_tokens/1e6:.1f}M")

    # WandB (master only)
    use_wandb = args.wandb and is_master
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or f"tpu_multi_{args.config}",
                config={
                    'model_id': args.model_id,
                    'config': args.config,
                    'batch_size': args.batch_size,
                    'effective_batch': effective_batch,
                    'max_steps': args.max_steps,
                    'lr': args.lr,
                    'num_chips': world_size,
                    'accumulation_steps': args.accumulation_steps,
                },
            )
        except ImportError:
            use_wandb = False

    # Training loop
    log("\n[4/4] Training...")
    log(f"  Steps: {args.max_steps}, Warmup: {args.warmup_steps}")

    model.train()
    optimizer.zero_grad()

    step = 0
    optimizer_step = 0
    total_loss = 0.0
    t_start = time.time()

    total_micro_steps = args.max_steps * args.accumulation_steps

    while step < total_micro_steps:
        sampler.set_epoch(step // len(dataloader))  # For proper shuffling

        for batch in dataloader:
            if step >= total_micro_steps:
                break

            # Forward pass with autocast
            with torch.amp.autocast(device_type='xla', dtype=torch.bfloat16):
                loss = compute_kd_loss_batch(
                    model, batch, device, args.temperature,
                    no_grad=False,
                    hard_top1_weight=args.hard_top1,
                    hard_full_weight=0.0,  # Disabled for TPU
                )

            # Scale for accumulation
            if args.accumulation_steps > 1:
                loss = loss / args.accumulation_steps

            # Backward
            loss.backward()

            # Optimizer step every accumulation_steps
            if (step + 1) % args.accumulation_steps == 0:
                # Gradient sync across chips
                xm.reduce_gradients(optimizer)

                # Optimizer step
                xm.optimizer_step(optimizer)

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad()
                optimizer_step += 1

                # Logging
                if optimizer_step % 20 == 0:
                    loss_val = loss.item() * args.accumulation_steps
                    elapsed = time.time() - t_start
                    eta = elapsed / optimizer_step * (args.max_steps - optimizer_step) if optimizer_step > 0 else 0
                    tok_per_sec = (optimizer_step * effective_batch * 128) / elapsed

                    if is_master:
                        print(f"  Step {optimizer_step}/{args.max_steps} | "
                              f"Loss: {loss_val:.4f} | "
                              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                              f"Tok/s: {tok_per_sec:.0f} | "
                              f"ETA: {eta/60:.1f}m", flush=True)

                    if use_wandb:
                        wandb.log({
                            'step': optimizer_step,
                            'loss': loss_val,
                            'lr': optimizer.param_groups[0]['lr'],
                            'tokens_per_sec': tok_per_sec,
                        })

                # Save checkpoint
                if args.save_steps > 0 and optimizer_step % args.save_steps == 0 and is_master:
                    os.makedirs(args.output_dir, exist_ok=True)
                    save_path = f"{args.output_dir}/checkpoint_step{optimizer_step}.pt"
                    # Only master saves (all chips have same weights after sync)
                    xm.save(model.state_dict(), save_path)
                    print(f"  Saved: {save_path}", flush=True)

            step += 1
            xm.mark_step()

    # Final save
    if is_master:
        elapsed = time.time() - t_start
        log(f"\n  Training complete: {elapsed/60:.1f} min")

        os.makedirs(args.output_dir, exist_ok=True)
        final_path = f"{args.output_dir}/checkpoint_fp32.pt"

        # Convert to FP32 for saving
        model_cpu = model.cpu().float()
        torch.save(model_cpu.state_dict(), final_path)
        log(f"  Saved: {final_path}")

        if use_wandb:
            wandb.finish()

    # Sync all workers before exit
    xm.rendezvous('training_done')


def main():
    args = parse_args()

    # Validate inputs
    if not args.v2_checkpoint and not args.from_scratch:
        raise ValueError("Must specify --v2-checkpoint or --from-scratch")
    if args.v2_checkpoint:
        assert os.path.exists(args.v2_checkpoint), f"Checkpoint not found: {args.v2_checkpoint}"
    assert os.path.exists(args.cache_dir), f"Cache dir not found: {args.cache_dir}"

    # Check TPU
    has_tpu, num_devices = check_tpu()
    if not has_tpu:
        print("ERROR: TPU not available. Use train_v2_simple.py for CPU/GPU.")
        sys.exit(1)

    num_chips = args.num_chips or num_devices
    print(f"[TPU Multi-chip] Launching on {num_chips} chips...")

    # Launch with xmp.spawn
    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(train_worker, args=(args,), nprocs=num_chips)


if __name__ == '__main__':
    main()
