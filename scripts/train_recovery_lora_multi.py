#!/usr/bin/env python3
"""
Multi-chip TPU Recovery LoRA Training Script

Uses all TPU chips with data parallelism for recovery LoRA training.

Usage on TPU v6e-4 (4 chips):
    python scripts/train_recovery_lora_multi.py \
        --model Qwen/Qwen3-0.6B \
        --checkpoint runs/v2_q4/best.pt \
        --kd-cache-dir caches/wikitext103_L1024_K128 \
        --recovery-r 8 \
        --max-steps 1000 \
        --batch-size 4 \
        --accumulation-steps 8

Effective batch = batch_size * accumulation_steps * num_chips
For batch=4, accum=8, chips=4: effective batch = 128

See docs/TPU.md for debugging tips.
"""

import argparse
import os
import sys
import importlib.util
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))


def check_tpu():
    """Check if torch_xla is installed without importing it."""
    return importlib.util.find_spec("torch_xla") is not None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-chip TPU Recovery LoRA Training')

    # Model/checkpoint args
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-0.6B',
                        help='Base model name or path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='V2 QAT checkpoint to load')
    parser.add_argument('--output-dir', type=str, default='runs/recovery_lora_multi',
                        help='Output directory for checkpoints')

    # KD cache args
    parser.add_argument('--kd-cache-dir', type=str, required=True,
                        help='Directory containing KD cache shards')
    parser.add_argument('--seq-len', type=int, default=1024,
                        help='Sequence length (must match cache)')

    # LoRA args
    parser.add_argument('--recovery-r', type=int, default=8,
                        help='LoRA rank for recovery adapters')
    parser.add_argument('--lora-alpha', type=float, default=None,
                        help='LoRA alpha (default: 2*r)')
    parser.add_argument('--mlp-only', action='store_true',
                        help='Apply LoRA to MLP layers only (skip attention)')
    parser.add_argument('--freeze-mags', action='store_true',
                        help='Freeze rank_magnitude (snap to FP16 values). Train only A, B, and LoRA.')
    parser.add_argument('--freeze-mags-mlp', action='store_true',
                        help='Freeze rank_magnitude for MLP layers only')

    # Training args
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Per-chip batch size')
    parser.add_argument('--accumulation-steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum optimizer steps')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=100,
                        help='LR warmup steps')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1,
                        help='Min LR ratio for cosine annealing')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0,
                        help='Gradient clipping max norm (0=disable)')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'bf16'],
                        help='Master weight dtype: fp32 (stable) or bf16 (faster, less memory)')

    # KD loss args
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='KD temperature')
    parser.add_argument('--hard-top1', type=float, default=0.2,
                        help='Hard label top-1 weight')
    parser.add_argument('--hard-top1-end', type=float, default=None,
                        help='Hard label top-1 at end (for annealing)')

    # Logging/saving
    parser.add_argument('--log-every', type=int, default=10,
                        help='Log every N optimizer steps')
    parser.add_argument('--save-steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--keep-checkpoints', type=int, default=3,
                        help='Keep only last N checkpoints (0=keep all)')

    # Wandb
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='qwen3-recovery-lora',
                        help='Wandb project name')
    parser.add_argument('--wandb-run', type=str, default=None,
                        help='Wandb run name')

    # Multi-chip options
    parser.add_argument('--num-chips', type=int, default=None,
                        help='Number of TPU chips (default: all available)')

    # Google Drive upload
    parser.add_argument('--upload', action='store_true',
                        help='Auto-upload run to Google Drive after successful training (uses gdrive_sync.py)')
    parser.add_argument('--upload-exclude', type=str, action='append', default=None,
                        help='Glob patterns to exclude from upload (default: *checkpoint*). Can be used multiple times.')

    return parser.parse_args()


def train_worker(index, args):
    """Training worker function - runs on each TPU chip."""
    import time
    import gc
    import traceback
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    from torch.utils.data import DataLoader, IterableDataset
    from transformers import AutoModelForCausalLM

    # Get device for this worker
    try:
        device = torch_xla.device()
    except AttributeError:
        device = xm.xla_device()

    # Get world size
    try:
        import torch_xla.runtime as xr
        world_size = xr.world_size()
        rank = xr.global_ordinal()
    except (ImportError, AttributeError):
        world_size = xm.xrt_world_size() if hasattr(xm, 'xrt_world_size') else 1
        rank = xm.get_ordinal()

    is_master = (rank == 0)

    def log(msg, flush=True):
        if is_master:
            print(msg, flush=flush)

    def log_all(msg):
        print(f"[Rank {rank}] {msg}", flush=True)

    log_all(f"Worker started: device={device}, world_size={world_size}")

    try:
        _train_worker_impl(index, args, device, rank, world_size, is_master, log, log_all)
    except Exception as e:
        log_all(f"FATAL ERROR: {e}")
        traceback.print_exc()
        raise


def _train_worker_impl(index, args, device, rank, world_size, is_master, log, log_all):
    """Actual training implementation."""
    import time
    import gc
    import math
    import json
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    from torch.utils.data import DataLoader, IterableDataset
    from transformers import AutoModelForCausalLM
    from collections import deque

    from qat_lora import (
        AnemllQuantConfigV2,
        replace_linear_with_anemll_v2,
        freeze_Q_all,
    )
    from qat_lora.ane_qat_linear_v2 import (
        enable_recovery_lora_all,
        freeze_for_recovery_training,
        get_recovery_lora_params,
    )
    from qat_lora.layer_qat import compute_kd_loss_batch, collate_fn

    compute_dtype = torch.bfloat16
    # Master weight dtype: fp32 (stable) or bf16 (faster, ~2x less memory)
    train_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32

    log(f"\n[Recovery LoRA Multi-chip] Starting on {world_size} TPU chips")
    log(f"  Master dtype: {args.dtype}, Compute dtype: bf16")
    log(f"  Rank {rank}: device={device}")

    # =========================================================================
    # 1. Load model (rank 0 first, then others from cache)
    # =========================================================================
    log("\n[1/5] Loading model...")
    t0 = time.time()

    if is_master:
        log_all("Rank 0 loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=train_dtype,
            trust_remote_code=True,
        )
        log_all("Rank 0 model loaded")

    xm.rendezvous("model_download")

    if not is_master:
        log_all("Loading model from cache...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=train_dtype,
            trust_remote_code=True,
        )
        log_all("Model loaded")

    log(f"  Model loaded ({time.time()-t0:.1f}s)")

    # =========================================================================
    # 2. Replace with V2 layers and load checkpoint
    # =========================================================================
    log("\n[2/5] Setting up V2 quantized layers...")

    # Get config from checkpoint
    ckpt_dir = Path(args.checkpoint).parent
    config_path = ckpt_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            ckpt_config = json.load(f)
        mlp_lut = ckpt_config.get('mlp_lut_size', 16)
        mlp_rank = ckpt_config.get('mlp_scale_rank', 32)
        attn_lut = ckpt_config.get('attn_lut_size', 16)
        attn_rank = ckpt_config.get('attn_scale_rank', 32)
        group_size = ckpt_config.get('group_size', 32)
    else:
        # Defaults
        mlp_lut, mlp_rank = 16, 32
        attn_lut, attn_rank = 16, 32
        group_size = 32

    v2_mlp_config = AnemllQuantConfigV2(
        lut_size=mlp_lut, scale_rank=mlp_rank, group_size=group_size,
        force_positive_scales=False, magnitude_activation='identity', use_ste_fp16=True,
    )
    v2_attn_config = AnemllQuantConfigV2(
        lut_size=attn_lut, scale_rank=attn_rank, group_size=group_size,
        force_positive_scales=False, magnitude_activation='identity', use_ste_fp16=True,
    )

    # Rank 0 does setup, saves for others
    os.makedirs(args.output_dir, exist_ok=True)
    v2_cache_path = os.path.join(args.output_dir, ".v2_lora_state_temp.pt")

    if is_master:
        log_all("Rank 0: Creating V2 structure and loading checkpoint...")
        replace_linear_with_anemll_v2(
            model, mlp_config=v2_mlp_config, attn_config=v2_attn_config,
            quantize_attn=True, quantize_lm_head=False, skip_init=True,
        )

        # Load QAT checkpoint
        state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        log(f"  Loaded checkpoint: {args.checkpoint}")

        # Freeze Q
        freeze_Q_all(model)

        # Enable recovery LoRA (alpha=r gives scaling=1.0, standard convention)
        lora_alpha = args.lora_alpha if args.lora_alpha else float(args.recovery_r)
        enable_recovery_lora_all(
            model, r=args.recovery_r, alpha=lora_alpha,
            mlp_only=args.mlp_only, skip_k=True,
        )
        freeze_for_recovery_training(model)

        # Freeze mags if requested (snap to FP16 + freeze)
        if args.freeze_mags or args.freeze_mags_mlp:
            mags_frozen = 0
            mlp_patterns = ('gate_proj', 'up_proj', 'down_proj')
            for name, m in model.named_modules():
                if hasattr(m, 'rank_magnitude') and m.rank_magnitude is not None:
                    is_mlp = any(p in name for p in mlp_patterns)
                    if args.freeze_mags or (args.freeze_mags_mlp and is_mlp):
                        with torch.no_grad():
                            m.rank_magnitude.data = m.rank_magnitude.data.half().float()
                        m.rank_magnitude.requires_grad = False
                        mags_frozen += 1
            log(f"  Snapped & frozen {mags_frozen} rank_magnitude tensors")

        # Save for other ranks
        torch.save(model.state_dict(), v2_cache_path)
        log_all("Rank 0: V2+LoRA state saved for other ranks")

    xm.rendezvous("v2_lora_ready")

    if not is_master:
        log_all(f"Rank {rank}: Creating V2+LoRA structure...")
        replace_linear_with_anemll_v2(
            model, mlp_config=v2_mlp_config, attn_config=v2_attn_config,
            quantize_attn=True, quantize_lm_head=False, skip_init=True,
        )
        freeze_Q_all(model)

        lora_alpha = args.lora_alpha if args.lora_alpha else float(args.recovery_r)
        enable_recovery_lora_all(
            model, r=args.recovery_r, alpha=lora_alpha,
            mlp_only=args.mlp_only, skip_k=True,
        )
        freeze_for_recovery_training(model)

        # Freeze mags if requested (snap to FP16 + freeze)
        if args.freeze_mags or args.freeze_mags_mlp:
            mlp_patterns = ('gate_proj', 'up_proj', 'down_proj')
            for name, m in model.named_modules():
                if hasattr(m, 'rank_magnitude') and m.rank_magnitude is not None:
                    is_mlp = any(p in name for p in mlp_patterns)
                    if args.freeze_mags or (args.freeze_mags_mlp and is_mlp):
                        m.rank_magnitude.requires_grad = False
            # Values already snapped by rank 0, will be loaded below

        # Load weights from rank 0
        state_dict = torch.load(v2_cache_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        log_all(f"Rank {rank}: Loaded V2+LoRA weights from rank 0")

    # Move to device
    model.to(device=device, dtype=train_dtype)
    gc.collect()

    # Count params
    lora_params = get_recovery_lora_params(model)
    log(f"  LoRA params: {lora_params/1e6:.2f}M (r={args.recovery_r})")

    # =========================================================================
    # 3. Setup optimizer and scheduler
    # =========================================================================
    log("\n[3/5] Setting up optimizer...")

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    log(f"  Trainable params: {n_params/1e6:.2f}M")

    from torch.optim import AdamW
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduler with warmup + cosine decay
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    # =========================================================================
    # 4. Setup sharded dataset
    # =========================================================================
    log("\n[4/5] Loading KD cache (sharded)...")

    class ShardedKDDataset(IterableDataset):
        """Streaming dataset that shards cache files across workers."""
        def __init__(self, cache_dir, rank, world_size, shuffle=True):
            self.cache_dir = Path(cache_dir)
            self.all_files = sorted(self.cache_dir.glob('shard_*.pt'))
            if not self.all_files:
                self.all_files = sorted(self.cache_dir.glob('*.pt'))
            # Shard files across workers
            self.files = [f for i, f in enumerate(self.all_files) if i % world_size == rank]
            self.shuffle = shuffle
            self.rank = rank

        def _parse_shard(self, data):
            """Parse shard into list of examples."""
            ids = data.get('input_ids', data.get('ids'))
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)

            # Handle different key names
            topk_idx = data.get('topk_idx', data.get('topk_indices'))
            topk_logits = data.get('topk_logits', data.get('topk_probs'))

            if topk_idx.dim() == 2:
                topk_idx = topk_idx.unsqueeze(0)
            if topk_logits.dim() == 2:
                topk_logits = topk_logits.unsqueeze(0)

            attn_mask = data.get('attention_mask')
            if attn_mask is None:
                attn_mask = torch.ones_like(ids)
            elif attn_mask.dim() == 1:
                attn_mask = attn_mask.unsqueeze(0)

            examples = []
            for i in range(ids.shape[0]):
                examples.append({
                    'input_ids': ids[i],
                    'attention_mask': attn_mask[i],
                    'topk_idx': topk_idx[i],
                    'topk_logits': topk_logits[i],
                })
            return examples

        def __iter__(self):
            import random
            files = list(self.files)
            if self.shuffle:
                random.shuffle(files)
            for f in files:
                data = torch.load(f, map_location='cpu', weights_only=True)
                examples = self._parse_shard(data)
                if self.shuffle:
                    random.shuffle(examples)
                for ex in examples:
                    yield ex

        def __len__(self):
            return len(self.files) * 100  # Approximate

    dataset = ShardedKDDataset(args.kd_cache_dir, rank, world_size, shuffle=True)
    log(f"  Rank {rank}: {len(dataset.files)}/{len(dataset.all_files)} shard files")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_fn, drop_last=True,
    )

    effective_batch = args.batch_size * args.accumulation_steps * world_size
    log(f"  Effective batch: {args.batch_size} x {args.accumulation_steps} x {world_size} = {effective_batch}")

    # =========================================================================
    # 5. Wandb (master only)
    # =========================================================================
    use_wandb = args.wandb and is_master
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or f"recovery_lora_multi_r{args.recovery_r}",
                config=vars(args),
                resume="allow",  # Allow resuming if run was interrupted
            )
            # Define step metric to avoid "step less than current" warnings when resuming
            wandb.define_metric("*", step_metric="step", step_sync=False)
        except ImportError:
            use_wandb = False

    # =========================================================================
    # 6. XLA Warmup
    # =========================================================================
    xm.rendezvous("before_warmup")
    log("\n[XLA] Warmup: precompiling graphs...")

    model.train()
    optimizer.zero_grad()

    warmup_iter = iter(dataloader)
    warmup_batch = next(warmup_iter)

    # Forward
    log("  [warmup] forward...", flush=True)
    with torch.amp.autocast(device_type='xla', dtype=compute_dtype):
        warmup_loss = compute_kd_loss_batch(
            model, warmup_batch, device, args.temperature,
            no_grad=False, hard_top1_weight=args.hard_top1,
        )
    torch_xla.sync()
    log("  [warmup] forward done")

    # Backward
    log("  [warmup] backward...", flush=True)
    warmup_loss.backward()
    del warmup_loss
    torch_xla.sync()
    log("  [warmup] backward done")

    # Optimizer
    log("  [warmup] optimizer...", flush=True)
    xm.reduce_gradients(optimizer)
    xm.optimizer_step(optimizer)
    torch_xla.sync()
    log("  [warmup] optimizer done")

    # Reset
    optimizer.zero_grad()
    optimizer.state.clear()
    del warmup_iter, warmup_batch
    gc.collect()

    log("  [warmup] XLA compilation complete")

    # =========================================================================
    # 7. Training loop
    # =========================================================================
    log(f"\n[Training] Steps: {args.max_steps}, LR: {args.lr} → {args.lr * args.min_lr_ratio}")

    model.train()
    optimizer.zero_grad(set_to_none=False)  # TPU: avoid grad None->Tensor recompile

    step = 0
    optimizer_step = 0
    accum_loss = 0.0
    best_loss = float('inf')
    t_start = time.time()
    eta_history = deque(maxlen=5)

    total_micro_steps = args.max_steps * args.accumulation_steps

    epoch = 0
    while step < total_micro_steps:
        epoch += 1
        if is_master and epoch <= 3:
            log(f"  Epoch {epoch}...")

        for batch in dataloader:
            if step >= total_micro_steps:
                break

            # Calculate current hard_top1
            if args.hard_top1_end is not None:
                progress = min(1.0, optimizer_step / max(1, args.max_steps))
                current_hard_top1 = args.hard_top1 + (args.hard_top1_end - args.hard_top1) * progress
            else:
                current_hard_top1 = args.hard_top1

            # Forward
            with torch.amp.autocast(device_type='xla', dtype=compute_dtype):
                loss = compute_kd_loss_batch(
                    model, batch, device, args.temperature,
                    no_grad=False, hard_top1_weight=current_hard_top1,
                )

            if args.accumulation_steps > 1:
                loss = loss / args.accumulation_steps

            accum_loss += loss.detach()
            loss.backward()
            del loss

            # Optimizer step
            if (step + 1) % args.accumulation_steps == 0:
                xm.reduce_gradients(optimizer)

                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)

                xm.optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad(set_to_none=False)
                optimizer_step += 1

                last_loss = accum_loss
                accum_loss = 0.0

                # Logging
                if optimizer_step % args.log_every == 0:
                    torch_xla.sync()
                    loss_val = last_loss.item() if hasattr(last_loss, 'item') else float(last_loss)
                    current_time = time.time()
                    elapsed = current_time - t_start

                    eta_history.append((optimizer_step, current_time))

                    # Smoothed ETA
                    if len(eta_history) >= 2:
                        old_step, old_time = eta_history[0]
                        recent_elapsed = current_time - old_time
                        recent_steps = optimizer_step - old_step
                        if recent_steps > 0:
                            steps_per_sec = recent_steps / recent_elapsed
                            remaining = args.max_steps - optimizer_step
                            eta = remaining / steps_per_sec
                            tok_per_sec = steps_per_sec * effective_batch * args.seq_len
                        else:
                            eta, tok_per_sec = 0, 0
                    else:
                        eta = 0
                        tok_per_sec = (optimizer_step * effective_batch * args.seq_len) / elapsed if elapsed > 0 else 0

                    # Format time
                    def fmt_time(secs):
                        if secs < 3600:
                            return f"{secs/60:.1f}m"
                        return f"{secs/3600:.1f}h"

                    if is_master:
                        lr = optimizer.param_groups[0]['lr']
                        print(f"  Step {optimizer_step}/{args.max_steps}: "
                              f"loss={loss_val:.4f}, lr={lr:.2e}, "
                              f"tok/s={tok_per_sec:.0f}, "
                              f"elapsed={fmt_time(elapsed)}, ETA={fmt_time(eta)}", flush=True)

                        if loss_val < best_loss:
                            best_loss = loss_val
                            print(f"    [Saved best: {best_loss:.4f}]", flush=True)

                    if use_wandb:
                        import wandb
                        wandb.log({
                            'train/loss': loss_val,
                            'train/lr': optimizer.param_groups[0]['lr'],
                            'train/tokens_per_sec': tok_per_sec,
                            'train/step': optimizer_step,
                        }, step=optimizer_step)

                # Save checkpoint
                if args.save_steps > 0 and optimizer_step % args.save_steps == 0:
                    save_path = f"{args.output_dir}/checkpoint_step{optimizer_step}.pt"
                    if is_master:
                        os.makedirs(args.output_dir, exist_ok=True)
                    xm.save(model.state_dict(), save_path)
                    if is_master:
                        print(f"    [Saved: {save_path}]", flush=True)

                        # Cleanup old checkpoints
                        if args.keep_checkpoints > 0:
                            import glob
                            ckpts = sorted(glob.glob(f"{args.output_dir}/checkpoint_step*.pt"),
                                           key=os.path.getmtime)
                            while len(ckpts) > args.keep_checkpoints:
                                old = ckpts.pop(0)
                                try:
                                    os.remove(old)
                                except OSError:
                                    pass

            step += 1
            torch_xla.sync()

    # Final save
    xm.rendezvous('before_final_save')

    elapsed = time.time() - t_start
    if is_master:
        log(f"\n  Training complete: {elapsed/60:.1f} min")

        final_path = f"{args.output_dir}/recovery_lora_final.pt"
        model_cpu = model.cpu().float()
        torch.save(model_cpu.state_dict(), final_path)
        log(f"  Saved: {final_path}")

        # Save config.json if it doesn't exist
        config_path = f"{args.output_dir}/config.json"
        if not os.path.exists(config_path):
            config_data = {
                'version': 'v2',
                'model_id': args.model,
                # Quantization config
                'lut_bits': args.lut_bits,
                'mlp_lut_bits': args.lut_bits,
                'attn_lut_bits': args.attn_lut_bits,
                'scale_rank': args.scale_rank,
                'mlp_scale_rank': args.scale_rank,
                'attn_scale_rank': args.attn_scale_rank,
                # LoRA config
                'lora_r': args.recovery_r,
                'lora_alpha': lora_alpha,
                'lora_mlp_only': args.mlp_only,
                # Training info
                'max_steps': args.max_steps,
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            log(f"  Config saved: {config_path}")

        # Upload to Google Drive if requested
        if args.upload:
            log("\n[Upload] Uploading run to Google Drive...")
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from gdrive_sync import sync_up

                # Default exclude pattern: *checkpoint* (intermediate checkpoints are large)
                exclude_patterns = args.upload_exclude if args.upload_exclude else ['*checkpoint*']

                success = sync_up(
                    local_path=args.output_dir,
                    run_name=None,  # Use output_dir name
                    dry_run=False,
                    is_cache=False,
                    exclude=exclude_patterns,
                    only=None,
                )
                if success:
                    log(f"  Upload complete: {args.output_dir}")
                else:
                    log(f"  Upload failed or skipped (check if Google Drive is mounted)")
            except ImportError as e:
                log(f"  ERROR: Could not import gdrive_sync: {e}")
            except Exception as e:
                log(f"  ERROR during upload: {e}")

        if use_wandb:
            import wandb
            wandb.finish()

    xm.rendezvous('training_done')


def main():
    args = parse_args()

    # Validate
    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    assert os.path.exists(args.kd_cache_dir), f"KD cache not found: {args.kd_cache_dir}"

    if not check_tpu():
        print("ERROR: TPU not available. Use train_recovery_lora.py for CPU/GPU.")
        sys.exit(1)

    if args.num_chips:
        os.environ['TPU_NUM_DEVICES'] = str(args.num_chips)
        print(f"[Recovery LoRA Multi] Using {args.num_chips} TPU chips")
    else:
        print("[Recovery LoRA Multi] Using all available TPU chips")

    import torch_xla.distributed.xla_multiprocessing as xmp

    try:
        xmp.spawn(train_worker, args=(args,), nprocs=None, start_method='spawn')
    except TypeError:
        xmp.spawn(train_worker, args=(args,), nprocs=None)


if __name__ == '__main__':
    main()
