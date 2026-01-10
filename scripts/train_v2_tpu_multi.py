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
import importlib.util
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

# Check for TPU - IMPORTANT: Don't call any XLA functions before xmp.spawn()!
def check_tpu():
    """Check if torch_xla is installed without importing it (avoids PJRT/XLA init)."""
    return importlib.util.find_spec("torch_xla") is not None


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
    parser.add_argument('--hard-top1', type=float, default=0.2, help='Hard label top-1 weight (start)')
    parser.add_argument('--hard-top1-end', type=float, default=None,
                        help='Hard label top-1 weight at end (for annealing). If set, decays from --hard-top1 to this.')
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--constant-lr', action='store_true')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1,
                        help='Minimum LR as ratio of peak LR for cosine annealing (default: 0.1)')
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--keep-checkpoints', type=int, default=0,
                        help='Keep only the last N checkpoints (0=keep all). Useful for long runs.')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (default: 1.0, 0=disable)')
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

    # Training mode
    parser.add_argument('--mlp-only', action='store_true',
                        help='Train only MLP layers (gate/up/down_proj), freeze attention (q/k/v/o_proj)')
    parser.add_argument('--freeze-mags', action='store_true',
                        help='Freeze rank_magnitude (snap to FP16 values). Train only A and B.')
    parser.add_argument('--freeze-mags-mlp', action='store_true',
                        help='Freeze rank_magnitude for MLP layers only')

    # Multi-chip options
    parser.add_argument('--num-chips', type=int, default=None,
                        help='Number of TPU chips to use (default: all available)')

    # XLA compilation cache
    parser.add_argument('--xla-cache-dir', type=str, default=None,
                        help='XLA persistent cache directory (shared across ranks; may be unsupported on some PJRT/libtpu builds)')

    # Precision
    parser.add_argument('--bf16', action='store_true',
                        help='Use pure BF16 (saves ~50%% HBM, default is FP32 master weights)')

    return parser.parse_args()


def train_worker(index, args):
    """Training worker function - runs on each TPU chip."""
    import time
    import gc
    import traceback
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

    def log(msg, end="\n", flush=True):
        if is_master:
            print(msg, end=end, flush=flush)

    # Print from ALL ranks for debugging
    def log_all(msg):
        print(f"[Rank {rank}] {msg}", flush=True)

    log_all(f"Worker started: device={device}, world_size={world_size}")

    # Check if PT_XLA_DEBUG is enabled
    if os.environ.get('PT_XLA_DEBUG', '0') == '1':
        log_all("PT_XLA_DEBUG=1 - Extended XLA debugging enabled")

    try:
        _train_worker_impl(index, args, device, rank, world_size, is_master, log, log_all)
    except Exception as e:
        log_all(f"FATAL ERROR: {e}")
        traceback.print_exc()
        raise


def _train_worker_impl(index, args, device, rank, world_size, is_master, log, log_all):
    """Actual training implementation (wrapped for error handling)."""
    import time
    import gc
    import math
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met  # XLA metrics for compilation monitoring
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    from qat_lora import (
        AnemllQuantConfigV2,
        replace_linear_with_anemll_v2,
        freeze_Q_all,
    )
    from qat_lora.layer_qat import KDCacheDataset, compute_kd_loss_batch, collate_fn

    # Helper for timestamped checkpoints
    import time as time_module
    t_checkpoint = time_module.time()
    def checkpoint(msg):
        nonlocal t_checkpoint
        elapsed = time_module.time() - t_checkpoint
        log_all(f"CHECKPOINT [{elapsed:.1f}s]: {msg}")
        t_checkpoint = time_module.time()

    log(f"\n[TPU Multi-chip] Starting training on {world_size} chips")
    log(f"  Rank {rank}: device={device}")
    log(f"  Precision: Mixed (FP32 weights + BF16 compute)")
    checkpoint("Worker init complete")

    # Config presets
    CONFIG_PRESETS = {
        'q2a4': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 8},
        'q4a4': {'mlp_lut': 16, 'mlp_rank': 4, 'attn_lut': 16, 'attn_rank': 4},
        'q4a4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},
        'q4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},
        'q2a2': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 4, 'attn_rank': 32},
    }
    preset = CONFIG_PRESETS[args.config]

    # Mixed precision: FP32 master weights + BF16 compute via autocast
    # Precision mode: BF16 saves ~50% HBM but may have slightly less stable training
    if args.bf16:
        train_dtype = torch.bfloat16  # Pure BF16 (saves HBM)
        compute_dtype = torch.bfloat16
        if is_master:
            print("  Precision: Pure BF16 (--bf16 flag)")
    else:
        train_dtype = torch.float32  # FP32 for optimizer updates (stability)
        compute_dtype = torch.bfloat16  # BF16 for forward/backward (speed)
        if is_master:
            print("  Precision: Mixed (FP32 master weights + BF16 compute)")

    # Load model - serialize to avoid all workers downloading simultaneously
    log("\n[1/4] Loading model...")
    t0 = time.time()

    # Rank 0 loads first (downloads if needed), others wait then load from cache
    if is_master:
        checkpoint("Rank 0 loading model (others waiting)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=train_dtype,  # FP32 for stability
            trust_remote_code=True,
        )
        checkpoint("Rank 0 model loaded")

    # Barrier - wait for rank 0 to finish downloading
    xm.rendezvous("model_download")

    if not is_master:
        checkpoint("Loading model from cache...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=train_dtype,
            trust_remote_code=True,
        )
        checkpoint("Model loaded from cache")

    # Replace with V2 layers
    v2_mlp_config = AnemllQuantConfigV2(
        lut_size=preset['mlp_lut'],
        scale_rank=preset['mlp_rank'],
        group_size=args.group_size,
        force_positive_scales=False,
        magnitude_activation='identity',
        use_ste_fp16=True,  # Enable FP16 emulation for ANE compatibility
    )
    v2_attn_config = AnemllQuantConfigV2(
        lut_size=preset['attn_lut'],
        scale_rank=preset['attn_rank'],
        group_size=args.group_size,
        force_positive_scales=False,
        magnitude_activation='identity',
        use_ste_fp16=True,  # Enable FP16 emulation for ANE compatibility
    )

    # Rank 0 does layer replacement (SVD) once, saves state_dict for others to load
    # Other ranks still need to create the V2 structure, but load weights from rank 0
    # (they do SVD but it gets overwritten - future optimization: add skip_init option)
    # Use output_dir for temp file (avoids /tmp permission issues on TPU VMs)
    os.makedirs(args.output_dir, exist_ok=True)
    v2_cache_path = os.path.join(args.output_dir, ".v2_model_state_temp.pt")

    if is_master:
        do_skip_init = args.v2_checkpoint is not None
        checkpoint(
            "Rank 0: Replacing layers with V2 quantized layers "
            + ("(skip_init=True)..." if do_skip_init else "(SVD init)...")
        )
        replace_linear_with_anemll_v2(
            model,
            mlp_config=v2_mlp_config,
            attn_config=v2_attn_config,
            quantize_attn=True,
            quantize_lm_head=False,
            skip_init=do_skip_init,
        )
        checkpoint("Rank 0: V2 layer replacement complete")

        # Load user checkpoint if provided (before saving for others)
        if args.v2_checkpoint:
            state_dict = torch.load(args.v2_checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            del state_dict  # Free memory
            log(f"  Loaded checkpoint: {args.v2_checkpoint}")

        # Save for other ranks
        checkpoint("Rank 0: Saving V2 model state for other ranks...")
        torch.save(model.state_dict(), v2_cache_path)
        checkpoint("Rank 0: V2 state saved")

    # Wait for rank 0 to finish
    xm.rendezvous("v2_model_ready")

    if not is_master:
        # Other ranks: create V2 structure (skip SVD) then load rank 0's weights
        # skip_init=True makes this ~10x faster (no SVD computation)
        checkpoint(f"Rank {rank}: Creating V2 structure (skip_init=True)...")
        replace_linear_with_anemll_v2(
            model,
            mlp_config=v2_mlp_config,
            attn_config=v2_attn_config,
            quantize_attn=True,
            quantize_lm_head=False,
            skip_init=True,  # Skip SVD - will load from rank 0
        )
        checkpoint(f"Rank {rank}: Loading V2 weights from rank 0...")
        state_dict = torch.load(v2_cache_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        del state_dict  # Free memory
        gc.collect()
        checkpoint(f"Rank {rank}: V2 weights loaded")

    # Freeze Q BEFORE moving to device (avoids XLA compilations on TPU)
    freeze_Q_all(model)
    checkpoint("Model frozen")

    # Freeze attention layers if --mlp-only
    if args.mlp_only:
        attn_proj_names = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
        attn_frozen = 0
        for name, module in model.named_modules():
            if type(module).__name__ in ('AnemllQATLinear', 'AnemllQATLinearV2'):
                if any(proj in name for proj in attn_proj_names):
                    for p in module.parameters():
                        p.requires_grad = False
                        attn_frozen += p.numel()
        if is_master:
            print(f"  MLP-only mode: frozen {attn_frozen:,} attention params")

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
        if is_master:
            print(f"  Snapped & frozen {mags_frozen} rank_magnitude tensors")

    # Move to device
    checkpoint("Moving model to device...")
    model.to(device=device, dtype=train_dtype)
    gc.collect()  # Free CPU memory after move
    checkpoint("Model on device")
    log(f"  Model loaded ({time.time()-t0:.1f}s)")

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
        min_lr_ratio = args.min_lr_ratio

        def lr_lambda(step):
            if step < args.warmup_steps:
                return float(step) / float(max(1, args.warmup_steps))
            if not args.constant_lr:
                progress = float(step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)

    # Dataset: streaming with file sharding across workers (no preload = less memory)
    log("\n[3/4] Loading dataset (streaming mode)...")

    from torch.utils.data import IterableDataset
    from pathlib import Path

    class ShardedKDDataset(IterableDataset):
        """Streaming dataset that shards cache files across workers."""
        def __init__(self, cache_dir, rank, world_size, shuffle=True):
            self.cache_dir = Path(cache_dir)
            self.all_files = sorted(self.cache_dir.glob('*.pt'))
            # Shard files: rank 0 gets files [0, world_size, 2*world_size, ...]
            self.files = [f for i, f in enumerate(self.all_files) if i % world_size == rank]
            self.shuffle = shuffle
            self.rank = rank
            self.world_size = world_size

        def _parse_shard(self, data):
            """Parse shard into list of examples."""
            ids = data['input_ids']
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            topk_idx = data['topk_idx']
            if topk_idx.dim() == 2:
                topk_idx = topk_idx.unsqueeze(0)
            topk_logits = data['topk_logits']
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
            files = list(self.files)
            if self.shuffle:
                import random
                random.shuffle(files)
            for f in files:
                data = torch.load(f, map_location='cpu', weights_only=True)
                examples = self._parse_shard(data)
                if self.shuffle:
                    import random
                    random.shuffle(examples)
                for ex in examples:
                    yield ex

        def __len__(self):
            # Approximate: assume equal distribution across files
            return len(self.files) * 100  # Rough estimate

    dataset = ShardedKDDataset(args.cache_dir, rank, world_size, shuffle=True)
    log(f"  Rank {rank}: streaming from {len(dataset.files)}/{len(dataset.all_files)} shard files")
    checkpoint("Dataset created (streaming)")

    # No sampler needed - sharding is built into dataset
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
    )
    checkpoint("DataLoader created")

    # Effective batch size
    effective_batch = args.batch_size * args.accumulation_steps * world_size
    total_tokens = effective_batch * 128 * args.max_steps  # Assume seq_len=128
    log(f"  Effective batch: {args.batch_size} x {args.accumulation_steps} x {world_size} = {effective_batch}")
    log(f"  Total tokens: {total_tokens/1e6:.1f}M")

    # WandB (master only)
    checkpoint("Before WandB init")
    use_wandb = args.wandb and is_master
    if use_wandb:
        try:
            import wandb
            log("  Initializing WandB...")
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
                resume="allow",  # Allow resuming if run was interrupted
            )
            log("  WandB initialized")
        except ImportError:
            use_wandb = False
    checkpoint("After WandB init")

    # =========================================================================
    # XLA WARMUP: Precompile all graphs (forward + backward + optimizer)
    # This avoids compilation delays during actual training
    # =========================================================================
    # CRITICAL: Sync all ranks before warmup to ensure everyone is ready
    # Without this, rank 0 may start warmup while others are still loading
    xm.rendezvous("before_warmup")
    checkpoint("All ranks synchronized, starting warmup")

    # Helper to log HBM at each stage
    def log_hbm(stage_name):
        try:
            mem = xm.get_memory_info(device)
            if "bytes_used" in mem:
                used_gb = mem["bytes_used"] / 1e9
                total_gb = mem.get("bytes_limit", 0) / 1e9
                free_gb = total_gb - used_gb
                log(f"  [HBM {stage_name}] {used_gb:.1f}/{total_gb:.1f} GB ({100*used_gb/total_gb:.0f}%) - {free_gb:.1f} GB free")
                if use_wandb:
                    wandb.log({
                        f'warmup/{stage_name}_used_gb': used_gb,
                        f'warmup/{stage_name}_free_gb': free_gb,
                    }, step=0)
                return used_gb
        except Exception as e:
            log(f"  [HBM] Could not get memory info: {e}")
        return 0

    log_hbm("before_warmup")

    log("\n[XLA] Warmup: precompiling forward + backward + optimizer...")
    checkpoint("Starting XLA warmup")

    model.train()
    optimizer.zero_grad()

    # Get one batch for warmup
    warmup_iter = iter(dataloader)
    warmup_batch = next(warmup_iter)

    # Forward pass (use autocast for BF16 compute even with FP32 weights)
    log("  [warmup] forward...", end=" ")
    with torch.amp.autocast(device_type='xla', dtype=compute_dtype):
        warmup_loss = compute_kd_loss_batch(
            model, warmup_batch, device, args.temperature,
            no_grad=False,
            hard_top1_weight=args.hard_top1,
            hard_full_weight=0.0,
        )
    torch_xla.sync()
    log("done")
    checkpoint("Warmup forward complete")
    log_hbm("after_forward")

    # Backward pass
    log("  [warmup] backward...", end=" ", flush=True)
    warmup_loss.backward()
    del warmup_loss  # Free memory before sync
    log("graph built, syncing...", end=" ", flush=True)
    torch_xla.sync()
    log("done")
    checkpoint("Warmup backward complete")
    log_hbm("after_backward")

    # Gradient sync + optimizer step (this is the key missing piece!)
    log("  [warmup] optimizer...", end=" ")
    xm.reduce_gradients(optimizer)
    xm.optimizer_step(optimizer)
    torch_xla.sync()
    log("done")
    checkpoint("Warmup optimizer complete")
    log_hbm("after_optimizer")

    # Reset optimizer state (don't want warmup to affect training)
    optimizer.zero_grad()
    optimizer.state.clear()

    # Reset dataloader for actual training
    del warmup_iter, warmup_batch
    gc.collect()
    log_hbm("after_cleanup")

    log("  [warmup] XLA compilation complete. Training should be fast now.")
    checkpoint("XLA warmup complete")

    # Training loop
    log("\n[4/4] Training...")
    checkpoint("Starting training loop")
    log(f"  Steps: {args.max_steps}, Warmup: {args.warmup_steps}")
    log(f"  LR: {args.lr} → {args.lr * args.min_lr_ratio} (cosine, min_lr_ratio={args.min_lr_ratio})")
    if args.hard_top1_end is not None:
        log(f"  Hard-top1: {args.hard_top1} → {args.hard_top1_end} (annealing)")
    else:
        log(f"  Hard-top1: {args.hard_top1}")
    if args.keep_checkpoints > 0:
        log(f"  Checkpoints: save every {args.save_steps}, keep last {args.keep_checkpoints}")
    if args.clip_grad_norm > 0:
        log(f"  Gradient clipping: max_norm={args.clip_grad_norm}")

    model.train()
    optimizer.zero_grad()

    step = 0
    optimizer_step = 0
    accum_loss = 0.0  # Accumulated loss for current optimizer step
    last_loss = 0.0   # Last logged loss value
    t_start = time.time()

    # Rolling window for smoothed ETA/throughput (avoids XLA compile time skew)
    from collections import deque
    eta_history = deque(maxlen=5)  # Track (optimizer_step, time) pairs

    total_micro_steps = args.max_steps * args.accumulation_steps
    log(f"  Total micro-steps: {total_micro_steps}")
    log(f"  Streaming from {len(dataset.files)} shard files (rank {rank})")

    epoch = 0
    while step < total_micro_steps:
        # For streaming dataset, re-create iterator each epoch (shuffles files)
        epoch += 1
        log(f"  Starting epoch {epoch}...")

        for batch_idx, batch in enumerate(dataloader):
            if step >= total_micro_steps:
                break

            if step == 0:
                checkpoint("First batch received")
                log(f"  First batch received, starting XLA compilation...")

            # Heartbeat for steps 4-8 (post-optimizer debugging)
            if 4 <= step <= 8 and is_master:
                # Extra sync before forward to isolate any pending work
                print(f"  [heartbeat] step={step} pre-forward sync...", flush=True)
                t_presync = time.time()
                torch_xla.sync()
                presync_elapsed = time.time() - t_presync
                print(f"  [heartbeat] step={step} pre-forward sync done ({presync_elapsed:.1f}s), starting forward...", flush=True)
                if step == 4:
                    print("  [NOTE] If step 4 forward is slow (~60-90s), this may be XLA recompiling due to optimizer state.", flush=True)
                    print("         This is expected on first optimizer step. Step 5+ should be fast.", flush=True)

            # Forward pass with autocast for BF16 compute (saves memory vs FP32 activations)
            if step == 0:
                checkpoint("Starting first forward pass...")

            # Calculate current hard_top1 (with optional annealing)
            if args.hard_top1_end is not None:
                # Linear decay from hard_top1 to hard_top1_end
                progress = min(1.0, optimizer_step / max(1, args.max_steps))
                current_hard_top1 = args.hard_top1 + (args.hard_top1_end - args.hard_top1) * progress
            else:
                current_hard_top1 = args.hard_top1

            # Pass debug_step for steps 4-8 to help debug post-optimizer hang
            dbg_step = step if (4 <= step <= 8 and is_master) else -1
            with torch.amp.autocast(device_type='xla', dtype=compute_dtype):
                loss = compute_kd_loss_batch(
                    model, batch, device, args.temperature,
                    no_grad=False,
                    hard_top1_weight=current_hard_top1,
                    hard_full_weight=0.0,  # Disabled for TPU
                    debug_step=dbg_step,
                )
            if step == 0:
                checkpoint("First forward pass complete")

            # Heartbeat for steps 4-8
            if 4 <= step <= 8 and is_master:
                print(f"  [heartbeat] step={step} forward done, starting backward...", flush=True)

            # Scale for accumulation
            if args.accumulation_steps > 1:
                loss = loss / args.accumulation_steps

            # Track loss for logging (before backward, while tensor is valid)
            accum_loss += loss.detach()

            # Backward
            if step == 0:
                checkpoint("Starting first backward pass...")
            loss.backward()

            # Free loss tensor to reduce HBM pressure during accumulation
            del loss

            if step == 0:
                checkpoint("First backward pass complete")

            # Heartbeat for steps 4-8
            if 4 <= step <= 8 and is_master:
                print(f"  [heartbeat] step={step} backward done", flush=True)

            # Print XLA compilation metrics after first pass
            if step == 0 and is_master:
                log("  XLA Compilation Metrics:")
                try:
                    def _fmt_metric(metric):
                        if not metric:
                            return None
                        count, total_ns, _samples = metric
                        return f"{count} calls, {total_ns/1e9:.2f}s total"

                    compile_time = _fmt_metric(met.metric_data('CompileTime'))
                    if compile_time:
                        log(f"    CompileTime: {compile_time}")
                    execute_time = _fmt_metric(met.metric_data('ExecuteTime'))
                    if execute_time:
                        log(f"    ExecuteTime: {execute_time}")
                except Exception as e:
                    log(f"    (metrics unavailable: {e})")

            # Optimizer step every accumulation_steps
            if (step + 1) % args.accumulation_steps == 0:
                if step < args.accumulation_steps:
                    checkpoint("First optimizer step - gradient sync...")
                # Gradient sync across chips
                xm.reduce_gradients(optimizer)
                if step < args.accumulation_steps:
                    checkpoint("Gradient sync complete")

                # Gradient clipping for stability
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)

                # Optimizer step
                if step < args.accumulation_steps:
                    checkpoint("Optimizer step starting...")
                xm.optimizer_step(optimizer)
                if step < args.accumulation_steps:
                    checkpoint("Optimizer step complete")

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad()
                optimizer_step += 1

                # Save accumulated loss and reset for next optimizer step
                last_loss = accum_loss
                accum_loss = 0.0

                if optimizer_step == 1:
                    checkpoint("Optimizer step 1 complete")

                # Logging
                if optimizer_step % 20 == 0:
                    # Sync before reading loss to ensure XLA graph is evaluated
                    torch_xla.sync()
                    loss_val = last_loss.item() if hasattr(last_loss, 'item') else last_loss
                    current_time = time.time()
                    elapsed = current_time - t_start

                    # Update rolling window for smoothed ETA/throughput
                    eta_history.append((optimizer_step, current_time))

                    # Calculate smoothed ETA and throughput using rolling window
                    if len(eta_history) >= 2:
                        old_step, old_time = eta_history[0]
                        recent_elapsed = current_time - old_time
                        recent_steps = optimizer_step - old_step
                        if recent_steps > 0 and recent_elapsed > 0:
                            steps_per_sec = recent_steps / recent_elapsed
                            remaining_steps = args.max_steps - optimizer_step
                            eta = remaining_steps / steps_per_sec
                            tok_per_sec = steps_per_sec * effective_batch * 128
                        else:
                            eta = 0
                            tok_per_sec = 0
                    else:
                        # Fallback for first log
                        eta = elapsed / optimizer_step * (args.max_steps - optimizer_step) if optimizer_step > 0 else 0
                        tok_per_sec = (optimizer_step * effective_batch * 128) / elapsed if elapsed > 0 else 0

                    # Format elapsed time
                    elapsed_str = f"{int(elapsed)//60}:{int(elapsed)%60:02d}"

                    if is_master:
                        print(f"  Step {optimizer_step}/{args.max_steps} | "
                              f"Loss: {loss_val:.4f} | "
                              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                              f"Tok/s: {tok_per_sec:.0f} | "
                              f"Time: {elapsed_str} | "
                              f"ETA: {eta/60:.1f}m", flush=True)

                    if use_wandb:
                        log_dict = {
                            'step': optimizer_step,
                            'loss': loss_val,
                            'lr': optimizer.param_groups[0]['lr'],
                            'tokens_per_sec': tok_per_sec,
                        }
                        # Add hard_top1 if annealing
                        if args.hard_top1_end is not None:
                            log_dict['hard_top1'] = current_hard_top1
                        # Add TPU memory stats
                        try:
                            mem = xm.get_memory_info(device)
                            if "kb_total" in mem:
                                log_dict['tpu/memory_used_gb'] = (mem["kb_total"] - mem.get("kb_free", 0)) / 1024 / 1024
                                log_dict['tpu/memory_total_gb'] = mem["kb_total"] / 1024 / 1024
                            elif "bytes_used" in mem:
                                log_dict['tpu/memory_used_gb'] = mem["bytes_used"] / 1e9
                                log_dict['tpu/memory_total_gb'] = mem.get("bytes_limit", 0) / 1e9
                        except Exception:
                            pass
                        wandb.log(log_dict)

                # Save checkpoint - ALL workers must call xm.save() for sync, but only master writes
                if args.save_steps > 0 and optimizer_step % args.save_steps == 0:
                    save_path = f"{args.output_dir}/checkpoint_step{optimizer_step}.pt"
                    if is_master:
                        os.makedirs(args.output_dir, exist_ok=True)
                    # xm.save MUST be called by all workers (handles sync internally)
                    # Only master actually writes the file
                    xm.save(model.state_dict(), save_path)
                    if is_master:
                        print(f"  Saved: {save_path}", flush=True)

                        # Clean up old checkpoints if keep_checkpoints is set
                        if args.keep_checkpoints > 0:
                            import glob
                            ckpt_pattern = f"{args.output_dir}/checkpoint_step*.pt"
                            ckpts = sorted(glob.glob(ckpt_pattern), key=os.path.getmtime)
                            while len(ckpts) > args.keep_checkpoints:
                                old_ckpt = ckpts.pop(0)
                                try:
                                    os.remove(old_ckpt)
                                    print(f"  [Removed old: {os.path.basename(old_ckpt)}]", flush=True)
                                except OSError:
                                    pass

            step += 1
            # Use new API (torch_xla.sync) instead of deprecated xm.mark_step()
            t_sync_start = time.time()
            torch_xla.sync()
            sync_time = time.time() - t_sync_start

            # Log XLA metrics for first 12 steps OR when sync is slow (>10s = likely recompile)
            if is_master and (step <= 12 or sync_time > 10.0):
                try:
                    compile_data = met.metric_data('CompileTime')
                    if compile_data:
                        n_compiles, total_ns, _ = compile_data
                        print(f"  [XLA] step={step}: {n_compiles} compiles, {total_ns/1e9:.2f}s total, sync={sync_time:.1f}s", flush=True)
                except Exception:
                    pass

            if step <= args.accumulation_steps:
                checkpoint("XLA step sync complete")

            # Only log steps during warmup or when sync is slow (likely recompile)
            if is_master and (step <= 12 or sync_time > 1.0):
                elapsed = time.time() - t_start
                print(f"  [step {step}] opt={optimizer_step}, sync={sync_time:.1f}s, t={elapsed:.1f}s", flush=True)

    # Sync all workers before final save
    xm.rendezvous('before_final_save')

    # Final save - only master saves
    elapsed = time.time() - t_start
    if is_master:
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

    # Check TPU (import only - no XLA function calls!)
    if not check_tpu():
        print("ERROR: TPU not available. Use train_v2_simple.py for CPU/GPU.")
        sys.exit(1)

    # Set number of devices via environment variable (new torch_xla API)
    if args.num_chips:
        os.environ['TPU_NUM_DEVICES'] = str(args.num_chips)
        print(f"[TPU Multi-chip] Limiting to {args.num_chips} chips via TPU_NUM_DEVICES")
    else:
        print("[TPU Multi-chip] Using all available TPU chips")

    # Set up XLA persistent cache (shared across all ranks)
    # This allows compiled graphs to be reused, saving compilation time on restart
    if args.xla_cache_dir:
        os.makedirs(args.xla_cache_dir, exist_ok=True)
        os.environ['XLA_PERSISTENT_CACHE_PATH'] = args.xla_cache_dir
        print(f"[TPU Multi-chip] XLA cache: {args.xla_cache_dir}")

    # Launch with xmp.spawn - nprocs=None uses all available devices
    import torch_xla.distributed.xla_multiprocessing as xmp

    # Prefer `spawn` start method to avoid inheriting PJRT/XLA runtime state (can crash on restart).
    try:
        xmp.spawn(train_worker, args=(args,), nprocs=None, start_method='spawn')
    except TypeError:
        xmp.spawn(train_worker, args=(args,), nprocs=None)


if __name__ == '__main__':
    main()
