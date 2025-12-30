# QAT + LoRA utilities

from .ane_qat_linear import (
    AnemllQATLinear,
    AnemllQuantConfig,
    replace_linear_with_anemll,
    freeze_model_for_inference,
    unfreeze_model_for_training,
    compute_all_indices,
    snap_all_weights,
    export_quantized_model,
)

# V2: ANE-friendly rank-by-rank forward
from .ane_qat_linear_v2 import (
    AnemllQATLinearV2,
    AnemllQuantConfigV2,
    replace_linear_with_anemll_v2,
    freeze_Q_all,
    freeze_model_for_inference_v2,
    unfreeze_model_for_training_v2,
    set_factored_inference_v2,
    get_inference_mode_v2,
    set_batched_forward_v2,
    snap_model_for_ane_v2,
    convert_model_to_fp16_v2,
    load_v2_checkpoint,  # Proper loading with _Q and _indices
    ste_fp16,  # STE-FP16 rounding for ANE-matching training
)
from .layer_qat import (
    compute_kd_loss_batch,
    evaluate_kd_loss,
    train_layer,
    train_all_layers,
    train_e2e,
    save_checkpoint,
    load_checkpoint,
    KDCacheDataset,
    collate_fn,
    LocalMLPLoss,
    LocalLayerReconstructionLoss,  # Alias for backward compat
    get_mlp_frozen_weights,
    create_frozen_fp_layer,
    is_qat_linear,
)

__all__ = [
    # V1 (original)
    'AnemllQATLinear',
    'AnemllQuantConfig',
    'replace_linear_with_anemll',
    'freeze_model_for_inference',
    'unfreeze_model_for_training',
    'compute_all_indices',
    'snap_all_weights',
    'export_quantized_model',
    # V2 (ANE-friendly rank-by-rank)
    'AnemllQATLinearV2',
    'AnemllQuantConfigV2',
    'replace_linear_with_anemll_v2',
    'freeze_Q_all',
    'freeze_model_for_inference_v2',
    'unfreeze_model_for_training_v2',
    'set_factored_inference_v2',
    'get_inference_mode_v2',
    'set_batched_forward_v2',
    'snap_model_for_ane_v2',
    'convert_model_to_fp16_v2',
    'load_v2_checkpoint',
    'ste_fp16',
    # Training utilities
    'compute_kd_loss_batch',
    'evaluate_kd_loss',
    'train_layer',
    'train_all_layers',
    'train_e2e',
    'save_checkpoint',
    'load_checkpoint',
    'KDCacheDataset',
    'collate_fn',
    'LocalMLPLoss',
    'LocalLayerReconstructionLoss',
    'get_mlp_frozen_weights',
    'create_frozen_fp_layer',
    'is_qat_linear',
]
