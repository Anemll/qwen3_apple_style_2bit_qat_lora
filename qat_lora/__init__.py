# QAT + LoRA utilities

from .ane_qat_linear import AnemllQATLinear, AnemllQuantConfig, replace_linear_with_anemll
from .layer_qat import (
    compute_kd_loss_batch,
    evaluate_kd_loss,
    train_layer,
    train_all_layers,
    KDCacheDataset,
    collate_fn,
    LocalMLPLoss,
    LocalLayerReconstructionLoss,  # Alias for backward compat
    get_mlp_frozen_weights,
    create_frozen_fp_layer,
    is_qat_linear,
)

__all__ = [
    'AnemllQATLinear',
    'AnemllQuantConfig',
    'replace_linear_with_anemll',
    'compute_kd_loss_batch',
    'evaluate_kd_loss',
    'train_layer',
    'train_all_layers',
    'KDCacheDataset',
    'collate_fn',
    'LocalMLPLoss',
    'LocalLayerReconstructionLoss',
    'get_mlp_frozen_weights',
    'create_frozen_fp_layer',
    'is_qat_linear',
]
