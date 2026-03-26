# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
from .predictor import SamPredictor
from .automatic_mask_generator import SamAutomaticMaskGenerator
from .peft_utils import set_lora_train_mode, set_lora_eval_mode, apply_lora2vision_encoder_attention, enable_disable_lora_sam_vision_encoder
from .model_utils import compute_ref_target_cosim