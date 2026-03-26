import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
import torch
from typing import Tuple


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def preprocess(x: torch.Tensor, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375], img_size=1024) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""

    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def prepare_mask(image, target_length=1024):
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    mask = np.array(resize(to_pil_image(image), target_size))
    
    input_mask = torch.as_tensor(mask)
    input_mask = input_mask.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_mask = preprocess(input_mask)

    return input_mask


def postprocess_masks(masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...], img_size=1024) -> torch.Tensor:
      """
      Remove padding and upscale masks to the original image size.

      Arguments:
        masks (torch.Tensor): Batched masks from the mask_decoder,
          in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
          model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
          before resizing for input to the model, in (H, W) format.

      Returns:
        (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
          is given by original_size.
      """
      masks = F.interpolate(
          masks,
          (img_size, img_size),
          mode="bilinear",
          align_corners=False,
      )
      masks = masks[..., : input_size[0], : input_size[1]]
      masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
      return masks