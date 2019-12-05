from .build_enum import BuildEnum
from .printing import format_str
from .file import mkdir, mkdirs
from . import visdom
from . import state
from . import logging
from . import types

import torch


def output_to_image(outputs, reorder_dims=True, numpy=True):
    scale = 255 / 2.
    outputs = outputs.detach()
    if reorder_dims:
        if outputs.dim() == 4:
            outputs = outputs.permute(0, 2, 3, 1)
        else:
            outputs = outputs.permute(1, 2, 0)
    outputs = (outputs + 1) \
        .mul_(scale) \
        .add_(0.5) \
        .clamp_(0, 255) \
        .to('cpu', torch.uint8)
    if numpy:
        outputs = outputs.numpy()
    return outputs

__all__ = [BuildEnum, format_str, mkdir, mkdirs, visdom, state, logging, types, output_to_image]
