from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS, UDA, MTL,
                      build_backbone, build_head, build_loss, build_segmentor, build_multi_task)
from .decode_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
from .uda import *  # noqa: F401,F403

__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'UDA', 'MTL'
    'build_backbone', 'build_head', 'build_loss', 'build_segmentor', 'build_multi_task'
]
