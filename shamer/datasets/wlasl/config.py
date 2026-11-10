from yacs.config import CfgNode as CN

_CN = CN(new_allowed=True)
_CN.DATA_ROOT = "/mnt/datasets/wlasl/wlasl"
_CN.VERSION = "v0.3"


_CN.SUBSET = 100 # TODO: How do I configure this from the outside?
_CN.FRAMES_PER_CLIP = 16
_CN.FRAME_STEP = 2
_CN.STRIDE = 4
_CN.PAD_START = False
_CN.PAD_END = True


def get_defaults() -> CN:
    defaults = _CN.clone()
    defaults.freeze()
    return defaults
